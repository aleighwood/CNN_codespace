`timescale 1ns/1ps

module tb_line_buffer_3x3;
    localparam int DATA_W = 8;
    localparam int IMG_W = 5;
    localparam int IMG_H = 5;
    localparam int MAX_W = 8;
    localparam int MAX_H = 8;

    logic clk;
    logic rst_n;

    logic in_valid;
    logic in_ready;
    logic signed [DATA_W-1:0] in_data;

    logic start;
    logic [3:0] cfg_img_h;
    logic [3:0] cfg_img_w;
    logic [3:0] cfg_stride;

    logic out_valid;
    logic out_ready;
    logic signed [DATA_W*9-1:0] window_flat;
    logic [3:0] out_row;
    logic [3:0] out_col;

    int i;

    line_buffer_3x3 #(
        .DATA_W(DATA_W),
        .MAX_IMG_W(MAX_W),
        .MAX_IMG_H(MAX_H)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_data(in_data),
        .start(start),
        .cfg_img_h(cfg_img_h),
        .cfg_img_w(cfg_img_w),
        .cfg_stride(cfg_stride),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .window_flat(window_flat),
        .out_row(out_row),
        .out_col(out_col)
    );

    always #5 clk = ~clk;

    task automatic run_stream(input bit start_with_first);
        begin
            $display("=== SCENARIO %s ===", start_with_first ? "B (start w/ first data)" : "A (start before data)");
            // Pulse start, optionally alongside first data.
            start <= 1'b1;
            if (!start_with_first) begin
                in_valid <= 1'b0;
                in_data <= '0;
                @(posedge clk);
            end
            start <= 1'b0;

            for (i = 0; i < IMG_H*IMG_W; i = i + 1) begin
                in_valid <= 1'b1;
                in_data <= i[DATA_W-1:0];
                @(posedge clk);
            end
            in_valid <= 1'b0;
            in_data <= '0;
            repeat (10) @(posedge clk);
        end
    endtask

    always @(posedge clk) begin
        #1;
        if (out_valid && out_ready) begin
            $display("OUT row=%0d col=%0d br=%0d", out_row, out_col,
                     $signed(window_flat[DATA_W-1:0]));
        end
    end

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        in_valid = 1'b0;
        in_data = '0;
        start = 1'b0;
        out_ready = 1'b1;
        cfg_img_h = IMG_H[3:0];
        cfg_img_w = IMG_W[3:0];
        cfg_stride = 4'd2;

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        run_stream(1'b0);

        // Reset between scenarios.
        rst_n = 1'b0;
        repeat (2) @(posedge clk);
        rst_n = 1'b1;
        @(posedge clk);

        run_stream(1'b1);

        $finish;
    end
endmodule
