module line_buffer_3x3 #(
    parameter int DATA_W = 8,
    parameter int MAX_IMG_W = 224,
    parameter int MAX_IMG_H = 224,
    parameter int COL_W = (MAX_IMG_W <= 1) ? 1 : $clog2(MAX_IMG_W),
    parameter int ROW_W = (MAX_IMG_H <= 1) ? 1 : $clog2(MAX_IMG_H)
) (
    input  logic clk,
    input  logic rst_n,

    input  logic in_valid,
    output logic in_ready,
    input  logic signed [DATA_W-1:0] in_data,

    input  logic start,
    input  logic [ROW_W-1:0] cfg_img_h,
    input  logic [COL_W-1:0] cfg_img_w,
    input  logic [ROW_W-1:0] cfg_stride,

    output logic out_valid,
    input  logic out_ready,
    output logic signed [DATA_W*9-1:0] window_flat,
    output logic [ROW_W-1:0] out_row,
    output logic [COL_W-1:0] out_col
);
    localparam int PTR_W = COL_W;

    logic signed [DATA_W-1:0] line1_mem [0:MAX_IMG_W-1];
    logic signed [DATA_W-1:0] line2_mem [0:MAX_IMG_W-1];

    logic [PTR_W-1:0] col;
    logic [ROW_W-1:0] row;
    logic [PTR_W-1:0] ptr;

    logic signed [DATA_W-1:0] row0_shift0;
    logic signed [DATA_W-1:0] row0_shift1;
    logic signed [DATA_W-1:0] row1_shift0;
    logic signed [DATA_W-1:0] row1_shift1;
    logic signed [DATA_W-1:0] row2_shift0;
    logic signed [DATA_W-1:0] row2_shift1;

    logic signed [DATA_W-1:0] line1_val;
    logic signed [DATA_W-1:0] line2_val;

    logic window_valid;
    logic stride_hit;
    integer i;
    logic [ROW_W-1:0] stride_eff;

    assign line1_val = line1_mem[ptr];
    assign line2_val = line2_mem[ptr];

    assign in_ready = !out_valid || out_ready;

    assign stride_eff = (cfg_stride == 0) ? {{(ROW_W-1){1'b0}}, 1'b1} : cfg_stride;
    assign window_valid = (row >= 2) && (col >= 2);
    assign stride_hit = ((row % stride_eff) == 0) && ((col % stride_eff) == 0);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            col <= '0;
            row <= '0;
            ptr <= '0;
            out_valid <= 1'b0;
            window_flat <= '0;
            out_row <= '0;
            out_col <= '0;
            row0_shift0 <= '0;
            row0_shift1 <= '0;
            row1_shift0 <= '0;
            row1_shift1 <= '0;
            row2_shift0 <= '0;
            row2_shift1 <= '0;
        end else begin
            if (start) begin
                col <= '0;
                row <= '0;
                ptr <= '0;
                out_valid <= 1'b0;
                window_flat <= '0;
                out_row <= '0;
                out_col <= '0;
                row0_shift0 <= '0;
                row0_shift1 <= '0;
                row1_shift0 <= '0;
                row1_shift1 <= '0;
                row2_shift0 <= '0;
                row2_shift1 <= '0;
                for (i = 0; i < MAX_IMG_W; i = i + 1) begin
                    line1_mem[i] <= '0;
                    line2_mem[i] <= '0;
                end
            end

            if (out_valid && out_ready) begin
                out_valid <= 1'b0;
            end

            if (!start && in_valid && in_ready) begin
                window_flat <= {
                    row2_shift1, row2_shift0, line2_val,
                    row1_shift1, row1_shift0, line1_val,
                    row0_shift1, row0_shift0, in_data
                };

                out_row <= (row >= 2) ? (row - 2) : '0;
                out_col <= (col >= 2) ? (col - 2) : '0;
                out_valid <= window_valid && stride_hit;

                row2_shift1 <= row2_shift0;
                row2_shift0 <= line2_val;
                row1_shift1 <= row1_shift0;
                row1_shift0 <= line1_val;
                row0_shift1 <= row0_shift0;
                row0_shift0 <= in_data;

                line1_mem[ptr] <= in_data;
                line2_mem[ptr] <= line1_val;

                if (col == cfg_img_w - 1'b1) begin
                    col <= '0;
                    ptr <= '0;
                    if (row == cfg_img_h - 1'b1) begin
                        row <= '0;
                    end else begin
                        row <= row + 1'b1;
                    end
                end else begin
                    col <= col + 1'b1;
                    ptr <= ptr + 1'b1;
                end
            end
        end
    end
endmodule
