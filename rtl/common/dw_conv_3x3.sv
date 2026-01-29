module dw_conv_3x3 #(
    parameter int DATA_W = 8,
    parameter int ACC_W = 32
) (
    input  logic clk,
    input  logic rst_n,

    input  logic in_valid,
    output logic in_ready,
    input  logic signed [DATA_W*9-1:0] window_flat,
    input  logic signed [DATA_W*9-1:0] weight_flat,

    output logic out_valid,
    input  logic out_ready,
    output logic signed [ACC_W-1:0] out_acc
);
    localparam int WIN_ELEMS = 9;

    logic signed [ACC_W-1:0] sum_comb;
    integer i;

    always_comb begin
        sum_comb = '0;
        for (i = 0; i < WIN_ELEMS; i = i + 1) begin
            sum_comb += $signed(window_flat[i*DATA_W +: DATA_W]) *
                        $signed(weight_flat[i*DATA_W +: DATA_W]);
        end
    end

    assign in_ready = !out_valid || out_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_acc <= '0;
        end else begin
            if (out_valid && out_ready) begin
                out_valid <= 1'b0;
            end

            if (in_valid && in_ready) begin
                out_acc <= sum_comb;
                out_valid <= 1'b1;
            end
        end
    end
endmodule
