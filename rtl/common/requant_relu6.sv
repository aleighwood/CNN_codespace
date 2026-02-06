module requant_relu6 #(
    parameter int DATA_W = 8,
    parameter int ACC_W = 32,
    parameter int MUL_W = 32,
    parameter int BIAS_W = 32,
    parameter int SHIFT_W = 6
) (
    input  logic clk,
    input  logic rst_n,

    input  logic in_valid,
    output logic in_ready,
    input  logic signed [ACC_W-1:0] in_acc,

    input  logic signed [MUL_W-1:0] mul,
    input  logic signed [BIAS_W-1:0] bias,
    input  logic [SHIFT_W-1:0] shift,
    input  logic signed [DATA_W-1:0] relu6_max,
    input  logic relu6_en,

    output logic out_valid,
    input  logic out_ready,
    output logic signed [DATA_W-1:0] out_q
);
    localparam int SCALE_W = ACC_W + MUL_W;
    localparam int Q_MIN = -(1 << (DATA_W - 1));
    localparam int Q_MAX = (1 << (DATA_W - 1)) - 1;

    logic signed [SCALE_W-1:0] mult;
    logic signed [SCALE_W:0] scaled;
    logic signed [SCALE_W:0] shifted;
    logic signed [SCALE_W:0] relu_val;
    logic signed [SCALE_W:0] relu6_max_ext;
    logic signed [DATA_W-1:0] quantized;

    always_comb begin
        mult = $signed(in_acc) * $signed(mul);
        scaled = $signed(mult) + $signed(bias);
        shifted = scaled >>> shift;

        relu6_max_ext = {{(SCALE_W + 1 - DATA_W){1'b0}}, relu6_max};

        if (relu6_en) begin
            if (shifted < 0) begin
                relu_val = '0;
            end else if (shifted > relu6_max_ext) begin
                relu_val = relu6_max_ext;
            end else begin
                relu_val = shifted;
            end
        end else begin
            relu_val = shifted;
        end

        if (relu_val > Q_MAX) begin
            quantized = $signed(Q_MAX);
        end else if (relu_val < Q_MIN) begin
            quantized = $signed(Q_MIN);
        end else begin
            quantized = relu_val[DATA_W-1:0];
        end
    end

    assign in_ready = !out_valid || out_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_q <= '0;
        end else begin
            if (out_valid && out_ready) begin
                out_valid <= 1'b0;
            end

            if (in_valid && in_ready) begin
                out_q <= quantized;
                out_valid <= 1'b1;
            end
        end
    end
endmodule
