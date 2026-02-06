module depthwise_stage #(
    parameter int DATA_W = 8,
    parameter int ACC_W = 32,
    parameter int MAX_IMG_W = 224,
    parameter int MAX_IMG_H = 224,
    parameter int MUL_W = 32,
    parameter int BIAS_W = 32,
    parameter int SHIFT_W = 6
) (
    input  logic clk,
    input  logic rst_n,

    input  logic in_valid,
    output logic in_ready,
    input  logic signed [DATA_W-1:0] in_data,

    input  logic start,
    input  logic [$clog2(MAX_IMG_H)-1:0] cfg_img_h,
    input  logic [$clog2(MAX_IMG_W)-1:0] cfg_img_w,
    input  logic [$clog2(MAX_IMG_H)-1:0] cfg_stride,

    input  logic signed [DATA_W*9-1:0] dw_weight_flat,
    input  logic signed [MUL_W-1:0] dw_mul,
    input  logic signed [ACC_W-1:0] dw_bias_acc,
    input  logic [SHIFT_W-1:0] dw_shift,
    input  logic signed [DATA_W-1:0] dw_relu6_max,
    input  logic signed [DATA_W-1:0] dw_relu6_min,

    output logic out_valid,
    input  logic out_ready,
    output logic signed [DATA_W-1:0] out_data
);
    logic lb_valid;
    logic lb_ready;
    logic signed [DATA_W*9-1:0] lb_window;

    logic dw_valid;
    logic dw_ready;
    logic signed [ACC_W-1:0] dw_acc;
    logic signed [ACC_W-1:0] dw_acc_bias;

    line_buffer_3x3 #(
        .DATA_W(DATA_W),
        .MAX_IMG_W(MAX_IMG_W),
        .MAX_IMG_H(MAX_IMG_H)
    ) u_line_buffer (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .in_data(in_data),
        .start(start),
        .cfg_img_h(cfg_img_h),
        .cfg_img_w(cfg_img_w),
        .cfg_stride(cfg_stride),
        .out_valid(lb_valid),
        .out_ready(lb_ready),
        .window_flat(lb_window),
        .out_row(),
        .out_col()
    );

    dw_conv_3x3 #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W)
    ) u_dw_conv (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(lb_valid),
        .in_ready(lb_ready),
        .window_flat(lb_window),
        .weight_flat(dw_weight_flat),
        .out_valid(dw_valid),
        .out_ready(dw_ready),
        .out_acc(dw_acc)
    );

    assign dw_acc_bias = dw_acc + dw_bias_acc;

    requant_q31 #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .MUL_W(MUL_W),
        .SHIFT_W(SHIFT_W)
    ) u_requant (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(dw_valid),
        .in_ready(dw_ready),
        .in_acc(dw_acc_bias),
        .mul_q31(dw_mul),
        .shift(dw_shift),
        .zp_out(dw_relu6_min),
        .relu6_max(dw_relu6_max),
        .relu6_en(1'b1),
        .out_valid(out_valid),
        .out_ready(out_ready),
        .out_q(out_data)
    );
endmodule
