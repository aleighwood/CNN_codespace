module dws_block #(
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

    input  logic dw_in_valid,
    output logic dw_in_ready,
    input  logic signed [DATA_W-1:0] dw_in_data,

    input  logic dw_start,
    input  logic [$clog2(MAX_IMG_H)-1:0] dw_cfg_img_h,
    input  logic [$clog2(MAX_IMG_W)-1:0] dw_cfg_img_w,
    input  logic [$clog2(MAX_IMG_H)-1:0] dw_cfg_stride,

    input  logic signed [DATA_W*9-1:0] dw_weight_flat,
    input  logic signed [MUL_W-1:0] dw_mul,
    input  logic signed [ACC_W-1:0] dw_bias_acc,
    input  logic [SHIFT_W-1:0] dw_shift,
    input  logic signed [DATA_W-1:0] dw_relu6_max,
    input  logic signed [DATA_W-1:0] dw_relu6_min,

    output logic dw_out_valid,
    input  logic dw_out_ready,
    output logic signed [DATA_W-1:0] dw_out_data,

    input  logic pw_in_valid,
    output logic pw_in_ready,
    input  logic signed [DATA_W-1:0] pw_in_data,
    input  logic signed [DATA_W-1:0] pw_weight,
    input  logic signed [ACC_W-1:0] pw_bias_acc,
    input  logic pw_first_in_ch,
    input  logic pw_last_in_ch,

    input  logic signed [MUL_W-1:0] pw_mul,
    input  logic signed [BIAS_W-1:0] pw_bias_requant,
    input  logic [SHIFT_W-1:0] pw_shift,
    input  logic signed [DATA_W-1:0] pw_relu6_max,
    input  logic signed [DATA_W-1:0] pw_relu6_min,

    output logic pw_out_valid,
    input  logic pw_out_ready,
    output logic signed [DATA_W-1:0] pw_out_data
);
    logic pw_acc_valid;
    logic pw_acc_ready;
    logic signed [ACC_W-1:0] pw_acc;

    depthwise_stage #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .MAX_IMG_W(MAX_IMG_W),
        .MAX_IMG_H(MAX_IMG_H),
        .MUL_W(MUL_W),
        .BIAS_W(BIAS_W),
        .SHIFT_W(SHIFT_W)
    ) u_dw_stage (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(dw_in_valid),
        .in_ready(dw_in_ready),
        .in_data(dw_in_data),
        .start(dw_start),
        .cfg_img_h(dw_cfg_img_h),
        .cfg_img_w(dw_cfg_img_w),
        .cfg_stride(dw_cfg_stride),
        .dw_weight_flat(dw_weight_flat),
        .dw_mul(dw_mul),
        .dw_bias_acc(dw_bias_acc),
        .dw_shift(dw_shift),
        .dw_relu6_max(dw_relu6_max),
        .dw_relu6_min(dw_relu6_min),
        .out_valid(dw_out_valid),
        .out_ready(dw_out_ready),
        .out_data(dw_out_data)
    );

    pw_conv_1x1 #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W)
    ) u_pw_conv (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(pw_in_valid),
        .in_ready(pw_in_ready),
        .in_data(pw_in_data),
        .weight(pw_weight),
        .bias(pw_bias_acc),
        .first_in_ch(pw_first_in_ch),
        .last_in_ch(pw_last_in_ch),
        .out_valid(pw_acc_valid),
        .out_ready(pw_acc_ready),
        .out_acc(pw_acc)
    );

    requant_q31 #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .MUL_W(MUL_W),
        .SHIFT_W(SHIFT_W)
    ) u_pw_requant (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(pw_acc_valid),
        .in_ready(pw_acc_ready),
        .in_acc(pw_acc),
        .mul_q31(pw_mul),
        .shift(pw_shift),
        .zp_out(pw_relu6_min),
        .relu6_max(pw_relu6_max),
        .relu6_en(1'b1),
        .out_valid(pw_out_valid),
        .out_ready(pw_out_ready),
        .out_q(pw_out_data)
    );
endmodule
