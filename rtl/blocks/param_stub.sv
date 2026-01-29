module mobilenet_v1_param_stub #(
    parameter int DATA_W = 8,
    parameter int ACC_W = 32,
    parameter int MUL_W = 16,
    parameter int BIAS_W = 32,
    parameter int SHIFT_W = 6,
    parameter int OC_PAR = 4,
    parameter int DIM_W = 16
) (
    input  logic [DIM_W-1:0] layer_idx,

    input  logic [DIM_W-1:0] conv1_ic_idx,
    input  logic [DIM_W-1:0] conv1_oc_group_idx,

    input  logic [DIM_W-1:0] dw_ch_idx,
    input  logic [DIM_W-1:0] pw_in_ch_idx,
    input  logic [DIM_W-1:0] pw_out_ch_idx,

    output logic signed [OC_PAR*DATA_W*9-1:0] conv1_weight_flat_vec,
    output logic signed [OC_PAR*ACC_W-1:0] conv1_bias_acc_vec,
    output logic signed [OC_PAR*MUL_W-1:0] conv1_mul_vec,
    output logic signed [OC_PAR*BIAS_W-1:0] conv1_bias_requant_vec,
    output logic [OC_PAR*SHIFT_W-1:0] conv1_shift_vec,
    output logic signed [OC_PAR*DATA_W-1:0] conv1_relu6_max_vec,

    output logic signed [DATA_W*9-1:0] dw_weight_flat,
    output logic signed [MUL_W-1:0] dw_mul,
    output logic signed [BIAS_W-1:0] dw_bias,
    output logic [SHIFT_W-1:0] dw_shift,
    output logic signed [DATA_W-1:0] dw_relu6_max,

    output logic signed [DATA_W-1:0] pw_weight,
    output logic signed [ACC_W-1:0] pw_bias_acc,
    output logic signed [MUL_W-1:0] pw_mul,
    output logic signed [BIAS_W-1:0] pw_bias_requant,
    output logic [SHIFT_W-1:0] pw_shift,
    output logic signed [DATA_W-1:0] pw_relu6_max
);
    // Stub outputs. Replace with ROM/BRAM or external memory interface.
    always_comb begin
        conv1_weight_flat_vec = '0;
        conv1_bias_acc_vec = '0;
        conv1_mul_vec = '0;
        conv1_bias_requant_vec = '0;
        conv1_shift_vec = '0;
        conv1_relu6_max_vec = '0;

        dw_weight_flat = '0;
        dw_mul = '0;
        dw_bias = '0;
        dw_shift = '0;
        dw_relu6_max = '0;

        pw_weight = '0;
        pw_bias_acc = '0;
        pw_mul = '0;
        pw_bias_requant = '0;
        pw_shift = '0;
        pw_relu6_max = '0;
    end
endmodule
