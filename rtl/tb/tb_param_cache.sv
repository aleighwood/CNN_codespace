`timescale 1ns/1ps

module tb_param_cache;
    localparam int DATA_W = 8;
    localparam int ACC_W = 32;
    localparam int MUL_W = 32;
    localparam int BIAS_W = 32;
    localparam int SHIFT_W = 6;
    localparam int DIM_W = 16;
    localparam int PW_OC_PAR = 32;
    localparam int PW_IC_PAR = 16;

    logic clk;
    logic rst_n;

    logic [DIM_W-1:0] layer_idx;
    logic [DIM_W-1:0] layer_in_c;
    logic [DIM_W-1:0] layer_out_c;

    logic [DIM_W-1:0] conv1_ic_idx;
    logic [DIM_W-1:0] conv1_oc_group_idx;

    logic [DIM_W-1:0] dw_ch_idx;
    logic [DIM_W-1:0] pw_in_ch_idx;
    logic [DIM_W-1:0] pw_out_ch_idx;

    logic pw_group_req;
    logic [DIM_W-1:0] pw_group_idx;
    logic pw_group_ready;

    logic [DIM_W-1:0] gap_ch_idx;
    logic [DIM_W-1:0] fc_in_idx;
    logic [DIM_W-1:0] fc_out_idx;

    logic signed [DATA_W*9-1:0] dw_weight_flat;
    logic signed [MUL_W-1:0] dw_mul;
    logic signed [BIAS_W-1:0] dw_bias;
    logic [SHIFT_W-1:0] dw_shift;

    logic signed [PW_OC_PAR*PW_IC_PAR*DATA_W-1:0] pw_weight_vec;
    logic signed [PW_OC_PAR*MUL_W-1:0] pw_mul_vec;
    logic signed [PW_OC_PAR*ACC_W-1:0] pw_bias_acc_vec;
    logic [PW_OC_PAR*SHIFT_W-1:0] pw_shift_vec;

    logic signed [MUL_W-1:0] gap_mul;
    logic signed [BIAS_W-1:0] gap_bias;
    logic [SHIFT_W-1:0] gap_shift;

    logic signed [DATA_W-1:0] fc_weight;
    logic signed [MUL_W-1:0] fc_mul;
    logic signed [BIAS_W-1:0] fc_bias;
    logic [SHIFT_W-1:0] fc_shift;

    mobilenet_v1_param_cache #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .MUL_W(MUL_W),
        .BIAS_W(BIAS_W),
        .SHIFT_W(SHIFT_W),
        .DIM_W(DIM_W),
        .PW_OC_PAR(PW_OC_PAR),
        .PW_IC_PAR(PW_IC_PAR),
        .INIT_CONV1_W("rtl/mem/conv1_weight.mem"),
        .INIT_CONV1_BIAS_ACC("rtl/mem/conv1_bias_acc.mem"),
        .INIT_CONV1_MUL("rtl/mem/conv1_mul.mem"),
        .INIT_CONV1_BIAS_RQ("rtl/mem/conv1_bias_rq.mem"),
        .INIT_CONV1_SHIFT("rtl/mem/conv1_shift.mem"),
        .INIT_CONV1_RELU6("rtl/mem/conv1_relu6.mem"),
        .INIT_CONV1_RELU6_MIN("rtl/mem/conv1_relu6_min.mem"),
        .INIT_DW_W("rtl/mem/dw_weight.mem"),
        .INIT_DW_MUL("rtl/mem/dw_mul.mem"),
        .INIT_DW_BIAS("rtl/mem/dw_bias_acc.mem"),
        .INIT_DW_SHIFT("rtl/mem/dw_shift.mem"),
        .INIT_DW_RELU6("rtl/mem/dw_relu6.mem"),
        .INIT_DW_RELU6_MIN("rtl/mem/dw_relu6_min.mem"),
        .INIT_PW_W("rtl/mem/pw_weight.mem"),
        .INIT_PW_BIAS_ACC("rtl/mem/pw_bias_acc.mem"),
        .INIT_PW_MUL("rtl/mem/pw_mul.mem"),
        .INIT_PW_BIAS_RQ("rtl/mem/pw_bias_rq.mem"),
        .INIT_PW_SHIFT("rtl/mem/pw_shift.mem"),
        .INIT_PW_RELU6("rtl/mem/pw_relu6.mem"),
        .INIT_PW_RELU6_MIN("rtl/mem/pw_relu6_min.mem"),
        .INIT_GAP_MUL("rtl/mem/gap_mul.mem"),
        .INIT_GAP_BIAS("rtl/mem/gap_bias.mem"),
        .INIT_GAP_SHIFT("rtl/mem/gap_shift.mem"),
        .INIT_FC_W("rtl/mem/fc_weight.mem"),
        .INIT_FC_MUL("rtl/mem/fc_mul.mem"),
        .INIT_FC_BIAS("rtl/mem/fc_bias_acc.mem"),
        .INIT_FC_SHIFT("rtl/mem/fc_shift.mem"),
        .INIT_FC_ZP("rtl/mem/fc_zp.mem")
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(1'b0),
        .wr_sel('0),
        .wr_addr('0),
        .wr_data('0),
        .layer_idx(layer_idx),
        .layer_in_c(layer_in_c),
        .layer_out_c(layer_out_c),
        .conv1_ic_idx(conv1_ic_idx),
        .conv1_oc_group_idx(conv1_oc_group_idx),
        .dw_ch_idx(dw_ch_idx),
        .pw_in_ch_idx(pw_in_ch_idx),
        .pw_out_ch_idx(pw_out_ch_idx),
        .pw_group_req(pw_group_req),
        .pw_group_idx(pw_group_idx),
        .pw_group_ready(pw_group_ready),
        .gap_ch_idx(gap_ch_idx),
        .fc_in_idx(fc_in_idx),
        .fc_out_idx(fc_out_idx),
        .conv1_weight_flat_vec(),
        .conv1_bias_acc_vec(),
        .conv1_mul_vec(),
        .conv1_bias_requant_vec(),
        .conv1_shift_vec(),
        .conv1_relu6_max_vec(),
        .conv1_relu6_min_vec(),
        .dw_weight_flat(dw_weight_flat),
        .dw_mul(dw_mul),
        .dw_bias(dw_bias),
        .dw_shift(dw_shift),
        .dw_relu6_max(),
        .dw_relu6_min(),
        .pw_weight_vec(pw_weight_vec),
        .pw_bias_acc_vec(pw_bias_acc_vec),
        .pw_mul_vec(pw_mul_vec),
        .pw_shift_vec(pw_shift_vec),
        .pw_relu6_max_vec(),
        .pw_relu6_min_vec(),
        .gap_mul(gap_mul),
        .gap_bias(gap_bias),
        .gap_shift(gap_shift),
        .fc_weight(fc_weight),
        .fc_mul(fc_mul),
        .fc_bias_acc(fc_bias),
        .fc_shift(fc_shift),
        .fc_zp()
    );

    always #5 clk = ~clk;

    initial begin
        clk = 0;
        rst_n = 0;
        layer_idx = 0;
        layer_in_c = 32;
        layer_out_c = 64;
        conv1_ic_idx = 0;
        conv1_oc_group_idx = 0;
        dw_ch_idx = 0;
        pw_in_ch_idx = 0;
        pw_out_ch_idx = 0;
        pw_group_idx = 0;
        pw_group_req = 0;
        gap_ch_idx = 0;
        fc_in_idx = 0;
        fc_out_idx = 0;

        #20;
        rst_n = 1;

        // Load pointwise group 0
        pw_group_req = 1'b1;
        #10;
        pw_group_req = 1'b0;
        wait (pw_group_ready == 1'b1);

        // Sample a few outputs
        #10;
        $display("DW weight[0]: %0d", dw_weight_flat[7:0]);
        $display("PW weight[0,0]: %0d", pw_weight_vec[0 +: DATA_W]);
        $display("PW mul[0]: %0d", pw_mul);
        $display("GAP mul[0]: %0d", gap_mul);
        $display("FC weight[0,0]: %0d", fc_weight);
        $display("FC mul[0]: %0d", fc_mul);

        #20;
        $finish;
    end
endmodule
