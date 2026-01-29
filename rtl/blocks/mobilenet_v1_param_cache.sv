module mobilenet_v1_param_cache #(
    parameter int DATA_W = 8,
    parameter int ACC_W = 32,
    parameter int MUL_W = 16,
    parameter int BIAS_W = 32,
    parameter int SHIFT_W = 6,
    parameter int OC_PAR = 4,
    parameter int DIM_W = 16,
    parameter int WR_W = 32,
    parameter int WR_ADDR_W = 20,
    parameter int CONV1_IN_CH = 3,
    parameter int CONV1_OUT_CH = 32,
    parameter int MAX_DW_CH = 1024,
    parameter int MAX_PW_IN_CH = 1024,
    parameter int MAX_PW_OUT_CH = 1024,
    parameter int GAP_CH = 1024,
    parameter int FC_IN_CH = 1024,
    parameter int FC_OUT_CH = 1000,
    parameter int PW_GROUP = 4,
    parameter bit REG_OUT = 0,
    parameter string INIT_CONV1_W = "",
    parameter string INIT_CONV1_BIAS_ACC = "",
    parameter string INIT_CONV1_MUL = "",
    parameter string INIT_CONV1_BIAS_RQ = "",
    parameter string INIT_CONV1_SHIFT = "",
    parameter string INIT_CONV1_RELU6 = "",
    parameter string INIT_DW_W = "",
    parameter string INIT_DW_MUL = "",
    parameter string INIT_DW_BIAS = "",
    parameter string INIT_DW_SHIFT = "",
    parameter string INIT_DW_RELU6 = "",
    parameter string INIT_PW_W = "",
    parameter string INIT_PW_BIAS_ACC = "",
    parameter string INIT_PW_MUL = "",
    parameter string INIT_PW_BIAS_RQ = "",
    parameter string INIT_PW_SHIFT = "",
    parameter string INIT_PW_RELU6 = "",
    parameter string INIT_GAP_MUL = "",
    parameter string INIT_GAP_BIAS = "",
    parameter string INIT_GAP_SHIFT = "",
    parameter string INIT_FC_W = "",
    parameter string INIT_FC_MUL = "",
    parameter string INIT_FC_BIAS = "",
    parameter string INIT_FC_SHIFT = ""
) (
    input  logic clk,
    input  logic rst_n,

    input  logic wr_en,
    input  logic [4:0] wr_sel,
    input  logic [WR_ADDR_W-1:0] wr_addr,
    input  logic [WR_W-1:0] wr_data,

    input  logic [DIM_W-1:0] layer_idx,
    input  logic [DIM_W-1:0] layer_in_c,
    input  logic [DIM_W-1:0] layer_out_c,

    input  logic [DIM_W-1:0] conv1_ic_idx,
    input  logic [DIM_W-1:0] conv1_oc_group_idx,

    input  logic [DIM_W-1:0] dw_ch_idx,
    input  logic [DIM_W-1:0] pw_in_ch_idx,
    input  logic [DIM_W-1:0] pw_out_ch_idx,

    input  logic pw_group_req,
    input  logic [DIM_W-1:0] pw_group_idx,
    output logic pw_group_ready,

    input  logic [DIM_W-1:0] gap_ch_idx,
    input  logic [DIM_W-1:0] fc_in_idx,
    input  logic [DIM_W-1:0] fc_out_idx,

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
    output logic signed [DATA_W-1:0] pw_relu6_max,

    output logic signed [MUL_W-1:0] gap_mul,
    output logic signed [BIAS_W-1:0] gap_bias,
    output logic [SHIFT_W-1:0] gap_shift,

    output logic signed [DATA_W-1:0] fc_weight,
    output logic signed [MUL_W-1:0] fc_mul,
    output logic signed [BIAS_W-1:0] fc_bias,
    output logic [SHIFT_W-1:0] fc_shift
);
    localparam int CONV1_WEIGHT_DEPTH = CONV1_OUT_CH * CONV1_IN_CH * 9;
    localparam int DW_TOTAL_CH = 4960;
    localparam int DW_WEIGHT_DEPTH = DW_TOTAL_CH * 9;
    localparam int PW_TOTAL_OUT_CH = 5952;
    localparam int PW_WEIGHT_DEPTH = 3139584;
    localparam int FC_WEIGHT_DEPTH = FC_OUT_CH * FC_IN_CH;
    localparam int PW_ADDR_W = (PW_WEIGHT_DEPTH <= 1) ? 1 : $clog2(PW_WEIGHT_DEPTH);

    logic signed [WR_W-1:0] conv1_weight_mem [0:CONV1_WEIGHT_DEPTH-1];
    logic signed [WR_W-1:0] conv1_bias_acc_mem [0:CONV1_OUT_CH-1];
    logic signed [WR_W-1:0] conv1_mul_mem [0:CONV1_OUT_CH-1];
    logic signed [WR_W-1:0] conv1_bias_requant_mem [0:CONV1_OUT_CH-1];
    logic [WR_W-1:0] conv1_shift_mem [0:CONV1_OUT_CH-1];
    logic signed [WR_W-1:0] conv1_relu6_mem [0:CONV1_OUT_CH-1];

    logic signed [WR_W-1:0] dw_weight_mem [0:DW_WEIGHT_DEPTH-1];
    logic signed [WR_W-1:0] dw_mul_mem [0:DW_TOTAL_CH-1];
    logic signed [WR_W-1:0] dw_bias_mem [0:DW_TOTAL_CH-1];
    logic [WR_W-1:0] dw_shift_mem [0:DW_TOTAL_CH-1];
    logic signed [WR_W-1:0] dw_relu6_mem [0:DW_TOTAL_CH-1];

    logic signed [WR_W-1:0] pw_weight_mem [0:PW_WEIGHT_DEPTH-1];
    logic signed [WR_W-1:0] pw_bias_acc_mem [0:PW_TOTAL_OUT_CH-1];
    logic signed [WR_W-1:0] pw_mul_mem [0:PW_TOTAL_OUT_CH-1];
    logic signed [WR_W-1:0] pw_bias_requant_mem [0:PW_TOTAL_OUT_CH-1];
    logic [WR_W-1:0] pw_shift_mem [0:PW_TOTAL_OUT_CH-1];
    logic signed [WR_W-1:0] pw_relu6_mem [0:PW_TOTAL_OUT_CH-1];

    logic signed [WR_W-1:0] gap_mul_mem [0:GAP_CH-1];
    logic signed [WR_W-1:0] gap_bias_mem [0:GAP_CH-1];
    logic [WR_W-1:0] gap_shift_mem [0:GAP_CH-1];

    logic signed [WR_W-1:0] fc_weight_mem [0:FC_WEIGHT_DEPTH-1];
    logic signed [WR_W-1:0] fc_mul_mem [0:FC_OUT_CH-1];
    logic signed [WR_W-1:0] fc_bias_mem [0:FC_OUT_CH-1];
    logic [WR_W-1:0] fc_shift_mem [0:FC_OUT_CH-1];

    localparam int PW_CACHE_DEPTH = PW_GROUP * MAX_PW_IN_CH;
    logic signed [WR_W-1:0] pw_cache_mem [0:PW_CACHE_DEPTH-1];
    logic [DIM_W-1:0] pw_cache_group;
    logic [DIM_W-1:0] pw_cache_layer;
    logic pw_cache_valid;
    logic pw_cache_load;
    logic [WR_ADDR_W-1:0] pw_cache_idx;
    logic [DIM_W-1:0] pw_cache_oc_off;
    logic [DIM_W-1:0] pw_cache_ic_off;
    logic [DIM_W-1:0] pw_cache_oc_global;
    logic [PW_ADDR_W-1:0] pw_cache_rom_addr;
    logic dbg_pw_en;

    // Write port. wr_sel selects which memory bank to write.
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
        end else if (wr_en) begin
            case (wr_sel)
                5'd0: if (wr_addr < CONV1_WEIGHT_DEPTH) conv1_weight_mem[wr_addr] <= wr_data;
                5'd1: if (wr_addr < CONV1_OUT_CH) conv1_bias_acc_mem[wr_addr] <= wr_data;
                5'd2: if (wr_addr < CONV1_OUT_CH) conv1_mul_mem[wr_addr] <= wr_data;
                5'd3: if (wr_addr < CONV1_OUT_CH) conv1_bias_requant_mem[wr_addr] <= wr_data;
                5'd4: if (wr_addr < CONV1_OUT_CH) conv1_shift_mem[wr_addr] <= wr_data;
                5'd5: if (wr_addr < CONV1_OUT_CH) conv1_relu6_mem[wr_addr] <= wr_data;

                5'd6: if (wr_addr < DW_WEIGHT_DEPTH) dw_weight_mem[wr_addr] <= wr_data;
                5'd7: if (wr_addr < DW_TOTAL_CH) dw_mul_mem[wr_addr] <= wr_data;
                5'd8: if (wr_addr < DW_TOTAL_CH) dw_bias_mem[wr_addr] <= wr_data;
                5'd9: if (wr_addr < DW_TOTAL_CH) dw_shift_mem[wr_addr] <= wr_data;
                5'd10: if (wr_addr < DW_TOTAL_CH) dw_relu6_mem[wr_addr] <= wr_data;

                5'd11: if (wr_addr < PW_WEIGHT_DEPTH) pw_weight_mem[wr_addr] <= wr_data;
                5'd12: if (wr_addr < PW_TOTAL_OUT_CH) pw_bias_acc_mem[wr_addr] <= wr_data;
                5'd13: if (wr_addr < PW_TOTAL_OUT_CH) pw_mul_mem[wr_addr] <= wr_data;
                5'd14: if (wr_addr < PW_TOTAL_OUT_CH) pw_bias_requant_mem[wr_addr] <= wr_data;
                5'd15: if (wr_addr < PW_TOTAL_OUT_CH) pw_shift_mem[wr_addr] <= wr_data;
                5'd16: if (wr_addr < PW_TOTAL_OUT_CH) pw_relu6_mem[wr_addr] <= wr_data;
                5'd17: if (wr_addr < GAP_CH) gap_mul_mem[wr_addr] <= wr_data;
                5'd18: if (wr_addr < GAP_CH) gap_bias_mem[wr_addr] <= wr_data;
                5'd19: if (wr_addr < GAP_CH) gap_shift_mem[wr_addr] <= wr_data;
                5'd20: if (wr_addr < FC_WEIGHT_DEPTH) fc_weight_mem[wr_addr] <= wr_data;
                5'd21: if (wr_addr < FC_OUT_CH) fc_mul_mem[wr_addr] <= wr_data;
                5'd22: if (wr_addr < FC_OUT_CH) fc_bias_mem[wr_addr] <= wr_data;
                5'd23: if (wr_addr < FC_OUT_CH) fc_shift_mem[wr_addr] <= wr_data;
                default: begin end
            endcase
        end
    end

    // Optional ROM initialization from files.
    initial begin
        if (INIT_CONV1_W != "") $readmemh(INIT_CONV1_W, conv1_weight_mem);
        if (INIT_CONV1_BIAS_ACC != "") $readmemh(INIT_CONV1_BIAS_ACC, conv1_bias_acc_mem);
        if (INIT_CONV1_MUL != "") $readmemh(INIT_CONV1_MUL, conv1_mul_mem);
        if (INIT_CONV1_BIAS_RQ != "") $readmemh(INIT_CONV1_BIAS_RQ, conv1_bias_requant_mem);
        if (INIT_CONV1_SHIFT != "") $readmemh(INIT_CONV1_SHIFT, conv1_shift_mem);
        if (INIT_CONV1_RELU6 != "") $readmemh(INIT_CONV1_RELU6, conv1_relu6_mem);

        if (INIT_DW_W != "") $readmemh(INIT_DW_W, dw_weight_mem);
        if (INIT_DW_MUL != "") $readmemh(INIT_DW_MUL, dw_mul_mem);
        if (INIT_DW_BIAS != "") $readmemh(INIT_DW_BIAS, dw_bias_mem);
        if (INIT_DW_SHIFT != "") $readmemh(INIT_DW_SHIFT, dw_shift_mem);
        if (INIT_DW_RELU6 != "") $readmemh(INIT_DW_RELU6, dw_relu6_mem);

        if (INIT_PW_W != "") $readmemh(INIT_PW_W, pw_weight_mem);
        if (INIT_PW_BIAS_ACC != "") $readmemh(INIT_PW_BIAS_ACC, pw_bias_acc_mem);
        if (INIT_PW_MUL != "") $readmemh(INIT_PW_MUL, pw_mul_mem);
        if (INIT_PW_BIAS_RQ != "") $readmemh(INIT_PW_BIAS_RQ, pw_bias_requant_mem);
        if (INIT_PW_SHIFT != "") $readmemh(INIT_PW_SHIFT, pw_shift_mem);
        if (INIT_PW_RELU6 != "") $readmemh(INIT_PW_RELU6, pw_relu6_mem);

        if (INIT_GAP_MUL != "") $readmemh(INIT_GAP_MUL, gap_mul_mem);
        if (INIT_GAP_BIAS != "") $readmemh(INIT_GAP_BIAS, gap_bias_mem);
        if (INIT_GAP_SHIFT != "") $readmemh(INIT_GAP_SHIFT, gap_shift_mem);

        if (INIT_FC_W != "") $readmemh(INIT_FC_W, fc_weight_mem);
        if (INIT_FC_MUL != "") $readmemh(INIT_FC_MUL, fc_mul_mem);
        if (INIT_FC_BIAS != "") $readmemh(INIT_FC_BIAS, fc_bias_mem);
        if (INIT_FC_SHIFT != "") $readmemh(INIT_FC_SHIFT, fc_shift_mem);
    end

    logic signed [OC_PAR*DATA_W*9-1:0] conv1_weight_flat_vec_c;
    logic signed [OC_PAR*ACC_W-1:0] conv1_bias_acc_vec_c;
    logic signed [OC_PAR*MUL_W-1:0] conv1_mul_vec_c;
    logic signed [OC_PAR*BIAS_W-1:0] conv1_bias_requant_vec_c;
    logic [OC_PAR*SHIFT_W-1:0] conv1_shift_vec_c;
    logic signed [OC_PAR*DATA_W-1:0] conv1_relu6_max_vec_c;

    logic signed [DATA_W*9-1:0] dw_weight_flat_c;
    logic signed [MUL_W-1:0] dw_mul_c;
    logic signed [BIAS_W-1:0] dw_bias_c;
    logic [SHIFT_W-1:0] dw_shift_c;
    logic signed [DATA_W-1:0] dw_relu6_max_c;

    logic signed [DATA_W-1:0] pw_weight_c;
    logic signed [ACC_W-1:0] pw_bias_acc_c;
    logic signed [MUL_W-1:0] pw_mul_c;
    logic signed [BIAS_W-1:0] pw_bias_requant_c;
    logic [SHIFT_W-1:0] pw_shift_c;
    logic signed [DATA_W-1:0] pw_relu6_max_c;

    logic signed [MUL_W-1:0] gap_mul_c;
    logic signed [BIAS_W-1:0] gap_bias_c;
    logic [SHIFT_W-1:0] gap_shift_c;

    logic signed [DATA_W-1:0] fc_weight_c;
    logic signed [MUL_W-1:0] fc_mul_c;
    logic signed [BIAS_W-1:0] fc_bias_c;
    logic [SHIFT_W-1:0] fc_shift_c;

    integer oc;
    integer k;
    logic [DIM_W-1:0] oc_global;
    logic [WR_ADDR_W-1:0] conv1_w_addr;
    logic [WR_ADDR_W-1:0] dw_w_addr;
    logic [WR_ADDR_W-1:0] pw_w_addr;
    logic [DIM_W-1:0] pw_group_local;
    logic [DIM_W-1:0] pw_group_in_ch;
    logic [DIM_W-1:0] pw_group_oc;

    logic [DIM_W-1:0] dws_idx;
    logic [WR_ADDR_W-1:0] dw_base;
    logic [PW_ADDR_W-1:0] pw_w_base;
    logic [WR_ADDR_W-1:0] pw_out_base;

    function automatic [WR_ADDR_W-1:0] dw_base_for_layer(input [DIM_W-1:0] idx);
        case (idx)
            0: dw_base_for_layer = 0;
            1: dw_base_for_layer = 32;
            2: dw_base_for_layer = 96;
            3: dw_base_for_layer = 224;
            4: dw_base_for_layer = 352;
            5: dw_base_for_layer = 608;
            6: dw_base_for_layer = 864;
            7: dw_base_for_layer = 1376;
            8: dw_base_for_layer = 1888;
            9: dw_base_for_layer = 2400;
            10: dw_base_for_layer = 2912;
            11: dw_base_for_layer = 3424;
            12: dw_base_for_layer = 3936;
            default: dw_base_for_layer = 0;
        endcase
    endfunction

    function automatic [WR_ADDR_W-1:0] pw_out_base_for_layer(input [DIM_W-1:0] idx);
        case (idx)
            0: pw_out_base_for_layer = 0;
            1: pw_out_base_for_layer = 64;
            2: pw_out_base_for_layer = 192;
            3: pw_out_base_for_layer = 320;
            4: pw_out_base_for_layer = 576;
            5: pw_out_base_for_layer = 832;
            6: pw_out_base_for_layer = 1344;
            7: pw_out_base_for_layer = 1856;
            8: pw_out_base_for_layer = 2368;
            9: pw_out_base_for_layer = 2880;
            10: pw_out_base_for_layer = 3392;
            11: pw_out_base_for_layer = 3904;
            12: pw_out_base_for_layer = 4928;
            default: pw_out_base_for_layer = 0;
        endcase
    endfunction

    function automatic [PW_ADDR_W-1:0] pw_w_base_for_layer(input [DIM_W-1:0] idx);
        case (idx)
            0: pw_w_base_for_layer = 0;
            1: pw_w_base_for_layer = 2048;
            2: pw_w_base_for_layer = 10240;
            3: pw_w_base_for_layer = 26624;
            4: pw_w_base_for_layer = 59392;
            5: pw_w_base_for_layer = 124928;
            6: pw_w_base_for_layer = 256000;
            7: pw_w_base_for_layer = 518144;
            8: pw_w_base_for_layer = 780288;
            9: pw_w_base_for_layer = 1042432;
            10: pw_w_base_for_layer = 1304576;
            11: pw_w_base_for_layer = 1566720;
            12: pw_w_base_for_layer = 2091008;
            default: pw_w_base_for_layer = 0;
        endcase
    endfunction

    always_comb begin
        conv1_w_addr = '0;
        dw_w_addr = '0;
        pw_w_addr = '0;
        oc_global = '0;
        pw_group_local = '0;
        pw_group_oc = '0;
        pw_group_in_ch = '0;

        if (layer_idx == 0) begin
            dws_idx = '0;
        end else begin
            dws_idx = layer_idx - 1'b1;
        end
        dw_base = dw_base_for_layer(dws_idx);
        pw_out_base = pw_out_base_for_layer(dws_idx);
        pw_w_base = pw_w_base_for_layer(dws_idx);

        conv1_weight_flat_vec_c = '0;
        conv1_bias_acc_vec_c = '0;
        conv1_mul_vec_c = '0;
        conv1_bias_requant_vec_c = '0;
        conv1_shift_vec_c = '0;
        conv1_relu6_max_vec_c = '0;

        for (oc = 0; oc < OC_PAR; oc = oc + 1) begin
            oc_global = conv1_oc_group_idx * OC_PAR + oc[DIM_W-1:0];
            if (oc_global < CONV1_OUT_CH) begin
                for (k = 0; k < 9; k = k + 1) begin
                    conv1_w_addr = ((oc_global * CONV1_IN_CH) + conv1_ic_idx) * 9 + k;
                    if (conv1_w_addr < CONV1_WEIGHT_DEPTH) begin
                        conv1_weight_flat_vec_c[(oc*9 + k)*DATA_W +: DATA_W] =
                            conv1_weight_mem[conv1_w_addr][DATA_W-1:0];
                    end
                end
                conv1_bias_acc_vec_c[oc*ACC_W +: ACC_W] =
                    conv1_bias_acc_mem[oc_global][ACC_W-1:0];
                conv1_mul_vec_c[oc*MUL_W +: MUL_W] =
                    conv1_mul_mem[oc_global][MUL_W-1:0];
                conv1_bias_requant_vec_c[oc*BIAS_W +: BIAS_W] =
                    conv1_bias_requant_mem[oc_global][BIAS_W-1:0];
                conv1_shift_vec_c[oc*SHIFT_W +: SHIFT_W] =
                    conv1_shift_mem[oc_global][SHIFT_W-1:0];
                conv1_relu6_max_vec_c[oc*DATA_W +: DATA_W] =
                    conv1_relu6_mem[oc_global][DATA_W-1:0];
            end
        end

        dw_weight_flat_c = '0;
        for (k = 0; k < 9; k = k + 1) begin
            dw_w_addr = (dw_base + dw_ch_idx) * 9 + k;
            if (dw_w_addr < DW_WEIGHT_DEPTH) begin
                dw_weight_flat_c[k*DATA_W +: DATA_W] = dw_weight_mem[dw_w_addr][DATA_W-1:0];
            end
        end
        if ((dw_base + dw_ch_idx) < DW_TOTAL_CH) begin
            dw_mul_c = dw_mul_mem[dw_base + dw_ch_idx][MUL_W-1:0];
            dw_bias_c = dw_bias_mem[dw_base + dw_ch_idx][BIAS_W-1:0];
            dw_shift_c = dw_shift_mem[dw_base + dw_ch_idx][SHIFT_W-1:0];
            dw_relu6_max_c = dw_relu6_mem[dw_base + dw_ch_idx][DATA_W-1:0];
        end else begin
            dw_mul_c = '0;
            dw_bias_c = '0;
            dw_shift_c = '0;
            dw_relu6_max_c = '0;
        end

        pw_weight_c = '0;
        pw_bias_acc_c = '0;
        pw_mul_c = '0;
        pw_bias_requant_c = '0;
        pw_shift_c = '0;
        pw_relu6_max_c = '0;

        gap_mul_c = '0;
        gap_bias_c = '0;
        gap_shift_c = '0;

        fc_weight_c = '0;
        fc_mul_c = '0;
        fc_bias_c = '0;
        fc_shift_c = '0;

        pw_group_local = pw_out_ch_idx / PW_GROUP;
        pw_group_oc = pw_out_ch_idx % PW_GROUP;
        pw_group_in_ch = pw_in_ch_idx;

        if (pw_cache_valid && (pw_cache_group == pw_group_local)) begin
            pw_w_addr = pw_group_oc * layer_in_c + pw_group_in_ch;
            if (pw_w_addr < PW_CACHE_DEPTH) begin
                pw_weight_c = pw_cache_mem[pw_w_addr][DATA_W-1:0];
            end
        end
        if ((pw_out_base + pw_out_ch_idx) < PW_TOTAL_OUT_CH) begin
            pw_bias_acc_c = pw_bias_acc_mem[pw_out_base + pw_out_ch_idx][ACC_W-1:0];
            pw_mul_c = pw_mul_mem[pw_out_base + pw_out_ch_idx][MUL_W-1:0];
            pw_bias_requant_c = pw_bias_requant_mem[pw_out_base + pw_out_ch_idx][BIAS_W-1:0];
            pw_shift_c = pw_shift_mem[pw_out_base + pw_out_ch_idx][SHIFT_W-1:0];
            pw_relu6_max_c = pw_relu6_mem[pw_out_base + pw_out_ch_idx][DATA_W-1:0];
        end

        if (gap_ch_idx < GAP_CH) begin
            gap_mul_c = gap_mul_mem[gap_ch_idx][MUL_W-1:0];
            gap_bias_c = gap_bias_mem[gap_ch_idx][BIAS_W-1:0];
            gap_shift_c = gap_shift_mem[gap_ch_idx][SHIFT_W-1:0];
        end

        if (fc_out_idx < FC_OUT_CH && fc_in_idx < FC_IN_CH) begin
            fc_weight_c = fc_weight_mem[(fc_out_idx * FC_IN_CH) + fc_in_idx][DATA_W-1:0];
            fc_mul_c = fc_mul_mem[fc_out_idx][MUL_W-1:0];
            fc_bias_c = fc_bias_mem[fc_out_idx][BIAS_W-1:0];
            fc_shift_c = fc_shift_mem[fc_out_idx][SHIFT_W-1:0];
        end
    end

    generate
        if (REG_OUT) begin : gen_reg
            always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    conv1_weight_flat_vec <= '0;
                    conv1_bias_acc_vec <= '0;
                    conv1_mul_vec <= '0;
                    conv1_bias_requant_vec <= '0;
                    conv1_shift_vec <= '0;
                    conv1_relu6_max_vec <= '0;
                    dw_weight_flat <= '0;
                    dw_mul <= '0;
                    dw_bias <= '0;
                    dw_shift <= '0;
                    dw_relu6_max <= '0;
                    pw_weight <= '0;
                    pw_bias_acc <= '0;
                    pw_mul <= '0;
                    pw_bias_requant <= '0;
                    pw_shift <= '0;
                    pw_relu6_max <= '0;
                    gap_mul <= '0;
                    gap_bias <= '0;
                    gap_shift <= '0;
                    fc_weight <= '0;
                    fc_mul <= '0;
                    fc_bias <= '0;
                    fc_shift <= '0;
                end else begin
                    conv1_weight_flat_vec <= conv1_weight_flat_vec_c;
                    conv1_bias_acc_vec <= conv1_bias_acc_vec_c;
                    conv1_mul_vec <= conv1_mul_vec_c;
                    conv1_bias_requant_vec <= conv1_bias_requant_vec_c;
                    conv1_shift_vec <= conv1_shift_vec_c;
                    conv1_relu6_max_vec <= conv1_relu6_max_vec_c;
                    dw_weight_flat <= dw_weight_flat_c;
                    dw_mul <= dw_mul_c;
                    dw_bias <= dw_bias_c;
                    dw_shift <= dw_shift_c;
                    dw_relu6_max <= dw_relu6_max_c;
                    pw_weight <= pw_weight_c;
                    pw_bias_acc <= pw_bias_acc_c;
                    pw_mul <= pw_mul_c;
                    pw_bias_requant <= pw_bias_requant_c;
                    pw_shift <= pw_shift_c;
                    pw_relu6_max <= pw_relu6_max_c;
                    gap_mul <= gap_mul_c;
                    gap_bias <= gap_bias_c;
                    gap_shift <= gap_shift_c;
                    fc_weight <= fc_weight_c;
                    fc_mul <= fc_mul_c;
                    fc_bias <= fc_bias_c;
                    fc_shift <= fc_shift_c;
                end
            end
        end else begin : gen_comb
            always_comb begin
                conv1_weight_flat_vec = conv1_weight_flat_vec_c;
                conv1_bias_acc_vec = conv1_bias_acc_vec_c;
                conv1_mul_vec = conv1_mul_vec_c;
                conv1_bias_requant_vec = conv1_bias_requant_vec_c;
                conv1_shift_vec = conv1_shift_vec_c;
                conv1_relu6_max_vec = conv1_relu6_max_vec_c;
                dw_weight_flat = dw_weight_flat_c;
                dw_mul = dw_mul_c;
                dw_bias = dw_bias_c;
                dw_shift = dw_shift_c;
                dw_relu6_max = dw_relu6_max_c;
                pw_weight = pw_weight_c;
                pw_bias_acc = pw_bias_acc_c;
                pw_mul = pw_mul_c;
                pw_bias_requant = pw_bias_requant_c;
                pw_shift = pw_shift_c;
                pw_relu6_max = pw_relu6_max_c;
                gap_mul = gap_mul_c;
                gap_bias = gap_bias_c;
                gap_shift = gap_shift_c;
                fc_weight = fc_weight_c;
                fc_mul = fc_mul_c;
                fc_bias = fc_bias_c;
                fc_shift = fc_shift_c;
            end
        end
    endgenerate

    // Pointwise weight cache loader (grouped by PW_GROUP output channels).
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            pw_cache_valid <= 1'b0;
            pw_cache_group <= '0;
            pw_cache_layer <= '0;
            pw_cache_load <= 1'b0;
            pw_cache_idx <= '0;
            pw_cache_oc_off <= '0;
            pw_cache_ic_off <= '0;
        end else begin
            dbg_pw_en <= $test$plusargs("DBG_PW_GROUP");
            if (pw_group_req &&
                (!pw_cache_valid || pw_cache_group != pw_group_idx || pw_cache_layer != layer_idx) &&
                !pw_cache_load) begin
                pw_cache_load <= 1'b1;
                pw_cache_idx <= '0;
                pw_cache_group <= pw_group_idx;
                pw_cache_layer <= layer_idx;
                pw_cache_valid <= 1'b0;
                pw_cache_oc_off <= '0;
                pw_cache_ic_off <= '0;
                if (dbg_pw_en) begin
                    $display("PW_CACHE load start layer=%0d grp=%0d in_c=%0d out_c=%0d base=%0d",
                             layer_idx, pw_group_idx, layer_in_c, layer_out_c, pw_w_base);
                end
            end

            if (pw_cache_load) begin
                if (pw_cache_idx < (PW_GROUP * layer_in_c)) begin
                    if (pw_cache_oc_global < layer_out_c && pw_cache_rom_addr < PW_WEIGHT_DEPTH) begin
                        pw_cache_mem[pw_cache_idx] <= pw_weight_mem[pw_cache_rom_addr];
                    end else begin
                        pw_cache_mem[pw_cache_idx] <= '0;
                    end

                    if (pw_cache_idx == (PW_GROUP * layer_in_c) - 1) begin
                        pw_cache_load <= 1'b0;
                        pw_cache_valid <= 1'b1;
                        if (dbg_pw_en) begin
                            $display("PW_CACHE load done layer=%0d grp=%0d", pw_cache_layer, pw_cache_group);
                        end
                    end else begin
                        pw_cache_idx <= pw_cache_idx + 1'b1;
                        if (pw_cache_ic_off == layer_in_c - 1'b1) begin
                            pw_cache_ic_off <= '0;
                            pw_cache_oc_off <= pw_cache_oc_off + 1'b1;
                        end else begin
                            pw_cache_ic_off <= pw_cache_ic_off + 1'b1;
                        end
                    end
                end else begin
                    pw_cache_load <= 1'b0;
                    pw_cache_valid <= 1'b1;
                end
            end
        end
    end

    always_comb begin
        pw_cache_oc_global = pw_cache_group * PW_GROUP + pw_cache_oc_off;
        pw_cache_rom_addr = pw_w_base + (pw_cache_oc_global * layer_in_c) + pw_cache_ic_off;
    end

    assign pw_group_ready = pw_cache_valid &&
                            (pw_cache_group == pw_group_idx) &&
                            (pw_cache_layer == layer_idx) &&
                            !pw_cache_load;
endmodule
