module mobilenet_v1_top #(
    parameter int DATA_W = 8,
    parameter int ACC_W = 32,
    parameter int MUL_W = 16,
    parameter int BIAS_W = 32,
    parameter int SHIFT_W = 6,
    parameter int ADDR_W = 32,
    parameter int DIM_W = 16,
    parameter int OC_PAR = 4,
    parameter int PW_GROUP = 4,
    parameter int FC_OUT_CH = 1000,
    parameter int TILE_H = 16,
    parameter int TILE_W = 16,
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

    input  logic start,
    output logic done,

    input  logic [DIM_W-1:0] cfg_in_img_h,
    input  logic [DIM_W-1:0] cfg_in_img_w,

    input  logic [ADDR_W-1:0] cfg_fm_base0,
    input  logic [ADDR_W-1:0] cfg_fm_base1,
    input  logic [ADDR_W-1:0] cfg_dw_buf_base,

    input  logic param_wr_en,
    input  logic [4:0] param_wr_sel,
    input  logic [19:0] param_wr_addr,
    input  logic [31:0] param_wr_data,

    output logic fm_rd_en,
    output logic [ADDR_W-1:0] fm_rd_addr,
    input  logic [DATA_W-1:0] fm_rd_data,

    output logic fm_wr_en,
    output logic [ADDR_W-1:0] fm_wr_addr,
    output logic [DATA_W-1:0] fm_wr_data,

    output logic dw_buf_rd_en,
    output logic [ADDR_W-1:0] dw_buf_rd_addr,
    input  logic [DATA_W-1:0] dw_buf_rd_data,

    output logic dw_buf_wr_en,
    output logic [ADDR_W-1:0] dw_buf_wr_addr,
    output logic [DATA_W-1:0] dw_buf_wr_data
);
    logic busy;
    logic core_start;
    logic core_done;

    typedef enum logic [2:0] {
        TOP_IDLE,
        TOP_CORE,
        TOP_GAP,
        TOP_FC,
        TOP_DONE
    } top_state_t;

    top_state_t top_state;

    logic layer_is_conv1;
    logic [DIM_W-1:0] layer_idx;

    logic [DIM_W-1:0] cur_in_h;
    logic [DIM_W-1:0] cur_in_w;
    logic [DIM_W-1:0] cur_out_h;
    logic [DIM_W-1:0] cur_out_w;
    logic [DIM_W-1:0] cur_in_c;
    logic [DIM_W-1:0] cur_out_c;
    logic [DIM_W-1:0] cur_stride;

    logic signed [DIM_W:0] tile_in_row;
    logic signed [DIM_W:0] tile_in_col;
    logic [DIM_W-1:0] tile_in_h;
    logic [DIM_W-1:0] tile_in_w;
    logic [DIM_W-1:0] tile_out_row;
    logic [DIM_W-1:0] tile_out_col;
    logic [DIM_W-1:0] tile_out_h;
    logic [DIM_W-1:0] tile_out_w;

    logic [ADDR_W-1:0] in_base_addr;
    logic [ADDR_W-1:0] out_base_addr;
    logic [ADDR_W-1:0] dw_buf_base_addr;

    logic conv1_start;
    logic conv1_busy;
    logic conv1_done;

    logic dws_start;
    logic dws_busy;
    logic dws_done;

    logic gap_start;
    logic gap_busy;
    logic gap_done;

    logic fc_start;
    logic fc_busy;
    logic fc_done;

    logic conv1_in_rd_en;
    logic [ADDR_W-1:0] conv1_in_rd_addr;
    logic conv1_out_wr_en;
    logic [ADDR_W-1:0] conv1_out_wr_addr;
    logic [DATA_W-1:0] conv1_out_wr_data;

    logic dws_in_rd_en;
    logic [ADDR_W-1:0] dws_in_rd_addr;
    logic dws_out_wr_en;
    logic [ADDR_W-1:0] dws_out_wr_addr;
    logic [DATA_W-1:0] dws_out_wr_data;

    logic gap_in_rd_en;
    logic [ADDR_W-1:0] gap_in_rd_addr;
    logic gap_out_wr_en;
    logic [ADDR_W-1:0] gap_out_wr_addr;
    logic [DATA_W-1:0] gap_out_wr_data;

    logic fc_in_rd_en;
    logic [ADDR_W-1:0] fc_in_rd_addr;
    logic fc_out_wr_en;
    logic [ADDR_W-1:0] fc_out_wr_addr;
    logic [DATA_W-1:0] fc_out_wr_data;

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

    logic signed [OC_PAR*DATA_W*9-1:0] conv1_weight_flat_vec;
    logic signed [OC_PAR*ACC_W-1:0] conv1_bias_acc_vec;
    logic signed [OC_PAR*MUL_W-1:0] conv1_mul_vec;
    logic signed [OC_PAR*BIAS_W-1:0] conv1_bias_requant_vec;
    logic [OC_PAR*SHIFT_W-1:0] conv1_shift_vec;
    logic signed [OC_PAR*DATA_W-1:0] conv1_relu6_max_vec;

    logic signed [DATA_W*9-1:0] dw_weight_flat;
    logic signed [MUL_W-1:0] dw_mul;
    logic signed [BIAS_W-1:0] dw_bias;
    logic [SHIFT_W-1:0] dw_shift;
    logic signed [DATA_W-1:0] dw_relu6_max;

    logic signed [DATA_W-1:0] pw_weight;
    logic signed [ACC_W-1:0] pw_bias_acc;
    logic signed [MUL_W-1:0] pw_mul;
    logic signed [BIAS_W-1:0] pw_bias_requant;
    logic [SHIFT_W-1:0] pw_shift;
    logic signed [DATA_W-1:0] pw_relu6_max;

    logic signed [MUL_W-1:0] gap_mul;
    logic signed [BIAS_W-1:0] gap_bias;
    logic [SHIFT_W-1:0] gap_shift;

    logic signed [DATA_W-1:0] fc_weight;
    logic signed [MUL_W-1:0] fc_mul;
    logic signed [BIAS_W-1:0] fc_bias;
    logic [SHIFT_W-1:0] fc_shift;

    mobilenet_v1_ctrl #(
        .DIM_W(DIM_W),
        .ADDR_W(ADDR_W),
        .TILE_H(TILE_H),
        .TILE_W(TILE_W)
    ) u_ctrl (
        .clk(clk),
        .rst_n(rst_n),
        .start(core_start),
        .busy(busy),
        .done(core_done),
        .cfg_in_img_h(cfg_in_img_h),
        .cfg_in_img_w(cfg_in_img_w),
        .cfg_fm_base0(cfg_fm_base0),
        .cfg_fm_base1(cfg_fm_base1),
        .cfg_dw_buf_base(cfg_dw_buf_base),
        .layer_is_conv1(layer_is_conv1),
        .layer_idx(layer_idx),
        .cur_in_h(cur_in_h),
        .cur_in_w(cur_in_w),
        .cur_out_h(cur_out_h),
        .cur_out_w(cur_out_w),
        .cur_in_c(cur_in_c),
        .cur_out_c(cur_out_c),
        .cur_stride(cur_stride),
        .tile_in_row(tile_in_row),
        .tile_in_col(tile_in_col),
        .tile_in_h(tile_in_h),
        .tile_in_w(tile_in_w),
        .tile_out_row(tile_out_row),
        .tile_out_col(tile_out_col),
        .tile_out_h(tile_out_h),
        .tile_out_w(tile_out_w),
        .in_base_addr(in_base_addr),
        .out_base_addr(out_base_addr),
        .dw_buf_base_addr(dw_buf_base_addr),
        .conv1_start(conv1_start),
        .conv1_busy(conv1_busy),
        .conv1_done(conv1_done),
        .dws_start(dws_start),
        .dws_busy(dws_busy),
        .dws_done(dws_done)
    );

    conv1_tile_runner #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .MUL_W(MUL_W),
        .BIAS_W(BIAS_W),
        .SHIFT_W(SHIFT_W),
        .ADDR_W(ADDR_W),
        .DIM_W(DIM_W),
        .OC_PAR(OC_PAR)
    ) u_conv1 (
        .clk(clk),
        .rst_n(rst_n),
        .start(conv1_start),
        .busy(conv1_busy),
        .done(conv1_done),
        .cfg_in_img_h(cur_in_h),
        .cfg_in_img_w(cur_in_w),
        .cfg_out_img_h(cur_out_h),
        .cfg_out_img_w(cur_out_w),
        .cfg_tile_in_row(tile_in_row),
        .cfg_tile_in_col(tile_in_col),
        .cfg_tile_in_h(tile_in_h),
        .cfg_tile_in_w(tile_in_w),
        .cfg_tile_out_row(tile_out_row),
        .cfg_tile_out_col(tile_out_col),
        .cfg_tile_out_h(tile_out_h),
        .cfg_tile_out_w(tile_out_w),
        .cfg_in_channels(cur_in_c),
        .cfg_out_channels(cur_out_c),
        .cfg_stride(cur_stride),
        .cfg_in_base_addr(in_base_addr),
        .cfg_out_base_addr(out_base_addr),
        .in_rd_en(conv1_in_rd_en),
        .in_rd_addr(conv1_in_rd_addr),
        .in_rd_data(fm_rd_data),
        .out_wr_en(conv1_out_wr_en),
        .out_wr_addr(conv1_out_wr_addr),
        .out_wr_data(conv1_out_wr_data),
        .weight_flat_vec(conv1_weight_flat_vec),
        .bias_acc_vec(conv1_bias_acc_vec),
        .mul_vec(conv1_mul_vec),
        .bias_requant_vec(conv1_bias_requant_vec),
        .shift_vec(conv1_shift_vec),
        .relu6_max_vec(conv1_relu6_max_vec),
        .ic_idx(conv1_ic_idx),
        .oc_group_idx(conv1_oc_group_idx)
    );

    dws_tile_runner #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .MUL_W(MUL_W),
        .BIAS_W(BIAS_W),
        .SHIFT_W(SHIFT_W),
        .ADDR_W(ADDR_W),
        .DIM_W(DIM_W),
        .PW_GROUP(PW_GROUP)
    ) u_dws (
        .clk(clk),
        .rst_n(rst_n),
        .start(dws_start),
        .busy(dws_busy),
        .done(dws_done),
        .cfg_in_img_h(cur_in_h),
        .cfg_in_img_w(cur_in_w),
        .cfg_out_img_h(cur_out_h),
        .cfg_out_img_w(cur_out_w),
        .cfg_tile_in_row(tile_in_row),
        .cfg_tile_in_col(tile_in_col),
        .cfg_tile_in_h(tile_in_h),
        .cfg_tile_in_w(tile_in_w),
        .cfg_tile_out_row(tile_out_row),
        .cfg_tile_out_col(tile_out_col),
        .cfg_tile_out_h(tile_out_h),
        .cfg_tile_out_w(tile_out_w),
        .cfg_in_channels(cur_in_c),
        .cfg_out_channels(cur_out_c),
        .cfg_stride(cur_stride),
        .cfg_in_base_addr(in_base_addr),
        .cfg_out_base_addr(out_base_addr),
        .cfg_dw_buf_base_addr(dw_buf_base_addr),
        .in_rd_en(dws_in_rd_en),
        .in_rd_addr(dws_in_rd_addr),
        .in_rd_data(fm_rd_data),
        .dw_buf_wr_en(dw_buf_wr_en),
        .dw_buf_wr_addr(dw_buf_wr_addr),
        .dw_buf_wr_data(dw_buf_wr_data),
        .dw_buf_rd_en(dw_buf_rd_en),
        .dw_buf_rd_addr(dw_buf_rd_addr),
        .dw_buf_rd_data(dw_buf_rd_data),
        .out_wr_en(dws_out_wr_en),
        .out_wr_addr(dws_out_wr_addr),
        .out_wr_data(dws_out_wr_data),
        .dw_weight_flat(dw_weight_flat),
        .dw_mul(dw_mul),
        .dw_bias(dw_bias),
        .dw_shift(dw_shift),
        .dw_relu6_max(dw_relu6_max),
        .pw_weight(pw_weight),
        .pw_bias_acc(pw_bias_acc),
        .pw_mul(pw_mul),
        .pw_bias_requant(pw_bias_requant),
        .pw_shift(pw_shift),
        .pw_relu6_max(pw_relu6_max),
        .dw_ch_idx(dw_ch_idx),
        .pw_in_ch_idx(pw_in_ch_idx),
        .pw_out_ch_idx(pw_out_ch_idx),
        .pw_group_req(pw_group_req),
        .pw_group_idx(pw_group_idx),
        .pw_group_ready(pw_group_ready)
    );

    gap_runner #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .MUL_W(MUL_W),
        .BIAS_W(BIAS_W),
        .SHIFT_W(SHIFT_W),
        .ADDR_W(ADDR_W),
        .DIM_W(DIM_W)
    ) u_gap (
        .clk(clk),
        .rst_n(rst_n),
        .start(gap_start),
        .busy(gap_busy),
        .done(gap_done),
        .cfg_in_h(cur_in_h),
        .cfg_in_w(cur_in_w),
        .cfg_in_c(cur_in_c),
        .cfg_in_base(in_base_addr),
        .cfg_out_base(out_base_addr),
        .in_rd_en(gap_in_rd_en),
        .in_rd_addr(gap_in_rd_addr),
        .in_rd_data(fm_rd_data),
        .out_wr_en(gap_out_wr_en),
        .out_wr_addr(gap_out_wr_addr),
        .out_wr_data(gap_out_wr_data),
        .gap_ch_idx(gap_ch_idx),
        .gap_mul(gap_mul),
        .gap_bias(gap_bias),
        .gap_shift(gap_shift)
    );

    fc_runner #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .MUL_W(MUL_W),
        .BIAS_W(BIAS_W),
        .SHIFT_W(SHIFT_W),
        .ADDR_W(ADDR_W),
        .DIM_W(DIM_W)
    ) u_fc (
        .clk(clk),
        .rst_n(rst_n),
        .start(fc_start),
        .busy(fc_busy),
        .done(fc_done),
        .cfg_in_c(cur_in_c),
        .cfg_out_c(FC_OUT_CH[DIM_W-1:0]),
        .cfg_in_base(out_base_addr),
        .cfg_out_base(in_base_addr),
        .in_rd_en(fc_in_rd_en),
        .in_rd_addr(fc_in_rd_addr),
        .in_rd_data(fm_rd_data),
        .out_wr_en(fc_out_wr_en),
        .out_wr_addr(fc_out_wr_addr),
        .out_wr_data(fc_out_wr_data),
        .fc_in_idx(fc_in_idx),
        .fc_out_idx(fc_out_idx),
        .fc_weight(fc_weight),
        .fc_mul(fc_mul),
        .fc_bias(fc_bias),
        .fc_shift(fc_shift)
    );

    mobilenet_v1_param_cache #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .MUL_W(MUL_W),
        .BIAS_W(BIAS_W),
        .SHIFT_W(SHIFT_W),
        .OC_PAR(OC_PAR),
        .DIM_W(DIM_W),
        .WR_ADDR_W(20),
        .WR_W(32),
        .PW_GROUP(PW_GROUP),
        .FC_OUT_CH(FC_OUT_CH),
        .INIT_CONV1_W(INIT_CONV1_W),
        .INIT_CONV1_BIAS_ACC(INIT_CONV1_BIAS_ACC),
        .INIT_CONV1_MUL(INIT_CONV1_MUL),
        .INIT_CONV1_BIAS_RQ(INIT_CONV1_BIAS_RQ),
        .INIT_CONV1_SHIFT(INIT_CONV1_SHIFT),
        .INIT_CONV1_RELU6(INIT_CONV1_RELU6),
        .INIT_DW_W(INIT_DW_W),
        .INIT_DW_MUL(INIT_DW_MUL),
        .INIT_DW_BIAS(INIT_DW_BIAS),
        .INIT_DW_SHIFT(INIT_DW_SHIFT),
        .INIT_DW_RELU6(INIT_DW_RELU6),
        .INIT_PW_W(INIT_PW_W),
        .INIT_PW_BIAS_ACC(INIT_PW_BIAS_ACC),
        .INIT_PW_MUL(INIT_PW_MUL),
        .INIT_PW_BIAS_RQ(INIT_PW_BIAS_RQ),
        .INIT_PW_SHIFT(INIT_PW_SHIFT),
        .INIT_PW_RELU6(INIT_PW_RELU6),
        .INIT_GAP_MUL(INIT_GAP_MUL),
        .INIT_GAP_BIAS(INIT_GAP_BIAS),
        .INIT_GAP_SHIFT(INIT_GAP_SHIFT),
        .INIT_FC_W(INIT_FC_W),
        .INIT_FC_MUL(INIT_FC_MUL),
        .INIT_FC_BIAS(INIT_FC_BIAS),
        .INIT_FC_SHIFT(INIT_FC_SHIFT)
    ) u_params (
        .clk(clk),
        .rst_n(rst_n),
        .wr_en(param_wr_en),
        .wr_sel(param_wr_sel),
        .wr_addr(param_wr_addr),
        .wr_data(param_wr_data),
        .layer_idx(layer_idx),
        .layer_in_c(cur_in_c),
        .layer_out_c(cur_out_c),
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
        .conv1_weight_flat_vec(conv1_weight_flat_vec),
        .conv1_bias_acc_vec(conv1_bias_acc_vec),
        .conv1_mul_vec(conv1_mul_vec),
        .conv1_bias_requant_vec(conv1_bias_requant_vec),
        .conv1_shift_vec(conv1_shift_vec),
        .conv1_relu6_max_vec(conv1_relu6_max_vec),
        .dw_weight_flat(dw_weight_flat),
        .dw_mul(dw_mul),
        .dw_bias(dw_bias),
        .dw_shift(dw_shift),
        .dw_relu6_max(dw_relu6_max),
        .pw_weight(pw_weight),
        .pw_bias_acc(pw_bias_acc),
        .pw_mul(pw_mul),
        .pw_bias_requant(pw_bias_requant),
        .pw_shift(pw_shift),
        .pw_relu6_max(pw_relu6_max),
        .gap_mul(gap_mul),
        .gap_bias(gap_bias),
        .gap_shift(gap_shift),
        .fc_weight(fc_weight),
        .fc_mul(fc_mul),
        .fc_bias(fc_bias),
        .fc_shift(fc_shift)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            top_state <= TOP_IDLE;
            done <= 1'b0;
            core_start <= 1'b0;
            gap_start <= 1'b0;
            fc_start <= 1'b0;
        end else begin
            done <= 1'b0;
            core_start <= 1'b0;
            gap_start <= 1'b0;
            fc_start <= 1'b0;

            case (top_state)
                TOP_IDLE: begin
                    if (start) begin
                        core_start <= 1'b1;
                        top_state <= TOP_CORE;
                    end
                end
                TOP_CORE: begin
                    if (core_done) begin
                        gap_start <= 1'b1;
                        top_state <= TOP_GAP;
                    end
                end
                TOP_GAP: begin
                    if (gap_done) begin
                        fc_start <= 1'b1;
                        top_state <= TOP_FC;
                    end
                end
                TOP_FC: begin
                    if (fc_done) begin
                        top_state <= TOP_DONE;
                    end
                end
                TOP_DONE: begin
                    done <= 1'b1;
                    top_state <= TOP_IDLE;
                end
                default: begin
                    top_state <= TOP_IDLE;
                end
            endcase
        end
    end

    always_comb begin
        fm_rd_en = 1'b0;
        fm_rd_addr = '0;
        fm_wr_en = 1'b0;
        fm_wr_addr = '0;
        fm_wr_data = '0;

        case (top_state)
            TOP_CORE: begin
                fm_rd_en = layer_is_conv1 ? conv1_in_rd_en : dws_in_rd_en;
                fm_rd_addr = layer_is_conv1 ? conv1_in_rd_addr : dws_in_rd_addr;
                fm_wr_en = layer_is_conv1 ? conv1_out_wr_en : dws_out_wr_en;
                fm_wr_addr = layer_is_conv1 ? conv1_out_wr_addr : dws_out_wr_addr;
                fm_wr_data = layer_is_conv1 ? conv1_out_wr_data : dws_out_wr_data;
            end
            TOP_GAP: begin
                fm_rd_en = gap_in_rd_en;
                fm_rd_addr = gap_in_rd_addr;
                fm_wr_en = gap_out_wr_en;
                fm_wr_addr = gap_out_wr_addr;
                fm_wr_data = gap_out_wr_data;
            end
            TOP_FC: begin
                fm_rd_en = fc_in_rd_en;
                fm_rd_addr = fc_in_rd_addr;
                fm_wr_en = fc_out_wr_en;
                fm_wr_addr = fc_out_wr_addr;
                fm_wr_data = fc_out_wr_data;
            end
            default: begin
            end
        endcase
    end
endmodule
