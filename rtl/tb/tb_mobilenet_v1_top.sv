`timescale 1ns/1ps

module tb_mobilenet_v1_top;
    localparam int DATA_W = 8;
    localparam int ACC_W = 32;
    localparam int MUL_W = 32;
    localparam int BIAS_W = 32;
    localparam int SHIFT_W = 6;
    localparam int ADDR_W = 32;
    localparam int DIM_W = 16;
    localparam int OC_PAR = 16;
    localparam int PW_GROUP = 32;
    localparam int PW_OC_PAR = 32;
    localparam int PW_IC_PAR = 16;
    localparam int FC_OUT_CH = 1000;
    localparam int TILE_H = 16;
    localparam int TILE_W = 16;
    localparam int INPUT_ZP = -1;
    localparam int ACT_ZP = -128;
    localparam string INPUT_MEM = "rtl/mem/input_rand.mem";
    localparam string FC_OUT_MEM = "rtl/mem/fc_out_hw.mem";
    localparam string FC_LOGITS_MEM = "rtl/mem/fc_logits_hw.mem";
    localparam string TILE_MASK_MEM = "rtl/mem/tile_mask.mem";

    logic clk;
    logic rst_n;
    logic start;
    logic done;
    logic tile_skip_en;

    logic [DIM_W-1:0] cfg_in_img_h;
    logic [DIM_W-1:0] cfg_in_img_w;

    logic [ADDR_W-1:0] cfg_fm_base0;
    logic [ADDR_W-1:0] cfg_fm_base1;
    logic [ADDR_W-1:0] cfg_dw_buf_base;

    logic param_wr_en;
    logic [4:0] param_wr_sel;
    logic [19:0] param_wr_addr;
    logic [31:0] param_wr_data;

    logic fm_rd_en;
    logic [ADDR_W-1:0] fm_rd_addr;
    logic [DATA_W-1:0] fm_rd_data;

    logic fm_wr_en0;
    logic [ADDR_W-1:0] fm_wr_addr0;
    logic [DATA_W-1:0] fm_wr_data0;
    logic fm_wr_en1;
    logic [ADDR_W-1:0] fm_wr_addr1;
    logic [DATA_W-1:0] fm_wr_data1;

    logic [PW_IC_PAR-1:0] dw_buf_rd_en;
    logic [PW_IC_PAR*ADDR_W-1:0] dw_buf_rd_addr;
    logic [PW_IC_PAR*DATA_W-1:0] dw_buf_rd_data;

    logic dw_buf_wr_en;
    logic [ADDR_W-1:0] dw_buf_wr_addr;
    logic [DATA_W-1:0] dw_buf_wr_data;

    function automatic signed [31:0] srdhm_tb(
        input signed [31:0] a,
        input signed [31:0] b
    );
        logic signed [63:0] ab;
        logic signed [63:0] nudge;
        logic signed [63:0] res;
        begin
            ab = $signed(a) * $signed(b);
            if (ab >= 0) begin
                nudge = 64'sd1073741824;
            end else begin
                nudge = -64'sd1073741823;
            end
            res = (ab + nudge) >>> 31;
            if (res > 64'sd2147483647) begin
                res = 64'sd2147483647;
            end else if (res < -64'sd2147483648) begin
                res = -64'sd2147483648;
            end
            srdhm_tb = res[31:0];
        end
    endfunction

    function automatic signed [31:0] rdivp_tb(
        input signed [31:0] x,
        input int unsigned shift_amt
    );
        logic signed [31:0] mask;
        logic signed [31:0] remainder;
        logic signed [31:0] threshold;
        begin
            if (shift_amt == 0) begin
                rdivp_tb = x;
            end else begin
                mask = (32'sd1 <<< shift_amt) - 1;
                remainder = x & mask;
                threshold = (mask >>> 1);
                if (x < 0) begin
                    threshold = threshold + 1;
                end
                rdivp_tb = (x >>> shift_amt) + ((remainder > threshold) ? 1 : 0);
            end
        end
    endfunction

    mobilenet_v1_top #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .MUL_W(MUL_W),
        .BIAS_W(BIAS_W),
        .SHIFT_W(SHIFT_W),
        .ADDR_W(ADDR_W),
        .DIM_W(DIM_W),
        .OC_PAR(OC_PAR),
        .PW_GROUP(PW_GROUP),
        .PW_OC_PAR(PW_OC_PAR),
        .PW_IC_PAR(PW_IC_PAR),
        .FC_OUT_CH(FC_OUT_CH),
        .TILE_H(TILE_H),
        .TILE_W(TILE_W),
        .TILE_MASK_DEPTH(4096),
        .INPUT_ZP(INPUT_ZP),
        .ACT_ZP(ACT_ZP),
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
        .INIT_FC_ZP("rtl/mem/fc_zp.mem"),
        .INIT_TILE_MASK(TILE_MASK_MEM)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .tile_skip_en(tile_skip_en),
        .cfg_in_img_h(cfg_in_img_h),
        .cfg_in_img_w(cfg_in_img_w),
        .cfg_fm_base0(cfg_fm_base0),
        .cfg_fm_base1(cfg_fm_base1),
        .cfg_dw_buf_base(cfg_dw_buf_base),
        .param_wr_en(param_wr_en),
        .param_wr_sel(param_wr_sel),
        .param_wr_addr(param_wr_addr),
        .param_wr_data(param_wr_data),
        .fm_rd_en(fm_rd_en),
        .fm_rd_addr(fm_rd_addr),
        .fm_rd_data(fm_rd_data),
        .fm_wr_en0(fm_wr_en0),
        .fm_wr_addr0(fm_wr_addr0),
        .fm_wr_data0(fm_wr_data0),
        .fm_wr_en1(fm_wr_en1),
        .fm_wr_addr1(fm_wr_addr1),
        .fm_wr_data1(fm_wr_data1),
        .dw_buf_rd_en(dw_buf_rd_en),
        .dw_buf_rd_addr(dw_buf_rd_addr),
        .dw_buf_rd_data(dw_buf_rd_data),
        .dw_buf_wr_en(dw_buf_wr_en),
        .dw_buf_wr_addr(dw_buf_wr_addr),
        .dw_buf_wr_data(dw_buf_wr_data)
    );

    always #5 clk = ~clk;

    integer cycle;
    int img_h_arg;
    int img_w_arg;
    int max_cycles;
    bit verbose;
    longint signed acc64;
    longint signed mult64;
    longint signed scaled64;
    longint signed shifted64;
    logic [DIM_W-1:0] last_layer_idx;
    logic last_layer_is_conv1;
    integer layer_start_cycle;
    integer layer_tile_count;
    bit layer_seen;
    logic last_done;
    logic [2:0] last_top_state;
    logic last_dws_start;
    logic last_dws_busy;
    logic last_pw_group_ready;
    logic [DIM_W-1:0] last_pw_group_idx;
    logic [2:0] last_ctrl_state;
    logic last_tile_valid;
    logic last_tile_ready;
    logic last_tile_done;
    logic last_tile_active;
    logic [2:0] last_dws_state;
    logic [DIM_W-1:0] last_dw_ch_idx;
    logic [DIM_W-1:0] last_pw_out_ch_idx;
    logic [DIM_W-1:0] last_pw_in_ch_idx;
    logic last_dw_reader_done;
    logic last_dw_writer_done;
    logic last_pw_reader_done;
    logic last_pw_writer_done;
    integer last_heartbeat;
    logic [2:0] last_gap_state;
    logic [DIM_W-1:0] last_gap_ch_idx;
    logic last_gap_busy;
    logic [2:0] last_fc_state;
    logic [DIM_W-1:0] last_fc_out_idx;
    logic last_fc_busy;
    integer last_gap_heartbeat;
    integer gap_debug_window;
    logic base_sel_tb;
    integer layer_fh;
    integer layer_base;
    integer layer_plane;
    integer layer_total;
    string layer_fname;
    logic signed [31:0] fc_logits_mem [0:FC_OUT_CH-1];
    bit dumped_dw;
    bit dumped_pw_q;
    bit dumped_pw_acc;
    integer dw_dump_fh;
    integer dw_dump_total;
    integer dw_dump_i;
    integer pw_q_fh;
    integer pw_acc_fh;
    integer dw_wr_seen;
    localparam int TARGET_CH = 13;
    localparam int TARGET_ROW = 98;
    localparam int TARGET_COL = 105;
    localparam int TARGET_PW_PIX = 73;
    localparam int DWS_S_IDLE = 0;
    localparam int DWS_S_DW_RUN = 1;
    localparam int DWS_S_PW_LOAD = 2;
    localparam int DWS_S_PW_RUN = 3;
    localparam int DWS_S_PW_WRITE = 4;
    localparam int DWS_S_DONE = 5;
    logic [ADDR_W-1:0] target_addr;
    integer target_hits;
    logic [DATA_W-1:0] target_last_data;
    integer target_q_hits;
    logic signed [DATA_W-1:0] target_last_q;
    integer target_acc_hits;
    logic signed [ACC_W-1:0] target_last_acc;
    integer pw_acc_pix_idx;
    integer dws_dw_run_cycles;
    integer dws_pw_load_cycles;
    integer dws_pw_run_cycles;
    integer dws_pw_write_cycles;
    integer dws_other_cycles;

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        img_h_arg = 16;
        img_w_arg = 16;
        max_cycles = 50000000;
        verbose = 1'b0;
        tile_skip_en = 1'b0;
        void'($value$plusargs("IMG_H=%d", img_h_arg));
        void'($value$plusargs("IMG_W=%d", img_w_arg));
        void'($value$plusargs("MAX_CYCLES=%d", max_cycles));
        if ($test$plusargs("TILE_SKIP")) begin
            tile_skip_en = 1'b1;
        end
        if ($test$plusargs("VERBOSE")) begin
            verbose = 1'b1;
        end
        cfg_in_img_h = img_h_arg[DIM_W-1:0];
        cfg_in_img_w = img_w_arg[DIM_W-1:0];
        cfg_fm_base0 = '0;
        cfg_fm_base1 = 32'h0010_0000;
        cfg_dw_buf_base = 32'h0020_0000;
        param_wr_en = 1'b0;
        param_wr_sel = '0;
        param_wr_addr = '0;
        param_wr_data = '0;
        cycle = 0;
        last_layer_idx = '1;
        last_layer_is_conv1 = 1'b0;
        layer_start_cycle = 0;
        layer_tile_count = 0;
        layer_seen = 1'b0;
        last_done = 1'b0;
        last_top_state = '1;
        last_dws_start = 1'b0;
        last_dws_busy = 1'b0;
        last_pw_group_ready = 1'b0;
        last_pw_group_idx = '1;
        last_ctrl_state = '1;
        last_tile_valid = 1'b0;
        last_tile_ready = 1'b0;
        last_tile_done = 1'b0;
        last_tile_active = 1'b0;
        last_dws_state = '1;
        last_dw_ch_idx = '1;
        last_pw_out_ch_idx = '1;
        last_pw_in_ch_idx = '1;
        last_dw_reader_done = 1'b0;
        last_dw_writer_done = 1'b0;
        last_pw_reader_done = 1'b0;
        last_pw_writer_done = 1'b0;
        last_heartbeat = 0;
        last_gap_state = '1;
        last_gap_ch_idx = '1;
        last_gap_busy = 1'b0;
        last_fc_state = '1;
        last_fc_out_idx = '1;
        last_fc_busy = 1'b0;
        last_gap_heartbeat = 0;
        gap_debug_window = 0;
        base_sel_tb = 1'b0;
        dumped_dw = 1'b0;
        dumped_pw_q = 1'b0;
        dumped_pw_acc = 1'b0;
        target_hits = 0;
        target_q_hits = 0;
        target_last_data = '0;
        target_last_q = '0;
        target_acc_hits = 0;
        target_last_acc = '0;
        pw_acc_pix_idx = 0;
        dws_dw_run_cycles = 0;
        dws_pw_load_cycles = 0;
        dws_pw_run_cycles = 0;
        dws_pw_write_cycles = 0;
        dws_other_cycles = 0;
        dw_wr_seen = 0;
        for (mi = 0; mi < FC_OUT_CH; mi = mi + 1) begin
            fc_logits_mem[mi] = '0;
        end

        #20;
        rst_n = 1'b1;
        #10;
        start = 1'b1;
        #10;
        start = 1'b0;
    end

    always @(posedge clk) begin
        cycle <= cycle + 1;
        if (!dut.layer_is_conv1 && dut.dws_busy) begin
            case (dut.u_dws.state)
                DWS_S_DW_RUN: dws_dw_run_cycles <= dws_dw_run_cycles + 1;
                DWS_S_PW_LOAD: dws_pw_load_cycles <= dws_pw_load_cycles + 1;
                DWS_S_PW_RUN: dws_pw_run_cycles <= dws_pw_run_cycles + 1;
                DWS_S_PW_WRITE: dws_pw_write_cycles <= dws_pw_write_cycles + 1;
                default: dws_other_cycles <= dws_other_cycles + 1;
            endcase
        end
        if (dut.u_ctrl.tile_valid && dut.u_ctrl.tile_ready) begin
            layer_tile_count <= layer_tile_count + 1;
        end
        if (dut.u_ctrl.state == 3'd4 && last_ctrl_state != 3'd4) begin
            layer_plane = dut.cur_out_h * dut.cur_out_w;
            layer_total = layer_plane * dut.cur_out_c;
            layer_base = base_sel_tb ? cfg_fm_base0 : cfg_fm_base1;
            layer_fname = $sformatf("rtl/mem/layer%0d_out_hw.mem", dut.layer_idx);
            layer_fh = $fopen(layer_fname, "w");
            for (out_i = 0; out_i < layer_total; out_i = out_i + 1) begin
                $fdisplay(layer_fh, "%02x", fm_mem[layer_base + out_i]);
            end
            $fclose(layer_fh);
            base_sel_tb <= ~base_sel_tb;
        end
        if (!dumped_dw &&
            (dut.u_ctrl.state == 3'd4) &&
            (last_ctrl_state != 3'd4) &&
            (dut.layer_idx == 1)) begin
            dw_dump_fh = $fopen("rtl/mem/layer1_dw_hw.mem", "w");
            dw_dump_total = dut.cur_in_c * dut.cur_out_h * dut.cur_out_w;
            for (dw_dump_i = 0; dw_dump_i < dw_dump_total; dw_dump_i = dw_dump_i + 1) begin
                $fdisplay(dw_dump_fh, "%02x", dw_mem[cfg_dw_buf_base + dw_dump_i]);
            end
            $fclose(dw_dump_fh);
            dumped_dw <= 1'b1;
        end
        if (!dumped_pw_q &&
            (dut.layer_idx == 1) &&
            dut.u_dws.pw_q_valid &&
            (dut.u_dws.pw_pix_idx == 0)) begin
            pw_q_fh = $fopen("rtl/mem/layer1_pw_qvec_hw.mem", "w");
            for (out_i = 0; out_i < PW_OC_PAR; out_i = out_i + 1) begin
                $fdisplay(pw_q_fh, "%02x", dut.u_dws.pw_q_vec[out_i*DATA_W +: DATA_W]);
            end
            $fclose(pw_q_fh);
            $display("PW_Q capture: in_ch_idx=%0d", dut.u_dws.pw_in_ch_idx);
            dumped_pw_q <= 1'b1;
        end
        if (!dumped_pw_acc &&
            (dut.layer_idx == 1) &&
            dut.u_dws.pw_acc_valid &&
            (dut.u_dws.pw_pix_idx == 0)) begin
            pw_acc_fh = $fopen("rtl/mem/layer1_pw_acc_hw.mem", "w");
            for (out_i = 0; out_i < PW_OC_PAR; out_i = out_i + 1) begin
                $fdisplay(pw_acc_fh, "%08x", dut.u_dws.pw_acc_vec[out_i*ACC_W +: ACC_W]);
            end
            $fclose(pw_acc_fh);
            $display("PW_ACC capture: in_ch_idx=%0d last_in_ch=%0d", dut.u_dws.pw_in_ch_idx, dut.u_dws.pw_in_last);
            dumped_pw_acc <= 1'b1;
        end
        if (dut.layer_idx == 1) begin
            if (dut.u_dws.pw_start_pulse) begin
                pw_acc_pix_idx <= 0;
            end else if (dut.u_dws.pw_acc_valid) begin
                pw_acc_pix_idx <= pw_acc_pix_idx + 1;
            end
            target_addr <= dut.out_base_addr +
                           (TARGET_CH * (dut.cur_out_h * dut.cur_out_w)) +
                           (TARGET_ROW * dut.cur_out_w) +
                           TARGET_COL;
            if (fm_wr_en0 && (fm_wr_addr0 == target_addr)) begin
                target_hits <= target_hits + 1;
                target_last_data <= fm_wr_data0;
                $display("TARGET write0 cycle=%0d data=%0d (0x%02x) tile_row=%0d tile_col=%0d pw_write_oc=%0d pw_write_idx=%0d pw_out_ch_idx=%0d",
                         cycle, $signed(fm_wr_data0), fm_wr_data0,
                         dut.u_dws.tile_out_row_reg, dut.u_dws.tile_out_col_reg,
                         dut.u_dws.pw_write_oc, dut.u_dws.pw_write_idx, dut.u_dws.pw_out_ch_idx);
            end
            if (fm_wr_en1 && (fm_wr_addr1 == target_addr)) begin
                target_hits <= target_hits + 1;
                target_last_data <= fm_wr_data1;
                $display("TARGET write1 cycle=%0d data=%0d (0x%02x) tile_row=%0d tile_col=%0d pw_write_oc=%0d pw_write_idx=%0d pw_out_ch_idx=%0d",
                         cycle, $signed(fm_wr_data1), fm_wr_data1,
                         dut.u_dws.tile_out_row_reg, dut.u_dws.tile_out_col_reg,
                         dut.u_dws.pw_write_oc, dut.u_dws.pw_write_idx, dut.u_dws.pw_out_ch_idx);
            end
            if (dut.u_dws.pw_acc_valid &&
                (pw_acc_pix_idx == TARGET_PW_PIX) &&
                (dut.u_dws.pw_out_ch_idx <= TARGET_CH) &&
                (TARGET_CH < (dut.u_dws.pw_out_ch_idx + PW_OC_PAR)) &&
                (dut.u_dws.tile_out_row_reg == 96) &&
                (dut.u_dws.tile_out_col_reg == 96)) begin
                int target_ch_local;
                target_ch_local = TARGET_CH - dut.u_dws.pw_out_ch_idx;
                target_acc_hits <= target_acc_hits + 1;
                target_last_acc <= $signed(dut.u_dws.pw_acc_vec[target_ch_local*ACC_W +: ACC_W]);
                $display("TARGET acc cycle=%0d acc=%0d (0x%08x) pix_idx=%0d out_ch_base=%0d",
                         cycle,
                         $signed(dut.u_dws.pw_acc_vec[target_ch_local*ACC_W +: ACC_W]),
                         dut.u_dws.pw_acc_vec[target_ch_local*ACC_W +: ACC_W],
                         pw_acc_pix_idx,
                         dut.u_dws.pw_out_ch_idx);
            end
            if (dut.u_dws.pw_q_valid &&
                (dut.u_dws.pw_pix_idx == TARGET_PW_PIX) &&
                (dut.u_dws.pw_out_ch_idx <= TARGET_CH) &&
                (TARGET_CH < (dut.u_dws.pw_out_ch_idx + PW_OC_PAR)) &&
                (dut.u_dws.tile_out_row_reg == 96) &&
                (dut.u_dws.tile_out_col_reg == 96)) begin
                int target_ch_local;
                target_ch_local = TARGET_CH - dut.u_dws.pw_out_ch_idx;
                target_q_hits <= target_q_hits + 1;
                target_last_q <= $signed(dut.u_dws.pw_q_vec[target_ch_local*DATA_W +: DATA_W]);
                $display("TARGET qvec cycle=%0d q=%0d (0x%02x) pix_idx=%0d out_ch_base=%0d",
                         cycle,
                         $signed(dut.u_dws.pw_q_vec[target_ch_local*DATA_W +: DATA_W]),
                         dut.u_dws.pw_q_vec[target_ch_local*DATA_W +: DATA_W],
                         dut.u_dws.pw_pix_idx,
                         dut.u_dws.pw_out_ch_idx);
            end
        end
        if (dw_buf_wr_en) begin
            dw_wr_seen <= dw_wr_seen + 1;
        end
        if (dut.top_state != last_top_state) begin
            if (verbose) begin
                $display("TOP state=%0d at cycle %0d", dut.top_state, cycle);
            end
            last_top_state <= dut.top_state;
        end
        if (dut.u_ctrl.state != last_ctrl_state) begin
            if (verbose) begin
                $display("CTRL state=%0d at cycle %0d", dut.u_ctrl.state, cycle);
            end
            last_ctrl_state <= dut.u_ctrl.state;
        end
        if (dut.u_ctrl.tile_valid != last_tile_valid ||
            dut.u_ctrl.tile_ready != last_tile_ready ||
            dut.u_ctrl.tile_done != last_tile_done ||
            dut.u_ctrl.tile_active != last_tile_active) begin
            if (verbose) begin
                $display("tile_valid=%0d ready=%0d done=%0d active=%0d at cycle %0d",
                         dut.u_ctrl.tile_valid,
                         dut.u_ctrl.tile_ready,
                         dut.u_ctrl.tile_done,
                         dut.u_ctrl.tile_active,
                         cycle);
            end
            last_tile_valid <= dut.u_ctrl.tile_valid;
            last_tile_ready <= dut.u_ctrl.tile_ready;
            last_tile_done <= dut.u_ctrl.tile_done;
            last_tile_active <= dut.u_ctrl.tile_active;
        end
        if (dut.layer_idx != last_layer_idx || dut.layer_is_conv1 != last_layer_is_conv1) begin
            if (layer_seen) begin
                $display("LAYER_DONE idx=%0d conv1=%0d cycles=%0d tiles=%0d",
                         last_layer_idx,
                         last_layer_is_conv1,
                         cycle - layer_start_cycle,
                         layer_tile_count);
                if (!last_layer_is_conv1) begin
                    $display("DWS_BREAKDOWN idx=%0d DW=%0d PW_LOAD=%0d PW_RUN=%0d PW_WRITE=%0d OTHER=%0d",
                             last_layer_idx,
                             dws_dw_run_cycles,
                             dws_pw_load_cycles,
                             dws_pw_run_cycles,
                             dws_pw_write_cycles,
                             dws_other_cycles);
                end
            end
            layer_start_cycle <= cycle;
            layer_tile_count <= 0;
            layer_seen <= 1'b1;
            dws_dw_run_cycles <= 0;
            dws_pw_load_cycles <= 0;
            dws_pw_run_cycles <= 0;
            dws_pw_write_cycles <= 0;
            dws_other_cycles <= 0;
            if (verbose) begin
                $display("Layer %0d conv1=%0d in=%0dx%0d c=%0d out=%0dx%0d c=%0d stride=%0d at cycle %0d",
                         dut.layer_idx,
                         dut.layer_is_conv1,
                         dut.cur_in_h, dut.cur_in_w, dut.cur_in_c,
                         dut.cur_out_h, dut.cur_out_w, dut.cur_out_c,
                         dut.cur_stride,
                         cycle);
            end
            last_layer_idx <= dut.layer_idx;
            last_layer_is_conv1 <= dut.layer_is_conv1;
        end
        if (done && !last_done) begin
            if (layer_seen) begin
                $display("LAYER_DONE idx=%0d conv1=%0d cycles=%0d tiles=%0d",
                         last_layer_idx,
                         last_layer_is_conv1,
                         cycle - layer_start_cycle,
                         layer_tile_count);
                if (!last_layer_is_conv1) begin
                    $display("DWS_BREAKDOWN idx=%0d DW=%0d PW_LOAD=%0d PW_RUN=%0d PW_WRITE=%0d OTHER=%0d",
                             last_layer_idx,
                             dws_dw_run_cycles,
                             dws_pw_load_cycles,
                             dws_pw_run_cycles,
                             dws_pw_write_cycles,
                             dws_other_cycles);
                end
            end
            $display("TOTAL_CYCLES %0d", cycle);
        end
        last_done <= done;
        if (dut.conv1_done) begin
            if (verbose) begin
                $display("conv1_done at cycle %0d", cycle);
            end
        end
        if (dut.dws_start && !last_dws_start) begin
            if (verbose) begin
                $display("dws_start at cycle %0d", cycle);
            end
        end
        if (dut.dws_busy != last_dws_busy) begin
            if (verbose) begin
                $display("dws_busy=%0d at cycle %0d", dut.dws_busy, cycle);
            end
            last_dws_busy <= dut.dws_busy;
        end
        if (dut.dws_done) begin
            if (verbose) begin
                $display("dws_done at cycle %0d", cycle);
            end
        end
        if (dut.u_dws.state != last_dws_state) begin
            if (verbose) begin
                $display("DWS state=%0d at cycle %0d", dut.u_dws.state, cycle);
            end
            last_dws_state <= dut.u_dws.state;
        end
        if (dut.u_dws.dw_ch_idx != last_dw_ch_idx ||
            dut.u_dws.pw_out_ch_idx != last_pw_out_ch_idx ||
            dut.u_dws.pw_in_ch_idx != last_pw_in_ch_idx) begin
            if (verbose) begin
                $display("DWS idx dw_ch=%0d pw_out=%0d pw_in=%0d at cycle %0d",
                         dut.u_dws.dw_ch_idx,
                         dut.u_dws.pw_out_ch_idx,
                         dut.u_dws.pw_in_ch_idx,
                         cycle);
            end
            last_dw_ch_idx <= dut.u_dws.dw_ch_idx;
            last_pw_out_ch_idx <= dut.u_dws.pw_out_ch_idx;
            last_pw_in_ch_idx <= dut.u_dws.pw_in_ch_idx;
        end
        if (dut.u_dws.dw_reader_done && !last_dw_reader_done) begin
            if (verbose) begin
                $display("dw_reader_done at cycle %0d", cycle);
            end
        end
        if (dut.u_dws.dw_writer_done && !last_dw_writer_done) begin
            if (verbose) begin
                $display("dw_writer_done at cycle %0d", cycle);
            end
        end
        if (dut.u_dws.pw_reader_done && !last_pw_reader_done) begin
            if (verbose) begin
                $display("pw_reader_done at cycle %0d", cycle);
            end
        end
        if (dut.u_dws.pw_writer_done && !last_pw_writer_done) begin
            if (verbose) begin
                $display("pw_writer_done at cycle %0d", cycle);
            end
        end
        last_dw_reader_done <= dut.u_dws.dw_reader_done;
        last_dw_writer_done <= dut.u_dws.dw_writer_done;
        last_pw_reader_done <= dut.u_dws.pw_reader_done;
        last_pw_writer_done <= dut.u_dws.pw_writer_done;

        if (dut.dws_busy && (cycle - last_heartbeat) >= 200000) begin
            if (verbose) begin
                $display("DWS heartbeat state=%0d dw_ch=%0d pw_out=%0d pw_in=%0d at cycle %0d",
                         dut.u_dws.state,
                         dut.u_dws.dw_ch_idx,
                         dut.u_dws.pw_out_ch_idx,
                         dut.u_dws.pw_in_ch_idx,
                         cycle);
            end
            last_heartbeat = cycle;
        end
        if ((dut.pw_group_idx != last_pw_group_idx) || (dut.pw_group_ready != last_pw_group_ready)) begin
            if (verbose) begin
                $display("pw_group idx=%0d ready=%0d at cycle %0d",
                         dut.pw_group_idx, dut.pw_group_ready, cycle);
            end
            last_pw_group_idx <= dut.pw_group_idx;
            last_pw_group_ready <= dut.pw_group_ready;
        end
        if (dut.core_done) begin
            if (verbose) begin
                $display("core_done at cycle %0d", cycle);
            end
            gap_debug_window <= 50;
        end
        if (dut.gap_done) begin
            if (verbose) begin
                $display("gap_done at cycle %0d", cycle);
            end
        end
        if (dut.fc_done) begin
            if (verbose) begin
                $display("fc_done at cycle %0d", cycle);
            end
        end
        if (dut.gap_start) begin
            if (verbose) begin
                $display("gap_start at cycle %0d", cycle);
            end
        end
        if (dut.gap_busy != last_gap_busy) begin
            if (verbose) begin
                $display("gap_busy=%0d at cycle %0d", dut.gap_busy, cycle);
            end
            last_gap_busy <= dut.gap_busy;
        end
        if (dut.u_gap.state != last_gap_state) begin
            if (verbose) begin
                $display("GAP state=%0d ch=%0d row=%0d col=%0d h=%0d w=%0d c=%0d at cycle %0d",
                         dut.u_gap.state,
                         dut.u_gap.gap_ch_idx,
                         dut.u_gap.row,
                         dut.u_gap.col,
                         dut.u_gap.h_reg,
                         dut.u_gap.w_reg,
                         dut.u_gap.c_reg,
                         cycle);
            end
            last_gap_state <= dut.u_gap.state;
        end
        if (dut.u_gap.gap_ch_idx != last_gap_ch_idx) begin
            if (verbose) begin
                $display("GAP ch=%0d at cycle %0d", dut.u_gap.gap_ch_idx, cycle);
            end
            last_gap_ch_idx <= dut.u_gap.gap_ch_idx;
        end
        if (dut.fc_start) begin
            if (verbose) begin
                $display("fc_start at cycle %0d", cycle);
            end
        end
        if (dut.u_fc.state == 3 && last_fc_state != 3) begin
            acc64 = $signed(dut.u_fc.acc);
            shifted64 = rdivp_tb(srdhm_tb($signed(dut.u_fc.acc), $signed(dut.u_fc.fc_mul)),
                                 dut.u_fc.fc_shift);
            shifted64 = shifted64 + $signed(dut.u_fc.fc_zp);
            fc_logits_mem[dut.u_fc.fc_out_idx] <= shifted64[31:0];
        end
        if (dut.fc_busy != last_fc_busy) begin
            if (verbose) begin
                $display("fc_busy=%0d at cycle %0d", dut.fc_busy, cycle);
            end
            last_fc_busy <= dut.fc_busy;
        end
        if (dut.u_fc.state != last_fc_state) begin
            if (verbose) begin
                $display("FC state=%0d out=%0d in=%0d at cycle %0d",
                         dut.u_fc.state,
                         dut.u_fc.fc_out_idx,
                         dut.u_fc.fc_in_idx,
                         cycle);
            end
            last_fc_state <= dut.u_fc.state;
        end
        if (dut.u_fc.fc_out_idx != last_fc_out_idx) begin
            if (verbose) begin
                $display("FC out_idx=%0d at cycle %0d", dut.u_fc.fc_out_idx, cycle);
            end
            last_fc_out_idx <= dut.u_fc.fc_out_idx;
        end
        if (dut.gap_busy && (cycle - last_gap_heartbeat) >= 200000) begin
            if (verbose) begin
                $display("GAP heartbeat state=%0d ch=%0d row=%0d col=%0d at cycle %0d",
                         dut.u_gap.state,
                         dut.u_gap.gap_ch_idx,
                         dut.u_gap.row,
                         dut.u_gap.col,
                         cycle);
            end
            last_gap_heartbeat = cycle;
        end
        if (gap_debug_window > 0) begin
            if (verbose) begin
                $display("GAP dbg start=%0b state=%0d ch=%0d row=%0d col=%0d busy=%0b at cycle %0d",
                         dut.gap_start,
                         dut.u_gap.state,
                         dut.u_gap.gap_ch_idx,
                         dut.u_gap.row,
                         dut.u_gap.col,
                         dut.gap_busy,
                         cycle);
            end
            gap_debug_window <= gap_debug_window - 1;
        end
        if (dut.gap_done) begin
            layer_fname = "rtl/mem/gap_out_hw.mem";
            layer_fh = $fopen(layer_fname, "w");
            for (out_i = 0; out_i < dut.cur_in_c; out_i = out_i + 1) begin
                $fdisplay(layer_fh, "%02x", fm_mem[dut.out_base_addr + out_i]);
            end
            $fclose(layer_fh);
        end
        if (done) begin
            $display("DONE at cycle %0d", cycle);
            $display("DW writes seen: %0d", dw_wr_seen);
            $display("TARGET hits: %0d last_data=%0d (0x%02x)", target_hits, $signed(target_last_data), target_last_data);
            $display("TARGET q hits: %0d last_q=%0d (0x%02x)", target_q_hits, $signed(target_last_q), target_last_q);
            $display("TARGET acc hits: %0d last_acc=%0d (0x%08x)", target_acc_hits, $signed(target_last_acc), target_last_acc);
            fo = $fopen(FC_OUT_MEM, "w");
            fc_base = dut.in_base_addr;
            for (out_i = 0; out_i < FC_OUT_CH; out_i = out_i + 1) begin
                $fdisplay(fo, "%02x", fm_mem[fc_base + out_i]);
            end
            $fclose(fo);
            fo_logits = $fopen(FC_LOGITS_MEM, "w");
            for (out_i = 0; out_i < FC_OUT_CH; out_i = out_i + 1) begin
                $fdisplay(fo_logits, "%08x", fc_logits_mem[out_i]);
            end
            $fclose(fo_logits);
            $finish;
        end
        if (cycle == max_cycles) begin
            $display("TIMEOUT at cycle %0d", cycle);
            $finish;
        end
    end

    // Simple on-chip memory models (async read, sync write).
    localparam int FM_DEPTH = 4 * 1024 * 1024;
    localparam int DW_DEPTH = 4 * 1024 * 1024;
    logic [DATA_W-1:0] fm_mem [0:FM_DEPTH-1];
    logic [DATA_W-1:0] dw_mem [0:DW_DEPTH-1];
    integer mi;
    integer fo;
    integer fo_logits;
    integer out_i;
    integer fc_base;

    initial begin
        for (mi = 0; mi < FM_DEPTH; mi = mi + 1) begin
            fm_mem[mi] = '0;
        end
        for (mi = 0; mi < DW_DEPTH; mi = mi + 1) begin
            dw_mem[mi] = '0;
        end
        $display("Loading input mem: %s", INPUT_MEM);
        $readmemh(INPUT_MEM, fm_mem);
    end

    always_ff @(posedge clk) begin
        if (fm_wr_en0 && (fm_wr_addr0 < FM_DEPTH)) begin
            fm_mem[fm_wr_addr0] <= fm_wr_data0;
        end
        if (fm_wr_en1 && (fm_wr_addr1 < FM_DEPTH)) begin
            fm_mem[fm_wr_addr1] <= fm_wr_data1;
        end
        if (dw_buf_wr_en && (dw_buf_wr_addr < DW_DEPTH)) begin
            dw_mem[dw_buf_wr_addr] <= dw_buf_wr_data;
        end
    end

    always_comb begin
        if (fm_rd_addr < FM_DEPTH) begin
            fm_rd_data = fm_mem[fm_rd_addr];
        end else begin
            fm_rd_data = '0;
        end

        for (mi = 0; mi < PW_IC_PAR; mi = mi + 1) begin
            if (dw_buf_rd_en[mi] && (dw_buf_rd_addr[mi*ADDR_W +: ADDR_W] < DW_DEPTH)) begin
                dw_buf_rd_data[mi*DATA_W +: DATA_W] =
                    dw_mem[dw_buf_rd_addr[mi*ADDR_W +: ADDR_W]];
            end else begin
                dw_buf_rd_data[mi*DATA_W +: DATA_W] = '0;
            end
        end
    end
endmodule
