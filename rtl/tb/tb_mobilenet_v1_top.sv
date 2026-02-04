`timescale 1ns/1ps

module tb_mobilenet_v1_top;
    localparam int DATA_W = 8;
    localparam int ACC_W = 32;
    localparam int MUL_W = 16;
    localparam int BIAS_W = 32;
    localparam int SHIFT_W = 6;
    localparam int ADDR_W = 32;
    localparam int DIM_W = 16;
    localparam int OC_PAR = 16;
    localparam int PW_GROUP = 16;
    localparam int FC_OUT_CH = 1000;
    localparam int TILE_H = 16;
    localparam int TILE_W = 16;
    localparam string INPUT_MEM = "rtl/mem/input_rand.mem";
    localparam string FC_OUT_MEM = "rtl/mem/fc_out_hw.mem";
    localparam string FC_LOGITS_MEM = "rtl/mem/fc_logits_hw.mem";

    logic clk;
    logic rst_n;
    logic start;
    logic done;

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

    logic fm_wr_en;
    logic [ADDR_W-1:0] fm_wr_addr;
    logic [DATA_W-1:0] fm_wr_data;

    logic dw_buf_rd_en;
    logic [ADDR_W-1:0] dw_buf_rd_addr;
    logic [DATA_W-1:0] dw_buf_rd_data;

    logic dw_buf_wr_en;
    logic [ADDR_W-1:0] dw_buf_wr_addr;
    logic [DATA_W-1:0] dw_buf_wr_data;

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
        .FC_OUT_CH(FC_OUT_CH),
        .TILE_H(TILE_H),
        .TILE_W(TILE_W),
        .INIT_CONV1_W("rtl/mem/conv1_weight.mem"),
        .INIT_CONV1_BIAS_ACC("rtl/mem/conv1_bias_acc.mem"),
        .INIT_CONV1_MUL("rtl/mem/conv1_mul.mem"),
        .INIT_CONV1_BIAS_RQ("rtl/mem/conv1_bias_rq.mem"),
        .INIT_CONV1_SHIFT("rtl/mem/conv1_shift.mem"),
        .INIT_CONV1_RELU6("rtl/mem/conv1_relu6.mem"),
        .INIT_DW_W("rtl/mem/dw_weight.mem"),
        .INIT_DW_MUL("rtl/mem/dw_mul.mem"),
        .INIT_DW_BIAS("rtl/mem/dw_bias.mem"),
        .INIT_DW_SHIFT("rtl/mem/dw_shift.mem"),
        .INIT_DW_RELU6("rtl/mem/dw_relu6.mem"),
        .INIT_PW_W("rtl/mem/pw_weight.mem"),
        .INIT_PW_BIAS_ACC("rtl/mem/pw_bias_acc.mem"),
        .INIT_PW_MUL("rtl/mem/pw_mul.mem"),
        .INIT_PW_BIAS_RQ("rtl/mem/pw_bias_rq.mem"),
        .INIT_PW_SHIFT("rtl/mem/pw_shift.mem"),
        .INIT_PW_RELU6("rtl/mem/pw_relu6.mem"),
        .INIT_GAP_MUL("rtl/mem/gap_mul.mem"),
        .INIT_GAP_BIAS("rtl/mem/gap_bias.mem"),
        .INIT_GAP_SHIFT("rtl/mem/gap_shift.mem"),
        .INIT_FC_W("rtl/mem/fc_weight.mem"),
        .INIT_FC_MUL("rtl/mem/fc_mul.mem"),
        .INIT_FC_BIAS("rtl/mem/fc_bias.mem"),
        .INIT_FC_SHIFT("rtl/mem/fc_shift.mem")
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
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
        .fm_wr_en(fm_wr_en),
        .fm_wr_addr(fm_wr_addr),
        .fm_wr_data(fm_wr_data),
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

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
        start = 1'b0;
        img_h_arg = 16;
        img_w_arg = 16;
        max_cycles = 50000000;
        verbose = 1'b0;
        void'($value$plusargs("IMG_H=%d", img_h_arg));
        void'($value$plusargs("IMG_W=%d", img_w_arg));
        void'($value$plusargs("MAX_CYCLES=%d", max_cycles));
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
            mult64 = acc64 * $signed(dut.u_fc.fc_mul);
            scaled64 = mult64 + $signed(dut.u_fc.fc_bias);
            shifted64 = scaled64 >>> dut.u_fc.fc_shift;
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
        if (fm_wr_en && (fm_wr_addr < FM_DEPTH)) begin
            fm_mem[fm_wr_addr] <= fm_wr_data;
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

        if (dw_buf_rd_addr < DW_DEPTH) begin
            dw_buf_rd_data = dw_mem[dw_buf_rd_addr];
        end else begin
            dw_buf_rd_data = '0;
        end
    end
endmodule
