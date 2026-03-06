module mobilenet_v1_bram_wrapper #(
    parameter int DATA_W = 8,
    parameter int ACC_W = 32,
    parameter int MUL_W = 32,
    parameter int BIAS_W = 32,
    parameter int SHIFT_W = 6,
    parameter int ADDR_W = 32,
    parameter int DIM_W = 16,
    parameter int TILE_MASK_ADDR_W = 16,
    parameter int TILE_MASK_DEPTH = 4096,
    parameter int OC_PAR = 16,
    parameter int PW_GROUP = 32,
    parameter int PW_OC_PAR = 32,
    parameter int PW_IC_PAR = 16,
    parameter int FC_OUT_CH = 1000,
    parameter int TILE_H = 32,
    parameter int TILE_W = 32,
    parameter int INPUT_ZP = -1,
    parameter int ACT_ZP = -128,
    parameter int FM_DEPTH = 4 * 1024 * 1024,
    parameter int DW_DEPTH = 4 * 1024 * 1024,
    parameter bit USE_XPM_RAM = 1'b0,
    parameter logic [ADDR_W-1:0] FM_BASE0 = '0,
    parameter logic [ADDR_W-1:0] FM_BASE1 = 32'h0010_0000,
    parameter logic [ADDR_W-1:0] DW_BUF_BASE = 32'h0020_0000,
    parameter string INIT_CONV1_W = "",
    parameter string INIT_CONV1_BIAS_ACC = "",
    parameter string INIT_CONV1_MUL = "",
    parameter string INIT_CONV1_BIAS_RQ = "",
    parameter string INIT_CONV1_SHIFT = "",
    parameter string INIT_CONV1_RELU6 = "",
    parameter string INIT_CONV1_RELU6_MIN = "",
    parameter string INIT_DW_W = "",
    parameter string INIT_DW_MUL = "",
    parameter string INIT_DW_BIAS = "",
    parameter string INIT_DW_SHIFT = "",
    parameter string INIT_DW_RELU6 = "",
    parameter string INIT_DW_RELU6_MIN = "",
    parameter string INIT_PW_W = "",
    parameter string INIT_PW_BIAS_ACC = "",
    parameter string INIT_PW_MUL = "",
    parameter string INIT_PW_BIAS_RQ = "",
    parameter string INIT_PW_SHIFT = "",
    parameter string INIT_PW_RELU6 = "",
    parameter string INIT_PW_RELU6_MIN = "",
    parameter string INIT_GAP_MUL = "",
    parameter string INIT_GAP_BIAS = "",
    parameter string INIT_GAP_SHIFT = "",
    parameter string INIT_FC_W = "",
    parameter string INIT_FC_MUL = "",
    parameter string INIT_FC_BIAS = "",
    parameter string INIT_FC_SHIFT = "",
    parameter string INIT_FC_ZP = "",
    parameter string INIT_TILE_MASK = ""
) (
    input  logic clk,
    input  logic rst_n,

    input  logic start,
    output logic done,
    output logic busy,

    input  logic tile_skip_en,

    input  logic [DIM_W-1:0] cfg_in_img_h,
    input  logic [DIM_W-1:0] cfg_in_img_w,

    input  logic host_fm_en,
    input  logic [((DATA_W + 7) / 8)-1:0] host_fm_we,
    input  logic [ADDR_W-1:0] host_fm_addr,
    input  logic [DATA_W-1:0] host_fm_din,
    output logic [DATA_W-1:0] host_fm_dout,

    input  logic tile_mask_wr_en,
    input  logic [TILE_MASK_ADDR_W-1:0] tile_mask_wr_addr,
    input  logic tile_mask_wr_data,

    input  logic param_wr_en,
    input  logic [4:0] param_wr_sel,
    input  logic [19:0] param_wr_addr,
    input  logic [31:0] param_wr_data
);
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

    localparam int FM0_DEPTH = (FM_BASE1 > FM_BASE0) ? (FM_BASE1 - FM_BASE0) :
                               ((FM_DEPTH <= 1) ? 1 : FM_DEPTH);
    localparam int FM1_DEPTH = (DW_BUF_BASE > FM_BASE1) ? (DW_BUF_BASE - FM_BASE1) :
                               ((FM_DEPTH <= 1) ? 1 : FM_DEPTH);
    localparam int FM0_ADDR_W = (FM0_DEPTH <= 1) ? 1 : $clog2(FM0_DEPTH);
    localparam int FM1_ADDR_W = (FM1_DEPTH <= 1) ? 1 : $clog2(FM1_DEPTH);
    localparam int DW_LOCAL_DEPTH = (DW_DEPTH <= 1) ? 1 : DW_DEPTH;
    localparam int DW_ADDR_W = (DW_LOCAL_DEPTH <= 1) ? 1 : $clog2(DW_LOCAL_DEPTH);

    localparam logic [ADDR_W-1:0] FM0_END = FM_BASE0 + ADDR_W'(FM0_DEPTH);
    localparam logic [ADDR_W-1:0] FM1_END = FM_BASE1 + ADDR_W'(FM1_DEPTH);
    localparam logic [ADDR_W-1:0] DW_END = DW_BUF_BASE + ADDR_W'(DW_LOCAL_DEPTH);

    logic fm0_a_en;
    logic fm0_a_we;
    logic [FM0_ADDR_W-1:0] fm0_a_addr;
    logic [DATA_W-1:0] fm0_a_din;
    logic [DATA_W-1:0] fm0_a_dout;
    logic fm0_b_en;
    logic fm0_b_we;
    logic [FM0_ADDR_W-1:0] fm0_b_addr;
    logic [DATA_W-1:0] fm0_b_din;
    logic [DATA_W-1:0] fm0_b_dout;

    logic fm1_a_en;
    logic fm1_a_we;
    logic [FM1_ADDR_W-1:0] fm1_a_addr;
    logic [DATA_W-1:0] fm1_a_din;
    logic [DATA_W-1:0] fm1_a_dout;
    logic fm1_b_en;
    logic fm1_b_we;
    logic [FM1_ADDR_W-1:0] fm1_b_addr;
    logic [DATA_W-1:0] fm1_b_din;
    logic [DATA_W-1:0] fm1_b_dout;

    logic [PW_IC_PAR-1:0] dw_bank_b_en;
    logic [PW_IC_PAR*DW_ADDR_W-1:0] dw_bank_b_addr;
    logic [PW_IC_PAR*DATA_W-1:0] dw_bank_b_dout;

    logic dw_wr_hit;
    logic [ADDR_W-1:0] dw_wr_addr_local_full;
    logic [DW_ADDR_W-1:0] dw_wr_addr_local;

    logic core_rd_fm0;
    logic core_rd_fm1;
    logic wr0_fm0;
    logic wr0_fm1;
    logic wr1_fm0;
    logic wr1_fm1;
    logic host_wr_fm0;
    logic host_wr_fm1;
    logic host_rd_fm0;
    logic host_rd_fm1;
    logic host_fm_write;

    logic [ADDR_W-1:0] fm_rd_local0_full;
    logic [ADDR_W-1:0] fm_rd_local1_full;
    logic [ADDR_W-1:0] fm_wr0_local0_full;
    logic [ADDR_W-1:0] fm_wr0_local1_full;
    logic [ADDR_W-1:0] fm_wr1_local0_full;
    logic [ADDR_W-1:0] fm_wr1_local1_full;
    logic [ADDR_W-1:0] host_wr_local0_full;
    logic [ADDR_W-1:0] host_wr_local1_full;
    logic [ADDR_W-1:0] host_rd_local0_full;
    logic [ADDR_W-1:0] host_rd_local1_full;

    mobilenet_v1_top #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .MUL_W(MUL_W),
        .BIAS_W(BIAS_W),
        .SHIFT_W(SHIFT_W),
        .ADDR_W(ADDR_W),
        .DIM_W(DIM_W),
        .TILE_MASK_ADDR_W(TILE_MASK_ADDR_W),
        .TILE_MASK_DEPTH(TILE_MASK_DEPTH),
        .OC_PAR(OC_PAR),
        .PW_GROUP(PW_GROUP),
        .PW_OC_PAR(PW_OC_PAR),
        .PW_IC_PAR(PW_IC_PAR),
        .FC_OUT_CH(FC_OUT_CH),
        .TILE_H(TILE_H),
        .TILE_W(TILE_W),
        .INPUT_ZP(INPUT_ZP),
        .ACT_ZP(ACT_ZP),
        .INIT_CONV1_W(INIT_CONV1_W),
        .INIT_CONV1_BIAS_ACC(INIT_CONV1_BIAS_ACC),
        .INIT_CONV1_MUL(INIT_CONV1_MUL),
        .INIT_CONV1_BIAS_RQ(INIT_CONV1_BIAS_RQ),
        .INIT_CONV1_SHIFT(INIT_CONV1_SHIFT),
        .INIT_CONV1_RELU6(INIT_CONV1_RELU6),
        .INIT_CONV1_RELU6_MIN(INIT_CONV1_RELU6_MIN),
        .INIT_DW_W(INIT_DW_W),
        .INIT_DW_MUL(INIT_DW_MUL),
        .INIT_DW_BIAS(INIT_DW_BIAS),
        .INIT_DW_SHIFT(INIT_DW_SHIFT),
        .INIT_DW_RELU6(INIT_DW_RELU6),
        .INIT_DW_RELU6_MIN(INIT_DW_RELU6_MIN),
        .INIT_PW_W(INIT_PW_W),
        .INIT_PW_BIAS_ACC(INIT_PW_BIAS_ACC),
        .INIT_PW_MUL(INIT_PW_MUL),
        .INIT_PW_BIAS_RQ(INIT_PW_BIAS_RQ),
        .INIT_PW_SHIFT(INIT_PW_SHIFT),
        .INIT_PW_RELU6(INIT_PW_RELU6),
        .INIT_PW_RELU6_MIN(INIT_PW_RELU6_MIN),
        .INIT_GAP_MUL(INIT_GAP_MUL),
        .INIT_GAP_BIAS(INIT_GAP_BIAS),
        .INIT_GAP_SHIFT(INIT_GAP_SHIFT),
        .INIT_FC_W(INIT_FC_W),
        .INIT_FC_MUL(INIT_FC_MUL),
        .INIT_FC_BIAS(INIT_FC_BIAS),
        .INIT_FC_SHIFT(INIT_FC_SHIFT),
        .INIT_FC_ZP(INIT_FC_ZP),
        .INIT_TILE_MASK(INIT_TILE_MASK)
    ) u_core (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .done(done),
        .busy(busy),
        .tile_skip_en(tile_skip_en),
        .tile_mask_wr_en(tile_mask_wr_en),
        .tile_mask_wr_addr(tile_mask_wr_addr),
        .tile_mask_wr_data(tile_mask_wr_data),
        .cfg_in_img_h(cfg_in_img_h),
        .cfg_in_img_w(cfg_in_img_w),
        .cfg_fm_base0(FM_BASE0),
        .cfg_fm_base1(FM_BASE1),
        .cfg_dw_buf_base(DW_BUF_BASE),
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

    assign core_rd_fm0 = fm_rd_en && (fm_rd_addr >= FM_BASE0) && (fm_rd_addr < FM0_END);
    assign core_rd_fm1 = fm_rd_en && (fm_rd_addr >= FM_BASE1) && (fm_rd_addr < FM1_END);

    assign wr0_fm0 = fm_wr_en0 && (fm_wr_addr0 >= FM_BASE0) && (fm_wr_addr0 < FM0_END);
    assign wr0_fm1 = fm_wr_en0 && (fm_wr_addr0 >= FM_BASE1) && (fm_wr_addr0 < FM1_END);
    assign wr1_fm0 = fm_wr_en1 && (fm_wr_addr1 >= FM_BASE0) && (fm_wr_addr1 < FM0_END);
    assign wr1_fm1 = fm_wr_en1 && (fm_wr_addr1 >= FM_BASE1) && (fm_wr_addr1 < FM1_END);

    assign host_fm_write = host_fm_en && (|host_fm_we);
    assign host_wr_fm0 = host_fm_write && !busy && (host_fm_addr >= FM_BASE0) && (host_fm_addr < FM0_END);
    assign host_wr_fm1 = host_fm_write && !busy && (host_fm_addr >= FM_BASE1) && (host_fm_addr < FM1_END);
    assign host_rd_fm0 = host_fm_en && !host_fm_write && !busy && (host_fm_addr >= FM_BASE0) && (host_fm_addr < FM0_END);
    assign host_rd_fm1 = host_fm_en && !host_fm_write && !busy && (host_fm_addr >= FM_BASE1) && (host_fm_addr < FM1_END);

    assign fm_rd_local0_full = fm_rd_addr - FM_BASE0;
    assign fm_rd_local1_full = fm_rd_addr - FM_BASE1;
    assign fm_wr0_local0_full = fm_wr_addr0 - FM_BASE0;
    assign fm_wr0_local1_full = fm_wr_addr0 - FM_BASE1;
    assign fm_wr1_local0_full = fm_wr_addr1 - FM_BASE0;
    assign fm_wr1_local1_full = fm_wr_addr1 - FM_BASE1;
    assign host_wr_local0_full = host_fm_addr - FM_BASE0;
    assign host_wr_local1_full = host_fm_addr - FM_BASE1;
    assign host_rd_local0_full = host_fm_addr - FM_BASE0;
    assign host_rd_local1_full = host_fm_addr - FM_BASE1;

    assign dw_wr_hit = dw_buf_wr_en && (dw_buf_wr_addr >= DW_BUF_BASE) && (dw_buf_wr_addr < DW_END);
    assign dw_wr_addr_local_full = dw_buf_wr_addr - DW_BUF_BASE;
    assign dw_wr_addr_local = dw_wr_addr_local_full[DW_ADDR_W-1:0];

    dual_port_ram_async #(
        .DATA_W(DATA_W),
        .DEPTH(FM0_DEPTH),
        .MEM_STYLE("ultra"),
        .USE_XPM(USE_XPM_RAM)
    ) u_fm_bank0 (
        .clk(clk),
        .a_en(fm0_a_en),
        .a_we(fm0_a_we),
        .a_addr(fm0_a_addr),
        .a_din(fm0_a_din),
        .a_dout(fm0_a_dout),
        .b_en(fm0_b_en),
        .b_we(fm0_b_we),
        .b_addr(fm0_b_addr),
        .b_din(fm0_b_din),
        .b_dout(fm0_b_dout)
    );

    dual_port_ram_async #(
        .DATA_W(DATA_W),
        .DEPTH(FM1_DEPTH),
        .MEM_STYLE("ultra"),
        .USE_XPM(USE_XPM_RAM)
    ) u_fm_bank1 (
        .clk(clk),
        .a_en(fm1_a_en),
        .a_we(fm1_a_we),
        .a_addr(fm1_a_addr),
        .a_din(fm1_a_din),
        .a_dout(fm1_a_dout),
        .b_en(fm1_b_en),
        .b_we(fm1_b_we),
        .b_addr(fm1_b_addr),
        .b_din(fm1_b_din),
        .b_dout(fm1_b_dout)
    );

    genvar dw_lane;
    generate
        for (dw_lane = 0; dw_lane < PW_IC_PAR; dw_lane = dw_lane + 1) begin : gen_dw_replica
            logic [ADDR_W-1:0] dw_rd_local_full;

            assign dw_rd_local_full = dw_buf_rd_addr[dw_lane*ADDR_W +: ADDR_W] - DW_BUF_BASE;
            assign dw_bank_b_en[dw_lane] =
                dw_buf_rd_en[dw_lane] &&
                (dw_buf_rd_addr[dw_lane*ADDR_W +: ADDR_W] >= DW_BUF_BASE) &&
                (dw_buf_rd_addr[dw_lane*ADDR_W +: ADDR_W] < DW_END);
            assign dw_bank_b_addr[dw_lane*DW_ADDR_W +: DW_ADDR_W] = dw_rd_local_full[DW_ADDR_W-1:0];

            dual_port_ram_async #(
                .DATA_W(DATA_W),
                .DEPTH(DW_LOCAL_DEPTH),
                .MEM_STYLE("block"),
                .USE_XPM(USE_XPM_RAM)
            ) u_dw_bank (
                .clk(clk),
                .a_en(dw_wr_hit),
                .a_we(dw_wr_hit),
                .a_addr(dw_wr_addr_local),
                .a_din(dw_buf_wr_data),
                .a_dout(),
                .b_en(dw_bank_b_en[dw_lane]),
                .b_we(1'b0),
                .b_addr(dw_bank_b_addr[dw_lane*DW_ADDR_W +: DW_ADDR_W]),
                .b_din('0),
                .b_dout(dw_bank_b_dout[dw_lane*DATA_W +: DATA_W])
            );
        end
    endgenerate

    always_comb begin
        int lane;

        fm0_a_en = 1'b0;
        fm0_a_we = 1'b0;
        fm0_a_addr = '0;
        fm0_a_din = '0;
        fm0_b_en = 1'b0;
        fm0_b_we = 1'b0;
        fm0_b_addr = '0;
        fm0_b_din = '0;

        fm1_a_en = 1'b0;
        fm1_a_we = 1'b0;
        fm1_a_addr = '0;
        fm1_a_din = '0;
        fm1_b_en = 1'b0;
        fm1_b_we = 1'b0;
        fm1_b_addr = '0;
        fm1_b_din = '0;

        if (busy) begin
            if (core_rd_fm0) begin
                fm0_a_en = 1'b1;
                fm0_a_addr = fm_rd_local0_full[FM0_ADDR_W-1:0];
            end
            if (core_rd_fm1) begin
                fm1_a_en = 1'b1;
                fm1_a_addr = fm_rd_local1_full[FM1_ADDR_W-1:0];
            end

            if (wr0_fm0) begin
                if (!fm0_a_en) begin
                    fm0_a_en = 1'b1;
                    fm0_a_we = 1'b1;
                    fm0_a_addr = fm_wr0_local0_full[FM0_ADDR_W-1:0];
                    fm0_a_din = fm_wr_data0;
                end else if (!fm0_b_en) begin
                    fm0_b_en = 1'b1;
                    fm0_b_we = 1'b1;
                    fm0_b_addr = fm_wr0_local0_full[FM0_ADDR_W-1:0];
                    fm0_b_din = fm_wr_data0;
                end
            end

            if (wr1_fm0) begin
                if (!fm0_a_en) begin
                    fm0_a_en = 1'b1;
                    fm0_a_we = 1'b1;
                    fm0_a_addr = fm_wr1_local0_full[FM0_ADDR_W-1:0];
                    fm0_a_din = fm_wr_data1;
                end else if (!fm0_b_en) begin
                    fm0_b_en = 1'b1;
                    fm0_b_we = 1'b1;
                    fm0_b_addr = fm_wr1_local0_full[FM0_ADDR_W-1:0];
                    fm0_b_din = fm_wr_data1;
                end
            end

            if (wr0_fm1) begin
                if (!fm1_a_en) begin
                    fm1_a_en = 1'b1;
                    fm1_a_we = 1'b1;
                    fm1_a_addr = fm_wr0_local1_full[FM1_ADDR_W-1:0];
                    fm1_a_din = fm_wr_data0;
                end else if (!fm1_b_en) begin
                    fm1_b_en = 1'b1;
                    fm1_b_we = 1'b1;
                    fm1_b_addr = fm_wr0_local1_full[FM1_ADDR_W-1:0];
                    fm1_b_din = fm_wr_data0;
                end
            end

            if (wr1_fm1) begin
                if (!fm1_a_en) begin
                    fm1_a_en = 1'b1;
                    fm1_a_we = 1'b1;
                    fm1_a_addr = fm_wr1_local1_full[FM1_ADDR_W-1:0];
                    fm1_a_din = fm_wr_data1;
                end else if (!fm1_b_en) begin
                    fm1_b_en = 1'b1;
                    fm1_b_we = 1'b1;
                    fm1_b_addr = fm_wr1_local1_full[FM1_ADDR_W-1:0];
                    fm1_b_din = fm_wr_data1;
                end
            end
        end else begin
            if (host_wr_fm0) begin
                fm0_a_en = 1'b1;
                fm0_a_we = 1'b1;
                fm0_a_addr = host_wr_local0_full[FM0_ADDR_W-1:0];
                fm0_a_din = host_fm_din;
            end
            if (host_rd_fm0) begin
                fm0_b_en = 1'b1;
                fm0_b_addr = host_rd_local0_full[FM0_ADDR_W-1:0];
            end

            if (host_wr_fm1) begin
                fm1_a_en = 1'b1;
                fm1_a_we = 1'b1;
                fm1_a_addr = host_wr_local1_full[FM1_ADDR_W-1:0];
                fm1_a_din = host_fm_din;
            end
            if (host_rd_fm1) begin
                fm1_b_en = 1'b1;
                fm1_b_addr = host_rd_local1_full[FM1_ADDR_W-1:0];
            end
        end

        fm_rd_data = core_rd_fm0 ? fm0_a_dout :
                     core_rd_fm1 ? fm1_a_dout : '0;

        host_fm_dout = host_rd_fm0 ? fm0_b_dout :
                       host_rd_fm1 ? fm1_b_dout : '0;

        for (lane = 0; lane < PW_IC_PAR; lane = lane + 1) begin
            dw_buf_rd_data[lane*DATA_W +: DATA_W] = dw_bank_b_dout[lane*DATA_W +: DATA_W];
        end
    end
endmodule
