module mobilenet_v1_axi_lite #(
    parameter int AXI_ADDR_W = 8,
    parameter int AXI_DATA_W = 32,
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

    input  logic [AXI_ADDR_W-1:0] s_axi_awaddr,
    input  logic s_axi_awvalid,
    output logic s_axi_awready,

    input  logic [AXI_DATA_W-1:0] s_axi_wdata,
    input  logic [(AXI_DATA_W/8)-1:0] s_axi_wstrb,
    input  logic s_axi_wvalid,
    output logic s_axi_wready,

    output logic [1:0] s_axi_bresp,
    output logic s_axi_bvalid,
    input  logic s_axi_bready,

    input  logic [AXI_ADDR_W-1:0] s_axi_araddr,
    input  logic s_axi_arvalid,
    output logic s_axi_arready,

    output logic [AXI_DATA_W-1:0] s_axi_rdata,
    output logic [1:0] s_axi_rresp,
    output logic s_axi_rvalid,
    input  logic s_axi_rready,

    output logic irq
);
    logic aw_pending;
    logic [AXI_ADDR_W-1:0] awaddr_reg;

    logic w_pending;
    logic [AXI_DATA_W-1:0] wdata_reg;
    logic [(AXI_DATA_W/8)-1:0] wstrb_reg;

    logic ar_pending;
    logic [AXI_ADDR_W-1:0] araddr_reg;

    logic reg_wr_en;
    logic [7:0] reg_wr_addr;
    logic [31:0] reg_wr_data;
    logic reg_rd_en;
    logic [7:0] reg_rd_addr;
    logic [31:0] reg_rd_data;
    logic reg_rd_valid;

    assign s_axi_awready = !aw_pending && !s_axi_bvalid;
    assign s_axi_wready = !w_pending && !s_axi_bvalid;
    assign s_axi_bresp = 2'b00;

    assign s_axi_arready = !ar_pending && !s_axi_rvalid;
    assign s_axi_rresp = 2'b00;

    assign reg_wr_en = aw_pending && w_pending && !s_axi_bvalid;
    assign reg_wr_addr = awaddr_reg[7:0];
    assign reg_wr_data = wdata_reg[31:0];

    assign reg_rd_en = ar_pending && !s_axi_rvalid;
    assign reg_rd_addr = araddr_reg[7:0];

    mobilenet_v1_reg_shell #(
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
        .FM_DEPTH(FM_DEPTH),
        .DW_DEPTH(DW_DEPTH),
        .USE_XPM_RAM(USE_XPM_RAM),
        .FM_BASE0(FM_BASE0),
        .FM_BASE1(FM_BASE1),
        .DW_BUF_BASE(DW_BUF_BASE),
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
    ) u_regs (
        .clk(clk),
        .rst_n(rst_n),
        .reg_wr_en(reg_wr_en),
        .reg_wr_addr(reg_wr_addr),
        .reg_wr_data(reg_wr_data),
        .reg_rd_en(reg_rd_en),
        .reg_rd_addr(reg_rd_addr),
        .reg_rd_data(reg_rd_data),
        .reg_rd_valid(reg_rd_valid),
        .irq(irq)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            aw_pending <= 1'b0;
            awaddr_reg <= '0;
            w_pending <= 1'b0;
            wdata_reg <= '0;
            wstrb_reg <= '0;
            s_axi_bvalid <= 1'b0;

            ar_pending <= 1'b0;
            araddr_reg <= '0;
            s_axi_rvalid <= 1'b0;
            s_axi_rdata <= '0;
        end else begin
            if (s_axi_awready && s_axi_awvalid) begin
                aw_pending <= 1'b1;
                awaddr_reg <= s_axi_awaddr;
            end

            if (s_axi_wready && s_axi_wvalid) begin
                w_pending <= 1'b1;
                wdata_reg <= s_axi_wdata;
                wstrb_reg <= s_axi_wstrb;
            end

            if (reg_wr_en) begin
                aw_pending <= 1'b0;
                w_pending <= 1'b0;
                s_axi_bvalid <= 1'b1;
            end else if (s_axi_bvalid && s_axi_bready) begin
                s_axi_bvalid <= 1'b0;
            end

            if (s_axi_arready && s_axi_arvalid) begin
                ar_pending <= 1'b1;
                araddr_reg <= s_axi_araddr;
            end

            if (reg_rd_en && reg_rd_valid) begin
                ar_pending <= 1'b0;
                s_axi_rdata <= reg_rd_data;
                s_axi_rvalid <= 1'b1;
            end else if (s_axi_rvalid && s_axi_rready) begin
                s_axi_rvalid <= 1'b0;
            end
        end
    end
endmodule
