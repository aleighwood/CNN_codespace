module mobilenet_v1_reg_shell #(
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

    input  logic reg_wr_en,
    input  logic [7:0] reg_wr_addr,
    input  logic [31:0] reg_wr_data,
    input  logic reg_rd_en,
    input  logic [7:0] reg_rd_addr,
    output logic [31:0] reg_rd_data,
    output logic reg_rd_valid,

    output logic irq
);
    localparam logic [7:0] REG_CONTROL      = 8'h00;
    localparam logic [7:0] REG_STATUS       = 8'h04;
    localparam logic [7:0] REG_CFG_IN_H     = 8'h08;
    localparam logic [7:0] REG_CFG_IN_W     = 8'h0c;
    localparam logic [7:0] REG_FM_ADDR      = 8'h10;
    localparam logic [7:0] REG_FM_WDATA     = 8'h14;
    localparam logic [7:0] REG_FM_RDATA     = 8'h18;
    localparam logic [7:0] REG_FM_CMD       = 8'h1c;
    localparam logic [7:0] REG_MASK_ADDR    = 8'h20;
    localparam logic [7:0] REG_MASK_DATA    = 8'h24;
    localparam logic [7:0] REG_MASK_CMD     = 8'h28;
    localparam logic [7:0] REG_PARAM_SEL    = 8'h30;
    localparam logic [7:0] REG_PARAM_ADDR   = 8'h34;
    localparam logic [7:0] REG_PARAM_DATA   = 8'h38;
    localparam logic [7:0] REG_PARAM_CMD    = 8'h3c;

    logic start_pulse;
    logic tile_skip_en_reg;
    logic irq_enable_reg;
    logic done_sticky;

    logic done;
    logic busy;

    logic [DIM_W-1:0] cfg_in_img_h_reg;
    logic [DIM_W-1:0] cfg_in_img_w_reg;

    logic host_fm_wr_pulse;
    logic host_fm_rd_en;
    logic [ADDR_W-1:0] host_fm_addr_reg;
    logic [DATA_W-1:0] host_fm_wdata_reg;
    logic [DATA_W-1:0] host_fm_dout;

    logic tile_mask_wr_en;
    logic [TILE_MASK_ADDR_W-1:0] tile_mask_addr_reg;
    logic tile_mask_data_reg;

    logic param_wr_en;
    logic [4:0] param_wr_sel_reg;
    logic [19:0] param_wr_addr_reg;
    logic [31:0] param_wr_data_reg;

    mobilenet_v1_bram_wrapper #(
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
    ) u_wrap (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_pulse),
        .done(done),
        .busy(busy),
        .tile_skip_en(tile_skip_en_reg),
        .cfg_in_img_h(cfg_in_img_h_reg),
        .cfg_in_img_w(cfg_in_img_w_reg),
        .host_fm_en(host_fm_wr_pulse || host_fm_rd_en),
        .host_fm_we(host_fm_wr_pulse ? '1 : '0),
        .host_fm_addr(host_fm_addr_reg),
        .host_fm_din(host_fm_wdata_reg),
        .host_fm_dout(host_fm_dout),
        .tile_mask_wr_en(tile_mask_wr_en),
        .tile_mask_wr_addr(tile_mask_addr_reg),
        .tile_mask_wr_data(tile_mask_data_reg),
        .param_wr_en(param_wr_en),
        .param_wr_sel(param_wr_sel_reg),
        .param_wr_addr(param_wr_addr_reg),
        .param_wr_data(param_wr_data_reg)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            start_pulse <= 1'b0;
            host_fm_wr_pulse <= 1'b0;
            tile_mask_wr_en <= 1'b0;
            param_wr_en <= 1'b0;
            tile_skip_en_reg <= 1'b0;
            irq_enable_reg <= 1'b0;
            done_sticky <= 1'b0;
            cfg_in_img_h_reg <= 16'd224;
            cfg_in_img_w_reg <= 16'd224;
            host_fm_addr_reg <= '0;
            host_fm_wdata_reg <= '0;
            tile_mask_addr_reg <= '0;
            tile_mask_data_reg <= 1'b0;
            param_wr_sel_reg <= '0;
            param_wr_addr_reg <= '0;
            param_wr_data_reg <= '0;
        end else begin
            start_pulse <= 1'b0;
            host_fm_wr_pulse <= 1'b0;
            tile_mask_wr_en <= 1'b0;
            param_wr_en <= 1'b0;

            if (done) begin
                done_sticky <= 1'b1;
            end

            if (reg_wr_en) begin
                case (reg_wr_addr)
                    REG_CONTROL: begin
                        if (reg_wr_data[0]) begin
                            start_pulse <= 1'b1;
                        end
                        tile_skip_en_reg <= reg_wr_data[1];
                        irq_enable_reg <= reg_wr_data[2];
                        if (reg_wr_data[4]) begin
                            done_sticky <= 1'b0;
                        end
                    end
                    REG_CFG_IN_H: begin
                        cfg_in_img_h_reg <= reg_wr_data[DIM_W-1:0];
                    end
                    REG_CFG_IN_W: begin
                        cfg_in_img_w_reg <= reg_wr_data[DIM_W-1:0];
                    end
                    REG_FM_ADDR: begin
                        host_fm_addr_reg <= reg_wr_data[ADDR_W-1:0];
                    end
                    REG_FM_WDATA: begin
                        host_fm_wdata_reg <= reg_wr_data[DATA_W-1:0];
                    end
                    REG_FM_CMD: begin
                        if (reg_wr_data[0]) begin
                            host_fm_wr_pulse <= 1'b1;
                        end
                    end
                    REG_MASK_ADDR: begin
                        tile_mask_addr_reg <= reg_wr_data[TILE_MASK_ADDR_W-1:0];
                    end
                    REG_MASK_DATA: begin
                        tile_mask_data_reg <= reg_wr_data[0];
                    end
                    REG_MASK_CMD: begin
                        if (reg_wr_data[0]) begin
                            tile_mask_wr_en <= 1'b1;
                        end
                    end
                    REG_PARAM_SEL: begin
                        param_wr_sel_reg <= reg_wr_data[4:0];
                    end
                    REG_PARAM_ADDR: begin
                        param_wr_addr_reg <= reg_wr_data[19:0];
                    end
                    REG_PARAM_DATA: begin
                        param_wr_data_reg <= reg_wr_data;
                    end
                    REG_PARAM_CMD: begin
                        if (reg_wr_data[0]) begin
                            param_wr_en <= 1'b1;
                        end
                    end
                    default: begin
                    end
                endcase
            end
        end
    end

    always_comb begin
        reg_rd_data = 32'd0;

        case (reg_rd_addr)
            REG_CONTROL: begin
                reg_rd_data[1] = tile_skip_en_reg;
                reg_rd_data[2] = irq_enable_reg;
            end
            REG_STATUS: begin
                reg_rd_data[0] = busy;
                reg_rd_data[1] = done_sticky;
                reg_rd_data[2] = done;
                reg_rd_data[3] = irq;
            end
            REG_CFG_IN_H: begin
                reg_rd_data[DIM_W-1:0] = cfg_in_img_h_reg;
            end
            REG_CFG_IN_W: begin
                reg_rd_data[DIM_W-1:0] = cfg_in_img_w_reg;
            end
            REG_FM_ADDR: begin
                reg_rd_data[ADDR_W-1:0] = host_fm_addr_reg;
            end
            REG_FM_WDATA: begin
                reg_rd_data[DATA_W-1:0] = host_fm_wdata_reg;
            end
            REG_FM_RDATA: begin
                reg_rd_data[DATA_W-1:0] = host_fm_dout;
            end
            REG_MASK_ADDR: begin
                reg_rd_data[TILE_MASK_ADDR_W-1:0] = tile_mask_addr_reg;
            end
            REG_MASK_DATA: begin
                reg_rd_data[0] = tile_mask_data_reg;
            end
            REG_PARAM_SEL: begin
                reg_rd_data[4:0] = param_wr_sel_reg;
            end
            REG_PARAM_ADDR: begin
                reg_rd_data[19:0] = param_wr_addr_reg;
            end
            REG_PARAM_DATA: begin
                reg_rd_data = param_wr_data_reg;
            end
            default: begin
                reg_rd_data = 32'd0;
            end
        endcase
    end

    assign host_fm_rd_en = reg_rd_en && (reg_rd_addr == REG_FM_RDATA);
    assign reg_rd_valid = reg_rd_en;
    assign irq = irq_enable_reg && done_sticky;
endmodule
