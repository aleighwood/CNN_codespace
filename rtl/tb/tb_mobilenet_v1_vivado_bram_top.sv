`timescale 1ns/1ps

module tb_mobilenet_v1_vivado_bram_top;
    localparam int AXI_ADDR_W = 8;
    localparam int AXI_DATA_W = 32;
    localparam int DATA_W = 8;
    localparam int ADDR_W = 32;
    localparam int DIM_W = 16;
    localparam int TILE_MASK_ADDR_W = 8;
    localparam int FM_DEPTH = 256;
    localparam int DW_DEPTH = 256;

    localparam logic [7:0] REG_CONTROL    = 8'h00;
    localparam logic [7:0] REG_STATUS     = 8'h04;
    localparam logic [7:0] REG_CFG_IN_H   = 8'h08;
    localparam logic [7:0] REG_CFG_IN_W   = 8'h0c;
    localparam logic [7:0] REG_MASK_ADDR  = 8'h20;
    localparam logic [7:0] REG_MASK_DATA  = 8'h24;
    localparam logic [7:0] REG_MASK_CMD   = 8'h28;

    logic s_axi_aclk;
    logic s_axi_aresetn;

    logic [AXI_ADDR_W-1:0] s_axi_awaddr;
    logic s_axi_awvalid;
    logic s_axi_awready;
    logic [AXI_DATA_W-1:0] s_axi_wdata;
    logic [(AXI_DATA_W/8)-1:0] s_axi_wstrb;
    logic s_axi_wvalid;
    logic s_axi_wready;
    logic [1:0] s_axi_bresp;
    logic s_axi_bvalid;
    logic s_axi_bready;
    logic [AXI_ADDR_W-1:0] s_axi_araddr;
    logic s_axi_arvalid;
    logic s_axi_arready;
    logic [AXI_DATA_W-1:0] s_axi_rdata;
    logic [1:0] s_axi_rresp;
    logic s_axi_rvalid;
    logic s_axi_rready;

    logic bram_rst_a;
    logic bram_clk_a;
    logic bram_en_a;
    logic [((DATA_W + 7) / 8)-1:0] bram_we_a;
    logic [ADDR_W-1:0] bram_addr_a;
    logic [DATA_W-1:0] bram_wrdata_a;
    logic [DATA_W-1:0] bram_rddata_a;

    logic irq;

    integer failures;
    logic [31:0] rd_value;

    mobilenet_v1_vivado_bram_top #(
        .AXI_ADDR_W(AXI_ADDR_W),
        .AXI_DATA_W(AXI_DATA_W),
        .DATA_W(DATA_W),
        .ADDR_W(ADDR_W),
        .DIM_W(DIM_W),
        .TILE_MASK_ADDR_W(TILE_MASK_ADDR_W),
        .TILE_MASK_DEPTH(256),
        .FM_DEPTH(FM_DEPTH),
        .DW_DEPTH(DW_DEPTH),
        .FM_BASE0('0),
        .FM_BASE1(32'd64),
        .DW_BUF_BASE(32'd128)
    ) dut (
        .s_axi_aclk(s_axi_aclk),
        .s_axi_aresetn(s_axi_aresetn),
        .s_axi_awaddr(s_axi_awaddr),
        .s_axi_awvalid(s_axi_awvalid),
        .s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),
        .s_axi_wstrb(s_axi_wstrb),
        .s_axi_wvalid(s_axi_wvalid),
        .s_axi_wready(s_axi_wready),
        .s_axi_bresp(s_axi_bresp),
        .s_axi_bvalid(s_axi_bvalid),
        .s_axi_bready(s_axi_bready),
        .s_axi_araddr(s_axi_araddr),
        .s_axi_arvalid(s_axi_arvalid),
        .s_axi_arready(s_axi_arready),
        .s_axi_rdata(s_axi_rdata),
        .s_axi_rresp(s_axi_rresp),
        .s_axi_rvalid(s_axi_rvalid),
        .s_axi_rready(s_axi_rready),
        .bram_rst_a(bram_rst_a),
        .bram_clk_a(bram_clk_a),
        .bram_en_a(bram_en_a),
        .bram_we_a(bram_we_a),
        .bram_addr_a(bram_addr_a),
        .bram_wrdata_a(bram_wrdata_a),
        .bram_rddata_a(bram_rddata_a),
        .irq(irq)
    );

    always #5 s_axi_aclk = ~s_axi_aclk;
    always_comb bram_clk_a = s_axi_aclk;
    always_comb bram_rst_a = ~s_axi_aresetn;

    task automatic check_equal(
        input [31:0] got,
        input [31:0] expected,
        input [255:0] label
    );
        begin
            if (got !== expected) begin
                failures = failures + 1;
                $display("FAIL %0s: got=0x%08x expected=0x%08x", label, got, expected);
            end else begin
                $display("PASS %0s: 0x%08x", label, got);
            end
        end
    endtask

    task automatic axi_write(
        input [7:0] addr,
        input [31:0] data
    );
        begin
            @(posedge s_axi_aclk);
            s_axi_awaddr = addr;
            s_axi_awvalid = 1'b1;
            s_axi_wdata = data;
            s_axi_wstrb = 4'hf;
            s_axi_wvalid = 1'b1;
            s_axi_bready = 1'b1;

            while (!(s_axi_awready && s_axi_wready)) begin
                @(posedge s_axi_aclk);
            end

            @(posedge s_axi_aclk);
            s_axi_awvalid = 1'b0;
            s_axi_wvalid = 1'b0;

            while (!s_axi_bvalid) begin
                @(posedge s_axi_aclk);
            end

            check_equal({30'd0, s_axi_bresp}, 32'd0, "axi_write_bresp");

            @(posedge s_axi_aclk);
            s_axi_bready = 1'b0;
        end
    endtask

    task automatic axi_read(
        input [7:0] addr,
        output [31:0] data
    );
        begin
            @(posedge s_axi_aclk);
            s_axi_araddr = addr;
            s_axi_arvalid = 1'b1;
            s_axi_rready = 1'b1;

            while (!s_axi_arready) begin
                @(posedge s_axi_aclk);
            end

            @(posedge s_axi_aclk);
            s_axi_arvalid = 1'b0;

            while (!s_axi_rvalid) begin
                @(posedge s_axi_aclk);
            end

            data = s_axi_rdata;
            check_equal({30'd0, s_axi_rresp}, 32'd0, "axi_read_rresp");

            @(posedge s_axi_aclk);
            s_axi_rready = 1'b0;
        end
    endtask

    task automatic bram_write_byte(
        input [ADDR_W-1:0] addr,
        input [DATA_W-1:0] data
    );
        begin
            @(posedge s_axi_aclk);
            bram_addr_a = addr;
            bram_wrdata_a = data;
            bram_we_a = '1;
            bram_en_a = 1'b1;
            @(posedge s_axi_aclk);
            bram_en_a = 1'b0;
            bram_we_a = '0;
        end
    endtask

    task automatic bram_read_byte(
        input [ADDR_W-1:0] addr
    );
        begin
            @(posedge s_axi_aclk);
            bram_addr_a = addr;
            bram_we_a = '0;
            bram_en_a = 1'b1;
            @(posedge s_axi_aclk);
            bram_en_a = 1'b0;
        end
    endtask

    initial begin
        s_axi_aclk = 1'b0;
        s_axi_aresetn = 1'b0;
        failures = 0;

        s_axi_awaddr = '0;
        s_axi_awvalid = 1'b0;
        s_axi_wdata = '0;
        s_axi_wstrb = '0;
        s_axi_wvalid = 1'b0;
        s_axi_bready = 1'b0;
        s_axi_araddr = '0;
        s_axi_arvalid = 1'b0;
        s_axi_rready = 1'b0;

        bram_en_a = 1'b0;
        bram_we_a = '0;
        bram_addr_a = '0;
        bram_wrdata_a = '0;

        repeat (5) @(posedge s_axi_aclk);
        s_axi_aresetn = 1'b1;
        repeat (2) @(posedge s_axi_aclk);

        axi_read(REG_STATUS, rd_value);
        check_equal(rd_value, 32'd0, "status_after_reset");

        bram_write_byte(32'd6, 8'h3c);
        bram_read_byte(32'd6);
        check_equal({24'd0, bram_rddata_a}, 32'h0000003c, "bram_direct_readback");

        axi_write(REG_CFG_IN_H, 32'd144);
        axi_read(REG_CFG_IN_H, rd_value);
        check_equal(rd_value, 32'd144, "cfg_in_h_readback");

        axi_write(REG_CFG_IN_W, 32'd176);
        axi_read(REG_CFG_IN_W, rd_value);
        check_equal(rd_value, 32'd176, "cfg_in_w_readback");

        axi_write(REG_CONTROL, 32'h0000_0002);
        axi_read(REG_CONTROL, rd_value);
        check_equal(rd_value, 32'h0000_0002, "control_readback");

        axi_write(REG_MASK_ADDR, 32'd4);
        axi_write(REG_MASK_DATA, 32'd1);
        axi_write(REG_MASK_CMD, 32'd1);
        check_equal({31'd0, dut.u_top.u_core.u_wrap.u_core.u_tile_mask_mem.mem[4][0]},
                    32'd1,
                    "mask_mem_write");

        check_equal({31'd0, irq}, 32'd0, "irq_idle_low");

        if (failures == 0) begin
            $display("TB PASSED");
        end else begin
            $display("TB FAILED failures=%0d", failures);
        end
        $finish;
    end
endmodule
