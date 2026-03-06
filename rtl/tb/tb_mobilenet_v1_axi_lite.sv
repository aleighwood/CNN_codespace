`timescale 1ns/1ps

module tb_mobilenet_v1_axi_lite #(
    parameter int USE_XPM_RAM = 0
);
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
    localparam logic [7:0] REG_FM_ADDR    = 8'h10;
    localparam logic [7:0] REG_FM_WDATA   = 8'h14;
    localparam logic [7:0] REG_FM_RDATA   = 8'h18;
    localparam logic [7:0] REG_FM_CMD     = 8'h1c;
    localparam logic [7:0] REG_MASK_ADDR  = 8'h20;
    localparam logic [7:0] REG_MASK_DATA  = 8'h24;
    localparam logic [7:0] REG_MASK_CMD   = 8'h28;
    localparam logic [7:0] REG_PARAM_SEL  = 8'h30;
    localparam logic [7:0] REG_PARAM_ADDR = 8'h34;
    localparam logic [7:0] REG_PARAM_DATA = 8'h38;

    logic clk;
    logic rst_n;

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

    logic irq;

    integer failures;

    mobilenet_v1_axi_lite #(
        .AXI_ADDR_W(AXI_ADDR_W),
        .AXI_DATA_W(AXI_DATA_W),
        .DATA_W(DATA_W),
        .ADDR_W(ADDR_W),
        .DIM_W(DIM_W),
        .TILE_MASK_ADDR_W(TILE_MASK_ADDR_W),
        .TILE_MASK_DEPTH(256),
        .FM_DEPTH(FM_DEPTH),
        .DW_DEPTH(DW_DEPTH),
        .USE_XPM_RAM(USE_XPM_RAM),
        .FM_BASE0('0),
        .FM_BASE1(32'd64),
        .DW_BUF_BASE(32'd128)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
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
        .irq(irq)
    );

    always #5 clk = ~clk;

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

    task automatic axi_write_simul(
        input [7:0] addr,
        input [31:0] data
    );
        begin
            @(posedge clk);
            s_axi_awaddr = addr;
            s_axi_awvalid = 1'b1;
            s_axi_wdata = data;
            s_axi_wstrb = 4'hf;
            s_axi_wvalid = 1'b1;
            s_axi_bready = 1'b1;

            while (!(s_axi_awready && s_axi_wready)) begin
                @(posedge clk);
            end

            @(posedge clk);
            s_axi_awvalid = 1'b0;
            s_axi_wvalid = 1'b0;

            while (!s_axi_bvalid) begin
                @(posedge clk);
            end

            check_equal({30'd0, s_axi_bresp}, 32'd0, "axi_write_bresp");

            @(posedge clk);
            s_axi_bready = 1'b0;
        end
    endtask

    task automatic axi_write_aw_then_w(
        input [7:0] addr,
        input [31:0] data
    );
        begin
            @(posedge clk);
            s_axi_awaddr = addr;
            s_axi_awvalid = 1'b1;

            while (!s_axi_awready) begin
                @(posedge clk);
            end

            @(posedge clk);
            s_axi_awvalid = 1'b0;
            s_axi_wdata = data;
            s_axi_wstrb = 4'hf;
            s_axi_wvalid = 1'b1;
            s_axi_bready = 1'b1;

            while (!s_axi_wready) begin
                @(posedge clk);
            end

            @(posedge clk);
            s_axi_wvalid = 1'b0;

            while (!s_axi_bvalid) begin
                @(posedge clk);
            end

            check_equal({30'd0, s_axi_bresp}, 32'd0, "axi_write_aw_then_w_bresp");

            @(posedge clk);
            s_axi_bready = 1'b0;
        end
    endtask

    task automatic axi_write_w_then_aw(
        input [7:0] addr,
        input [31:0] data
    );
        begin
            @(posedge clk);
            s_axi_wdata = data;
            s_axi_wstrb = 4'hf;
            s_axi_wvalid = 1'b1;

            while (!s_axi_wready) begin
                @(posedge clk);
            end

            @(posedge clk);
            s_axi_wvalid = 1'b0;
            s_axi_awaddr = addr;
            s_axi_awvalid = 1'b1;
            s_axi_bready = 1'b1;

            while (!s_axi_awready) begin
                @(posedge clk);
            end

            @(posedge clk);
            s_axi_awvalid = 1'b0;

            while (!s_axi_bvalid) begin
                @(posedge clk);
            end

            check_equal({30'd0, s_axi_bresp}, 32'd0, "axi_write_w_then_aw_bresp");

            @(posedge clk);
            s_axi_bready = 1'b0;
        end
    endtask

    task automatic axi_read(
        input [7:0] addr,
        output [31:0] data
    );
        begin
            @(posedge clk);
            s_axi_araddr = addr;
            s_axi_arvalid = 1'b1;
            s_axi_rready = 1'b1;

            while (!s_axi_arready) begin
                @(posedge clk);
            end

            @(posedge clk);
            s_axi_arvalid = 1'b0;

            while (!s_axi_rvalid) begin
                @(posedge clk);
            end

            data = s_axi_rdata;
            check_equal({30'd0, s_axi_rresp}, 32'd0, "axi_read_rresp");

            @(posedge clk);
            s_axi_rready = 1'b0;
        end
    endtask

    logic [31:0] rd_value;

    initial begin
        clk = 1'b0;
        rst_n = 1'b0;
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

        repeat (4) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        axi_read(REG_STATUS, rd_value);
        check_equal(rd_value, 32'd0, "status_after_reset");

        axi_write_aw_then_w(REG_CFG_IN_H, 32'd16);
        axi_write_w_then_aw(REG_CFG_IN_W, 32'd32);

        axi_read(REG_CFG_IN_H, rd_value);
        check_equal(rd_value, 32'd16, "cfg_in_h_readback");

        axi_read(REG_CFG_IN_W, rd_value);
        check_equal(rd_value, 32'd32, "cfg_in_w_readback");

        axi_write_simul(REG_CONTROL, 32'h0000_0006);
        axi_read(REG_CONTROL, rd_value);
        check_equal(rd_value, 32'h0000_0006, "control_readback");

        axi_write_simul(REG_FM_ADDR, 32'd5);
        axi_write_simul(REG_FM_WDATA, 32'h0000_00a5);
        axi_write_simul(REG_FM_CMD, 32'd1);
        axi_read(REG_FM_RDATA, rd_value);
        check_equal(rd_value, 32'h0000_00a5, "fm_byte_readback");

        axi_write_simul(REG_MASK_ADDR, 32'd3);
        axi_write_simul(REG_MASK_DATA, 32'd1);
        axi_write_simul(REG_MASK_CMD, 32'd1);
        @(posedge clk);
        check_equal({31'd0, dut.u_regs.u_wrap.u_core.u_tile_mask_mem.mem[3][0]}, 32'd1, "mask_mem_write");

        axi_write_simul(REG_PARAM_SEL, 32'd11);
        axi_write_simul(REG_PARAM_ADDR, 32'h0000_0123);
        axi_write_simul(REG_PARAM_DATA, 32'hdead_beef);
        axi_read(REG_PARAM_SEL, rd_value);
        check_equal(rd_value, 32'd11, "param_sel_readback");
        axi_read(REG_PARAM_ADDR, rd_value);
        check_equal(rd_value, 32'h0000_0123, "param_addr_readback");
        axi_read(REG_PARAM_DATA, rd_value);
        check_equal(rd_value, 32'hdead_beef, "param_data_readback");

        if (irq !== 1'b0) begin
            failures = failures + 1;
            $display("FAIL irq should remain low during control-plane test");
        end else begin
            $display("PASS irq remains low");
        end

        if (failures == 0) begin
            $display("TB PASSED");
        end else begin
            $display("TB FAILED failures=%0d", failures);
        end
        $finish;
    end
endmodule
