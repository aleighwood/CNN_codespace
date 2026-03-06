`timescale 1ns/1ps

module tb_mobilenet_v1_axi_ctrl_bram;
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

    logic host_fm_en;
    logic [((DATA_W + 7) / 8)-1:0] host_fm_we;
    logic [ADDR_W-1:0] host_fm_addr;
    logic [DATA_W-1:0] host_fm_din;
    logic [DATA_W-1:0] host_fm_dout;

    logic irq;

    integer failures;
    logic [31:0] rd_value;

    mobilenet_v1_axi_ctrl_bram #(
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
        .host_fm_en(host_fm_en),
        .host_fm_we(host_fm_we),
        .host_fm_addr(host_fm_addr),
        .host_fm_din(host_fm_din),
        .host_fm_dout(host_fm_dout),
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

    task automatic axi_write(
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

    task automatic host_fm_write_byte(
        input [ADDR_W-1:0] addr,
        input [DATA_W-1:0] data
    );
        begin
            @(posedge clk);
            host_fm_addr = addr;
            host_fm_din = data;
            host_fm_we = '1;
            host_fm_en = 1'b1;
            @(posedge clk);
            host_fm_en = 1'b0;
            host_fm_we = '0;
        end
    endtask

    task automatic host_fm_read_byte(
        input [ADDR_W-1:0] addr
    );
        begin
            @(posedge clk);
            host_fm_addr = addr;
            host_fm_we = '0;
            host_fm_en = 1'b1;
            @(posedge clk);
            host_fm_en = 1'b0;
        end
    endtask

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

        host_fm_en = 1'b0;
        host_fm_we = '0;
        host_fm_addr = '0;
        host_fm_din = '0;

        repeat (5) @(posedge clk);
        rst_n = 1'b1;
        repeat (2) @(posedge clk);

        axi_read(REG_STATUS, rd_value);
        check_equal(rd_value, 32'd0, "status_after_reset");

        host_fm_write_byte(32'd4, 8'h5a);
        host_fm_read_byte(32'd4);
        check_equal({24'd0, host_fm_dout}, 32'h0000005a, "host_fm_direct_readback");

        axi_write(REG_CFG_IN_H, 32'd160);
        axi_read(REG_CFG_IN_H, rd_value);
        check_equal(rd_value, 32'd160, "cfg_in_h_readback");

        axi_write(REG_CFG_IN_W, 32'd192);
        axi_read(REG_CFG_IN_W, rd_value);
        check_equal(rd_value, 32'd192, "cfg_in_w_readback");

        axi_write(REG_CONTROL, 32'h0000_0002);
        axi_read(REG_CONTROL, rd_value);
        check_equal(rd_value, 32'h0000_0002, "control_readback");

        axi_write(REG_MASK_ADDR, 32'd3);
        axi_write(REG_MASK_DATA, 32'd1);
        axi_write(REG_MASK_CMD, 32'd1);
        check_equal({31'd0, dut.u_wrap.u_core.u_tile_mask_mem.mem[3][0]}, 32'd1, "mask_mem_write");

        axi_write(REG_PARAM_SEL, 32'd21);
        axi_write(REG_PARAM_ADDR, 32'd7);
        axi_write(REG_PARAM_DATA, 32'h12345678);
        axi_read(REG_PARAM_SEL, rd_value);
        check_equal(rd_value, 32'd21, "param_sel_readback");
        axi_read(REG_PARAM_ADDR, rd_value);
        check_equal(rd_value, 32'd7, "param_addr_readback");
        axi_read(REG_PARAM_DATA, rd_value);
        check_equal(rd_value, 32'h12345678, "param_data_readback");

        check_equal({31'd0, irq}, 32'd0, "irq_idle_low");

        if (failures == 0) begin
            $display("TB PASSED");
        end else begin
            $display("TB FAILED failures=%0d", failures);
        end
        $finish;
    end
endmodule
