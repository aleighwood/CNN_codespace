module dual_port_ram_async #(
    parameter int DATA_W = 8,
    parameter int DEPTH = 1024,
    parameter string MEM_STYLE = "auto",
    parameter string INIT_FILE = "",
    parameter bit USE_XPM = 1'b0
) (
    input  logic clk,

    input  logic a_en,
    input  logic a_we,
    input  logic [$clog2((DEPTH <= 1) ? 2 : DEPTH)-1:0] a_addr,
    input  logic [DATA_W-1:0] a_din,
    output logic [DATA_W-1:0] a_dout,

    input  logic b_en,
    input  logic b_we,
    input  logic [$clog2((DEPTH <= 1) ? 2 : DEPTH)-1:0] b_addr,
    input  logic [DATA_W-1:0] b_din,
    output logic [DATA_W-1:0] b_dout
);
    localparam int DEPTH_USE = (DEPTH <= 1) ? 2 : DEPTH;
    localparam int ADDR_W = $clog2(DEPTH_USE);
`ifdef XILINX_XPM
    localparam string XPM_MEM_PRIMITIVE = (MEM_STYLE == "ultra") ? "ultra" :
                                          (MEM_STYLE == "block") ? "block" : "auto";
    localparam string XPM_INIT_FILE = (INIT_FILE == "") ? "none" : INIT_FILE;
`endif

    generate
`ifdef XILINX_XPM
        if (USE_XPM) begin : gen_xpm
            logic [DATA_W-1:0] a_dout_xpm;
            logic [DATA_W-1:0] b_dout_xpm;

            // XPM BRAM/URAM uses synchronous reads. Keep this path opt-in until
            // the compute core is fully retimed for 1-cycle memory latency.
            xpm_memory_tdpram #(
                .ADDR_WIDTH_A(ADDR_W),
                .ADDR_WIDTH_B(ADDR_W),
                .AUTO_SLEEP_TIME(0),
                .BYTE_WRITE_WIDTH_A(DATA_W),
                .BYTE_WRITE_WIDTH_B(DATA_W),
                .CASCADE_HEIGHT(0),
                .CLOCKING_MODE("common_clock"),
                .ECC_MODE("no_ecc"),
                .MEMORY_INIT_FILE(XPM_INIT_FILE),
                .MEMORY_INIT_PARAM("0"),
                .MEMORY_OPTIMIZATION("true"),
                .MEMORY_PRIMITIVE(XPM_MEM_PRIMITIVE),
                .MEMORY_SIZE(DATA_W * DEPTH_USE),
                .MESSAGE_CONTROL(0),
                .READ_DATA_WIDTH_A(DATA_W),
                .READ_DATA_WIDTH_B(DATA_W),
                .READ_LATENCY_A(1),
                .READ_LATENCY_B(1),
                .READ_RESET_VALUE_A("0"),
                .READ_RESET_VALUE_B("0"),
                .RST_MODE_A("SYNC"),
                .RST_MODE_B("SYNC"),
                .SIM_ASSERT_CHK(0),
                .USE_EMBEDDED_CONSTRAINT(0),
                .USE_MEM_INIT(1),
                .WAKEUP_TIME("disable_sleep"),
                .WRITE_DATA_WIDTH_A(DATA_W),
                .WRITE_DATA_WIDTH_B(DATA_W),
                .WRITE_MODE_A("read_first"),
                .WRITE_MODE_B("read_first")
            ) u_xpm (
                .dbiterra(),
                .dbiterrb(),
                .douta(a_dout_xpm),
                .doutb(b_dout_xpm),
                .sbiterra(),
                .sbiterrb(),
                .addra(a_addr),
                .addrb(b_addr),
                .clka(clk),
                .clkb(clk),
                .dina(a_din),
                .dinb(b_din),
                .ena(a_en),
                .enb(b_en),
                .injectdbiterra(1'b0),
                .injectdbiterrb(1'b0),
                .injectsbiterra(1'b0),
                .injectsbiterrb(1'b0),
                .regcea(1'b1),
                .regceb(1'b1),
                .rsta(1'b0),
                .rstb(1'b0),
                .sleep(1'b0),
                .wea(a_we),
                .web(b_we)
            );

            assign a_dout = a_en ? a_dout_xpm : '0;
            assign b_dout = b_en ? b_dout_xpm : '0;
        end else begin : gen_model
`else
        begin : gen_model
`endif
            if (MEM_STYLE == "ultra") begin : gen_ultra
                (* ram_style = "ultra" *) logic [DATA_W-1:0] mem [0:DEPTH_USE-1];
                logic [DATA_W-1:0] a_dout_reg;
                logic [DATA_W-1:0] b_dout_reg;

                initial begin
                    if (INIT_FILE != "") begin
                        $readmemh(INIT_FILE, mem);
                    end
                end

                always_ff @(posedge clk) begin
                    if (a_en && a_we) begin
                        mem[a_addr] <= a_din;
                    end
                    if (b_en && b_we) begin
                        mem[b_addr] <= b_din;
                    end
                end

                if (USE_XPM) begin : gen_ultra_sync
                    always_ff @(posedge clk) begin
                        a_dout_reg <= a_en ? mem[a_addr] : '0;
                        b_dout_reg <= b_en ? mem[b_addr] : '0;
                    end

                    assign a_dout = a_dout_reg;
                    assign b_dout = b_dout_reg;
                end else begin : gen_ultra_async
                    always_comb begin
                        a_dout = a_en ? mem[a_addr] : '0;
                        b_dout = b_en ? mem[b_addr] : '0;
                    end
                end
            end else if (MEM_STYLE == "block") begin : gen_block
                (* ram_style = "block" *) logic [DATA_W-1:0] mem [0:DEPTH_USE-1];
                logic [DATA_W-1:0] a_dout_reg;
                logic [DATA_W-1:0] b_dout_reg;

                initial begin
                    if (INIT_FILE != "") begin
                        $readmemh(INIT_FILE, mem);
                    end
                end

                always_ff @(posedge clk) begin
                    if (a_en && a_we) begin
                        mem[a_addr] <= a_din;
                    end
                    if (b_en && b_we) begin
                        mem[b_addr] <= b_din;
                    end
                end

                if (USE_XPM) begin : gen_block_sync
                    always_ff @(posedge clk) begin
                        a_dout_reg <= a_en ? mem[a_addr] : '0;
                        b_dout_reg <= b_en ? mem[b_addr] : '0;
                    end

                    assign a_dout = a_dout_reg;
                    assign b_dout = b_dout_reg;
                end else begin : gen_block_async
                    always_comb begin
                        a_dout = a_en ? mem[a_addr] : '0;
                        b_dout = b_en ? mem[b_addr] : '0;
                    end
                end
            end else begin : gen_auto
                logic [DATA_W-1:0] mem [0:DEPTH_USE-1];
                logic [DATA_W-1:0] a_dout_reg;
                logic [DATA_W-1:0] b_dout_reg;

                initial begin
                    if (INIT_FILE != "") begin
                        $readmemh(INIT_FILE, mem);
                    end
                end

                always_ff @(posedge clk) begin
                    if (a_en && a_we) begin
                        mem[a_addr] <= a_din;
                    end
                    if (b_en && b_we) begin
                        mem[b_addr] <= b_din;
                    end
                end

                if (USE_XPM) begin : gen_auto_sync
                    always_ff @(posedge clk) begin
                        a_dout_reg <= a_en ? mem[a_addr] : '0;
                        b_dout_reg <= b_en ? mem[b_addr] : '0;
                    end

                    assign a_dout = a_dout_reg;
                    assign b_dout = b_dout_reg;
                end else begin : gen_auto_async
                    always_comb begin
                        a_dout = a_en ? mem[a_addr] : '0;
                        b_dout = b_en ? mem[b_addr] : '0;
                    end
                end
            end
        end
`ifdef XILINX_XPM
        end
`endif
    endgenerate
endmodule
