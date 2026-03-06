module tile_mask_mem #(
    parameter int ADDR_W = 16,
    parameter int DEPTH = 4096,
    parameter string INIT_FILE = ""
) (
    input  logic clk,
    input  logic rst_n,
    input  logic wr_en,
    input  logic [ADDR_W-1:0] wr_addr,
    input  logic wr_data,
    input  logic rd_en,
    input  logic [ADDR_W-1:0] addr,
    output logic data,
    output logic data_valid
);
    localparam int IDX_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH);
    logic [7:0] mem [0:DEPTH-1];

    initial begin
        if (INIT_FILE != "") begin
            $readmemh(INIT_FILE, mem);
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            data <= 1'b0;
            data_valid <= 1'b0;
        end else begin
            data_valid <= rd_en;
            if (wr_en) begin
                mem[wr_addr[IDX_W-1:0]] <= {7'd0, wr_data};
            end
            if (rd_en) begin
                data <= mem[addr[IDX_W-1:0]][0];
            end
        end
    end
endmodule
