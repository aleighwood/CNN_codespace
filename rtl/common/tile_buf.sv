module tile_buf #(
    parameter int DATA_W = 8,
    parameter int DEPTH = 4096,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
) (
    input  logic clk,
    input  logic rst_n,

    input  logic wr_en,
    input  logic [ADDR_W-1:0] wr_addr,
    input  logic [DATA_W-1:0] wr_data,

    input  logic rd_en,
    input  logic [ADDR_W-1:0] rd_addr,
    output logic [DATA_W-1:0] rd_data
);
    logic [DATA_W-1:0] mem [0:DEPTH-1];

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            rd_data <= '0;
        end else begin
            if (wr_en) begin
                mem[wr_addr] <= wr_data;
            end
            if (rd_en) begin
                rd_data <= mem[rd_addr];
            end
        end
    end
endmodule
