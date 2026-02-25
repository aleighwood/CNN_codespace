module tile_mask_rom #(
    parameter int ADDR_W = 16,
    parameter int DEPTH = 4096,
    parameter string INIT_FILE = ""
) (
    input  logic [ADDR_W-1:0] addr,
    output logic data
);
    localparam int IDX_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH);
    logic [7:0] mem [0:DEPTH-1];

    initial begin
        if (INIT_FILE != "") begin
            $readmemh(INIT_FILE, mem);
        end
    end

    assign data = mem[addr[IDX_W-1:0]][0];
endmodule
