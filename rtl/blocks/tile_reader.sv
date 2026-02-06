module tile_reader #(
    parameter int DATA_W = 8,
    parameter int ADDR_W = 32,
    parameter int DIM_W = 16
) (
    input  logic clk,
    input  logic rst_n,

    input  logic start,

    input  logic [DIM_W-1:0] cfg_img_h,
    input  logic [DIM_W-1:0] cfg_img_w,
    input  logic [ADDR_W-1:0] cfg_base_addr,
    input  logic signed [DIM_W:0] cfg_tile_in_row,
    input  logic signed [DIM_W:0] cfg_tile_in_col,
    input  logic [DIM_W-1:0] cfg_tile_in_h,
    input  logic [DIM_W-1:0] cfg_tile_in_w,
    input  logic signed [DATA_W-1:0] cfg_pad_value,

    output logic rd_en,
    output logic [ADDR_W-1:0] rd_addr,
    input  logic [DATA_W-1:0] rd_data,

    output logic out_valid,
    input  logic out_ready,
    output logic [DATA_W-1:0] out_data,

    output logic done
);
    logic active;

    logic [DIM_W-1:0] img_h_reg;
    logic [DIM_W-1:0] img_w_reg;
    logic [ADDR_W-1:0] base_addr_reg;
    logic signed [DIM_W:0] tile_in_row_reg;
    logic signed [DIM_W:0] tile_in_col_reg;
    logic [DIM_W-1:0] tile_in_h_reg;
    logic [DIM_W-1:0] tile_in_w_reg;
    logic signed [DATA_W-1:0] pad_value_reg;

    logic [DIM_W-1:0] row_idx;
    logic [DIM_W-1:0] col_idx;

    logic req_valid;
    logic pending_pad;
    logic pending_last;

    logic [DATA_W-1:0] out_data_reg;
    logic out_valid_reg;
    logic out_last_reg;

    logic signed [DIM_W:0] row_abs;
    logic signed [DIM_W:0] col_abs;
    logic row_in_bounds;
    logic col_in_bounds;
    logic in_bounds;

    logic [DIM_W:0] row_u;
    logic [DIM_W:0] col_u;
    logic [ADDR_W-1:0] addr_calc;

    logic last_pixel;
    logic issue_req;

    assign out_valid = out_valid_reg;
    assign out_data = out_data_reg;

    assign row_abs = tile_in_row_reg + $signed({1'b0, row_idx});
    assign col_abs = tile_in_col_reg + $signed({1'b0, col_idx});

    assign row_in_bounds = (row_abs >= 0) && (row_abs < $signed({1'b0, img_h_reg}));
    assign col_in_bounds = (col_abs >= 0) && (col_abs < $signed({1'b0, img_w_reg}));
    assign in_bounds = row_in_bounds && col_in_bounds;

    assign row_u = row_abs[DIM_W:0];
    assign col_u = col_abs[DIM_W:0];

    assign addr_calc = base_addr_reg + (row_u * img_w_reg) + col_u;

    assign last_pixel = (row_idx == tile_in_h_reg - 1'b1) &&
                        (col_idx == tile_in_w_reg - 1'b1);

    assign issue_req = active && !req_valid && (!out_valid_reg || out_ready);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active <= 1'b0;
            img_h_reg <= '0;
            img_w_reg <= '0;
            base_addr_reg <= '0;
            tile_in_row_reg <= '0;
            tile_in_col_reg <= '0;
            tile_in_h_reg <= '0;
            tile_in_w_reg <= '0;
            pad_value_reg <= '0;
            row_idx <= '0;
            col_idx <= '0;
            req_valid <= 1'b0;
            pending_pad <= 1'b0;
            pending_last <= 1'b0;
            out_valid_reg <= 1'b0;
            out_data_reg <= '0;
            out_last_reg <= 1'b0;
            done <= 1'b0;
            rd_en <= 1'b0;
            rd_addr <= '0;
        end else begin
            rd_en <= 1'b0;
            done <= 1'b0;

            if (start) begin
                active <= 1'b1;
                img_h_reg <= cfg_img_h;
                img_w_reg <= cfg_img_w;
                base_addr_reg <= cfg_base_addr;
                tile_in_row_reg <= cfg_tile_in_row;
                tile_in_col_reg <= cfg_tile_in_col;
                tile_in_h_reg <= cfg_tile_in_h;
                tile_in_w_reg <= cfg_tile_in_w;
                pad_value_reg <= cfg_pad_value;
                row_idx <= '0;
                col_idx <= '0;
                req_valid <= 1'b0;
                out_valid_reg <= 1'b0;
                out_last_reg <= 1'b0;
            end

            if (out_valid_reg && out_ready) begin
                out_valid_reg <= 1'b0;
                if (out_last_reg) begin
                    active <= 1'b0;
                    done <= 1'b1;
                end
            end

            if (req_valid) begin
                if (!out_valid_reg) begin
                    out_valid_reg <= 1'b1;
                    out_data_reg <= pending_pad ? pad_value_reg : rd_data;
                    out_last_reg <= pending_last;
                    req_valid <= 1'b0;
                end
            end

            if (issue_req) begin
                pending_pad <= !in_bounds;
                pending_last <= last_pixel;
                req_valid <= 1'b1;

                if (in_bounds) begin
                    rd_en <= 1'b1;
                    rd_addr <= addr_calc;
                end

                if (col_idx == tile_in_w_reg - 1'b1) begin
                    col_idx <= '0;
                    if (row_idx == tile_in_h_reg - 1'b1) begin
                        row_idx <= '0;
                    end else begin
                        row_idx <= row_idx + 1'b1;
                    end
                end else begin
                    col_idx <= col_idx + 1'b1;
                end
            end
        end
    end
endmodule
