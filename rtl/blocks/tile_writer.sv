module tile_writer #(
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
    input  logic [DIM_W-1:0] cfg_tile_out_row,
    input  logic [DIM_W-1:0] cfg_tile_out_col,
    input  logic [DIM_W-1:0] cfg_tile_out_h,
    input  logic [DIM_W-1:0] cfg_tile_out_w,

    input  logic in_valid,
    output logic in_ready,
    input  logic [DATA_W-1:0] in_data,

    output logic wr_en,
    output logic [ADDR_W-1:0] wr_addr,
    output logic [DATA_W-1:0] wr_data,

    output logic done
);
    logic active;

    logic [DIM_W-1:0] img_h_reg;
    logic [DIM_W-1:0] img_w_reg;
    logic [ADDR_W-1:0] base_addr_reg;
    logic [DIM_W-1:0] tile_out_row_reg;
    logic [DIM_W-1:0] tile_out_col_reg;
    logic [DIM_W-1:0] tile_out_h_reg;
    logic [DIM_W-1:0] tile_out_w_reg;

    logic [DIM_W-1:0] row_idx;
    logic [DIM_W-1:0] col_idx;

    logic last_pixel;
    logic active_this;
    logic [DIM_W-1:0] row_idx_use;
    logic [DIM_W-1:0] col_idx_use;
    logic [DIM_W-1:0] img_h_use;
    logic [DIM_W-1:0] img_w_use;
    logic [ADDR_W-1:0] base_addr_use;
    logic [DIM_W-1:0] tile_out_row_use;
    logic [DIM_W-1:0] tile_out_col_use;
    logic [DIM_W-1:0] tile_out_h_use;
    logic [DIM_W-1:0] tile_out_w_use;

    assign active_this = active || start;
    assign in_ready = active_this;
    logic [DATA_W-1:0] wr_data_reg;
    assign wr_data = wr_data_reg;

    always_comb begin
        if (start) begin
            row_idx_use = '0;
            col_idx_use = '0;
            img_h_use = cfg_img_h;
            img_w_use = cfg_img_w;
            base_addr_use = cfg_base_addr;
            tile_out_row_use = cfg_tile_out_row;
            tile_out_col_use = cfg_tile_out_col;
            tile_out_h_use = cfg_tile_out_h;
            tile_out_w_use = cfg_tile_out_w;
        end else begin
            row_idx_use = row_idx;
            col_idx_use = col_idx;
            img_h_use = img_h_reg;
            img_w_use = img_w_reg;
            base_addr_use = base_addr_reg;
            tile_out_row_use = tile_out_row_reg;
            tile_out_col_use = tile_out_col_reg;
            tile_out_h_use = tile_out_h_reg;
            tile_out_w_use = tile_out_w_reg;
        end
    end

    assign last_pixel = (row_idx_use == tile_out_h_use - 1'b1) &&
                        (col_idx_use == tile_out_w_use - 1'b1);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active <= 1'b0;
            img_h_reg <= '0;
            img_w_reg <= '0;
            base_addr_reg <= '0;
            tile_out_row_reg <= '0;
            tile_out_col_reg <= '0;
            tile_out_h_reg <= '0;
            tile_out_w_reg <= '0;
            row_idx <= '0;
            col_idx <= '0;
            wr_en <= 1'b0;
            wr_addr <= '0;
            done <= 1'b0;
        end else begin
            wr_en <= 1'b0;
            done <= 1'b0;
            wr_data_reg <= '0;

            if (start) begin
                active <= 1'b1;
                img_h_reg <= cfg_img_h;
                img_w_reg <= cfg_img_w;
                base_addr_reg <= cfg_base_addr;
                tile_out_row_reg <= cfg_tile_out_row;
                tile_out_col_reg <= cfg_tile_out_col;
                tile_out_h_reg <= cfg_tile_out_h;
                tile_out_w_reg <= cfg_tile_out_w;
                row_idx <= '0;
                col_idx <= '0;
            end

            if (active_this && in_valid) begin
                wr_en <= 1'b1;
                wr_data_reg <= in_data;
                wr_addr <= base_addr_use +
                           ((tile_out_row_use + row_idx_use) * img_w_use) +
                           (tile_out_col_use + col_idx_use);

                if (last_pixel) begin
                    active <= 1'b0;
                    done <= 1'b1;
                end

                if (col_idx_use == tile_out_w_use - 1'b1) begin
                    col_idx <= '0;
                    if (row_idx_use == tile_out_h_use - 1'b1) begin
                        row_idx <= '0;
                    end else begin
                        row_idx <= row_idx_use + 1'b1;
                    end
                end else begin
                    col_idx <= col_idx_use + 1'b1;
                end
            end
        end
    end
endmodule
