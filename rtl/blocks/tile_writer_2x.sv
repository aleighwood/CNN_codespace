module tile_writer_2x #(
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
    input  logic in_two,
    output logic in_ready,
    input  logic [DATA_W-1:0] in_data0,
    input  logic [DATA_W-1:0] in_data1,

    output logic wr_en0,
    output logic [ADDR_W-1:0] wr_addr0,
    output logic [DATA_W-1:0] wr_data0,
    output logic wr_en1,
    output logic [ADDR_W-1:0] wr_addr1,
    output logic [DATA_W-1:0] wr_data1,

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

    logic [DIM_W-1:0] row1;
    logic [DIM_W-1:0] col1;
    logic [DIM_W-1:0] row2;
    logic [DIM_W-1:0] col2;
    logic last0;
    logic last1;

    assign active_this = active || start;
    assign in_ready = active_this;

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

        row1 = row_idx_use;
        col1 = col_idx_use;
        if (col1 == tile_out_w_use - 1'b1) begin
            col1 = '0;
            row1 = row1 + 1'b1;
        end else begin
            col1 = col1 + 1'b1;
        end

        row2 = row1;
        col2 = col1;
        if (col2 == tile_out_w_use - 1'b1) begin
            col2 = '0;
            row2 = row2 + 1'b1;
        end else begin
            col2 = col2 + 1'b1;
        end

        last0 = (row_idx_use == tile_out_h_use - 1'b1) &&
                (col_idx_use == tile_out_w_use - 1'b1);
        last1 = (row1 == tile_out_h_use - 1'b1) &&
                (col1 == tile_out_w_use - 1'b1);
    end

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
            wr_en0 <= 1'b0;
            wr_en1 <= 1'b0;
            wr_addr0 <= '0;
            wr_addr1 <= '0;
            wr_data0 <= '0;
            wr_data1 <= '0;
            done <= 1'b0;
        end else begin
            wr_en0 <= 1'b0;
            wr_en1 <= 1'b0;
            done <= 1'b0;
            wr_data0 <= '0;
            wr_data1 <= '0;

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
                wr_en0 <= 1'b1;
                wr_data0 <= in_data0;
                wr_addr0 <= base_addr_use +
                            ((tile_out_row_use + row_idx_use) * img_w_use) +
                            (tile_out_col_use + col_idx_use);

                if (in_two) begin
                    wr_en1 <= 1'b1;
                    wr_data1 <= in_data1;
                    wr_addr1 <= base_addr_use +
                                ((tile_out_row_use + row1) * img_w_use) +
                                (tile_out_col_use + col1);
                end

                if ((last0 && !in_two) || (in_two && last1)) begin
                    active <= 1'b0;
                    done <= 1'b1;
                    row_idx <= '0;
                    col_idx <= '0;
                end else if (in_two) begin
                    row_idx <= row2;
                    col_idx <= col2;
                end else begin
                    row_idx <= row1;
                    col_idx <= col1;
                end
            end
        end
    end
endmodule
