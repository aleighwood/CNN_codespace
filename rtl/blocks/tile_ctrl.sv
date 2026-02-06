module tile_ctrl #(
    parameter int DIM_W = 16
) (
    input  logic clk,
    input  logic rst_n,

    input  logic cfg_valid,
    output logic cfg_ready,
    input  logic [DIM_W-1:0] cfg_img_h,
    input  logic [DIM_W-1:0] cfg_img_w,
    input  logic [DIM_W-1:0] cfg_tile_h,
    input  logic [DIM_W-1:0] cfg_tile_w,
    input  logic [DIM_W-1:0] cfg_stride,
    input  logic [DIM_W-1:0] cfg_pad,
    input  logic [DIM_W-1:0] cfg_kernel,

    input  logic start,

    output logic tile_valid,
    input  logic tile_ready,
    output logic [DIM_W-1:0] tile_out_row,
    output logic [DIM_W-1:0] tile_out_col,
    output logic [DIM_W-1:0] tile_out_h,
    output logic [DIM_W-1:0] tile_out_w,
    output logic signed [DIM_W:0] tile_in_row,
    output logic signed [DIM_W:0] tile_in_col,
    output logic [DIM_W-1:0] tile_in_h,
    output logic [DIM_W-1:0] tile_in_w,

    output logic done
);
    function automatic int ceil_div(input int a, input int b);
        if (b <= 0) begin
            return 0;
        end
        return (a + b - 1) / b;
    endfunction

    logic cfg_loaded;
    logic active;

    logic [DIM_W-1:0] img_h_reg;
    logic [DIM_W-1:0] img_w_reg;
    logic [DIM_W-1:0] tile_h_reg;
    logic [DIM_W-1:0] tile_w_reg;
    logic [DIM_W-1:0] stride_reg;
    logic [DIM_W-1:0] pad_reg;
    logic [DIM_W-1:0] kernel_reg;

    logic [DIM_W-1:0] out_h_reg;
    logic [DIM_W-1:0] out_w_reg;
    logic [DIM_W-1:0] tiles_h_reg;
    logic [DIM_W-1:0] tiles_w_reg;

    logic [DIM_W-1:0] tile_row_idx;
    logic [DIM_W-1:0] tile_col_idx;

    logic last_tile;
    logic handshake;

    logic [DIM_W-1:0] pad_top;
    logic [DIM_W-1:0] pad_left;

    assign cfg_ready = !active;
    assign handshake = tile_valid && tile_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cfg_loaded <= 1'b0;
            active <= 1'b0;
            img_h_reg <= '0;
            img_w_reg <= '0;
            tile_h_reg <= '0;
            tile_w_reg <= '0;
            stride_reg <= '0;
            pad_reg <= '0;
            kernel_reg <= '0;
            out_h_reg <= '0;
            out_w_reg <= '0;
            tiles_h_reg <= '0;
            tiles_w_reg <= '0;
            tile_row_idx <= '0;
            tile_col_idx <= '0;
            done <= 1'b0;
        end else begin
            done <= 1'b0;

            if (cfg_valid && cfg_ready) begin
                img_h_reg <= cfg_img_h;
                img_w_reg <= cfg_img_w;
                tile_h_reg <= cfg_tile_h;
                tile_w_reg <= cfg_tile_w;
                stride_reg <= cfg_stride;
                pad_reg <= cfg_pad;
                kernel_reg <= cfg_kernel;

                out_h_reg <= ceil_div(cfg_img_h, cfg_stride);
                out_w_reg <= ceil_div(cfg_img_w, cfg_stride);
                tiles_h_reg <= ceil_div(ceil_div(cfg_img_h, cfg_stride), cfg_tile_h);
                tiles_w_reg <= ceil_div(ceil_div(cfg_img_w, cfg_stride), cfg_tile_w);
                cfg_loaded <= 1'b1;
            end

            if (start && cfg_loaded && !active) begin
                active <= 1'b1;
                tile_row_idx <= '0;
                tile_col_idx <= '0;
            end

            if (active && handshake) begin
                if (last_tile) begin
                    active <= 1'b0;
                    done <= 1'b1;
                end else begin
                    if (tile_col_idx == tiles_w_reg - 1'b1) begin
                        tile_col_idx <= '0;
                        tile_row_idx <= tile_row_idx + 1'b1;
                    end else begin
                        tile_col_idx <= tile_col_idx + 1'b1;
                    end
                end
            end
        end
    end

    always_comb begin
        int pad_h_calc;
        int pad_w_calc;
        int pad_top_calc;
        int pad_left_calc;

        pad_h_calc = (out_h_reg - 1) * stride_reg + kernel_reg - img_h_reg;
        if (pad_h_calc < 0) begin
            pad_h_calc = 0;
        end
        pad_top_calc = pad_h_calc / 2;

        pad_w_calc = (out_w_reg - 1) * stride_reg + kernel_reg - img_w_reg;
        if (pad_w_calc < 0) begin
            pad_w_calc = 0;
        end
        pad_left_calc = pad_w_calc / 2;

        pad_top = pad_top_calc[DIM_W-1:0];
        pad_left = pad_left_calc[DIM_W-1:0];

        tile_out_row = tile_row_idx * tile_h_reg;
        tile_out_col = tile_col_idx * tile_w_reg;

        if (tile_out_row + tile_h_reg > out_h_reg) begin
            tile_out_h = out_h_reg - tile_out_row;
        end else begin
            tile_out_h = tile_h_reg;
        end

        if (tile_out_col + tile_w_reg > out_w_reg) begin
            tile_out_w = out_w_reg - tile_out_col;
        end else begin
            tile_out_w = tile_w_reg;
        end

        tile_in_row = $signed({1'b0, tile_out_row} * stride_reg) - $signed({1'b0, pad_top});
        tile_in_col = $signed({1'b0, tile_out_col} * stride_reg) - $signed({1'b0, pad_left});

        tile_in_h = (tile_out_h - 1'b1) * stride_reg + kernel_reg;
        tile_in_w = (tile_out_w - 1'b1) * stride_reg + kernel_reg;
    end

    assign last_tile = (tile_row_idx == tiles_h_reg - 1'b1) &&
                       (tile_col_idx == tiles_w_reg - 1'b1);

    assign tile_valid = active;
endmodule
