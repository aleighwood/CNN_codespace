module pw_tile_reader #(
    parameter int DATA_W = 8,
    parameter int ADDR_W = 32,
    parameter int DIM_W = 16
) (
    input  logic clk,
    input  logic rst_n,

    input  logic start,

    input  logic [DIM_W-1:0] cfg_tile_h,
    input  logic [DIM_W-1:0] cfg_tile_w,
    input  logic [DIM_W-1:0] cfg_channels,
    input  logic [ADDR_W-1:0] cfg_base_addr,

    output logic rd_en,
    output logic [ADDR_W-1:0] rd_addr,
    input  logic [DATA_W-1:0] rd_data,

    output logic out_valid,
    input  logic out_ready,
    output logic [DATA_W-1:0] out_data,
    output logic out_first_ch,
    output logic out_last_ch,
    output logic [DIM_W-1:0] out_in_ch_idx,

    output logic done
);
    logic active;

    logic [DIM_W-1:0] tile_h_reg;
    logic [DIM_W-1:0] tile_w_reg;
    logic [DIM_W-1:0] channels_reg;
    logic [ADDR_W-1:0] base_addr_reg;

    logic [DIM_W-1:0] row_idx;
    logic [DIM_W-1:0] col_idx;
    logic [DIM_W-1:0] ch_idx;

    logic req_valid;
    logic pending_first;
    logic pending_last;
    logic pending_done;
    logic [DIM_W-1:0] pending_in_ch;

    logic [DATA_W-1:0] out_data_reg;
    logic out_valid_reg;
    logic out_first_reg;
    logic out_last_reg;
    logic out_done_reg;
    logic [DIM_W-1:0] out_in_ch_reg;

    logic last_elem;
    logic issue_req;

    assign out_valid = out_valid_reg;
    assign out_data = out_data_reg;
    assign out_first_ch = out_first_reg;
    assign out_last_ch = out_last_reg;
    assign out_in_ch_idx = out_in_ch_reg;

    assign last_elem = (row_idx == tile_h_reg - 1'b1) &&
                       (col_idx == tile_w_reg - 1'b1) &&
                       (ch_idx == channels_reg - 1'b1);

    assign issue_req = active && !req_valid && (!out_valid_reg || out_ready);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            active <= 1'b0;
            tile_h_reg <= '0;
            tile_w_reg <= '0;
            channels_reg <= '0;
            base_addr_reg <= '0;
            row_idx <= '0;
            col_idx <= '0;
            ch_idx <= '0;
            req_valid <= 1'b0;
            pending_first <= 1'b0;
            pending_last <= 1'b0;
            pending_done <= 1'b0;
            pending_in_ch <= '0;
            out_valid_reg <= 1'b0;
            out_data_reg <= '0;
            out_first_reg <= 1'b0;
            out_last_reg <= 1'b0;
            out_done_reg <= 1'b0;
            out_in_ch_reg <= '0;
            done <= 1'b0;
            rd_en <= 1'b0;
            rd_addr <= '0;
        end else begin
            rd_en <= 1'b0;
            done <= 1'b0;

            if (start) begin
                active <= 1'b1;
                tile_h_reg <= cfg_tile_h;
                tile_w_reg <= cfg_tile_w;
                channels_reg <= cfg_channels;
                base_addr_reg <= cfg_base_addr;
                row_idx <= '0;
                col_idx <= '0;
                ch_idx <= '0;
                req_valid <= 1'b0;
                out_valid_reg <= 1'b0;
            end

            if (out_valid_reg && out_ready) begin
                out_valid_reg <= 1'b0;
                if (out_done_reg) begin
                    active <= 1'b0;
                    done <= 1'b1;
                end
            end

            if (req_valid) begin
                if (!out_valid_reg) begin
                    out_valid_reg <= 1'b1;
                    out_data_reg <= rd_data;
                    out_first_reg <= pending_first;
                    out_last_reg <= pending_last;
                    out_done_reg <= pending_done;
                    out_in_ch_reg <= pending_in_ch;
                    req_valid <= 1'b0;
                end
            end

            if (issue_req) begin
                pending_first <= (ch_idx == '0);
                pending_last <= (ch_idx == channels_reg - 1'b1);
                pending_done <= last_elem;
                pending_in_ch <= ch_idx;
                req_valid <= 1'b1;

                rd_en <= 1'b1;
                rd_addr <= base_addr_reg +
                           (ch_idx * (tile_h_reg * tile_w_reg)) +
                           (row_idx * tile_w_reg) +
                           col_idx;

                if (ch_idx == channels_reg - 1'b1) begin
                    ch_idx <= '0;
                    if (col_idx == tile_w_reg - 1'b1) begin
                        col_idx <= '0;
                        if (row_idx == tile_h_reg - 1'b1) begin
                            row_idx <= '0;
                        end else begin
                            row_idx <= row_idx + 1'b1;
                        end
                    end else begin
                        col_idx <= col_idx + 1'b1;
                    end
                end else begin
                    ch_idx <= ch_idx + 1'b1;
                end
            end
        end
    end
endmodule
