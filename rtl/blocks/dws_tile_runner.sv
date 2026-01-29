module dws_tile_runner #(
    parameter int DATA_W = 8,
    parameter int ACC_W = 32,
    parameter int MUL_W = 16,
    parameter int BIAS_W = 32,
    parameter int SHIFT_W = 6,
    parameter int ADDR_W = 32,
    parameter int DIM_W = 16,
    parameter int MAX_TILE_IN_W = 33,
    parameter int MAX_TILE_IN_H = 33,
    parameter int PW_GROUP = 4
) (
    input  logic clk,
    input  logic rst_n,

    input  logic start,
    output logic busy,
    output logic done,

    input  logic [DIM_W-1:0] cfg_in_img_h,
    input  logic [DIM_W-1:0] cfg_in_img_w,
    input  logic [DIM_W-1:0] cfg_out_img_h,
    input  logic [DIM_W-1:0] cfg_out_img_w,
    input  logic signed [DIM_W:0] cfg_tile_in_row,
    input  logic signed [DIM_W:0] cfg_tile_in_col,
    input  logic [DIM_W-1:0] cfg_tile_in_h,
    input  logic [DIM_W-1:0] cfg_tile_in_w,
    input  logic [DIM_W-1:0] cfg_tile_out_row,
    input  logic [DIM_W-1:0] cfg_tile_out_col,
    input  logic [DIM_W-1:0] cfg_tile_out_h,
    input  logic [DIM_W-1:0] cfg_tile_out_w,
    input  logic [DIM_W-1:0] cfg_in_channels,
    input  logic [DIM_W-1:0] cfg_out_channels,
    input  logic [DIM_W-1:0] cfg_stride,

    output logic pw_group_req,
    output logic [DIM_W-1:0] pw_group_idx,
    input  logic pw_group_ready,

    input  logic [ADDR_W-1:0] cfg_in_base_addr,
    input  logic [ADDR_W-1:0] cfg_out_base_addr,
    input  logic [ADDR_W-1:0] cfg_dw_buf_base_addr,

    output logic in_rd_en,
    output logic [ADDR_W-1:0] in_rd_addr,
    input  logic [DATA_W-1:0] in_rd_data,

    output logic dw_buf_wr_en,
    output logic [ADDR_W-1:0] dw_buf_wr_addr,
    output logic [DATA_W-1:0] dw_buf_wr_data,

    output logic dw_buf_rd_en,
    output logic [ADDR_W-1:0] dw_buf_rd_addr,
    input  logic [DATA_W-1:0] dw_buf_rd_data,

    output logic out_wr_en,
    output logic [ADDR_W-1:0] out_wr_addr,
    output logic [DATA_W-1:0] out_wr_data,

    input  logic signed [DATA_W*9-1:0] dw_weight_flat,
    input  logic signed [MUL_W-1:0] dw_mul,
    input  logic signed [BIAS_W-1:0] dw_bias,
    input  logic [SHIFT_W-1:0] dw_shift,
    input  logic signed [DATA_W-1:0] dw_relu6_max,

    input  logic signed [DATA_W-1:0] pw_weight,
    input  logic signed [ACC_W-1:0] pw_bias_acc,
    input  logic signed [MUL_W-1:0] pw_mul,
    input  logic signed [BIAS_W-1:0] pw_bias_requant,
    input  logic [SHIFT_W-1:0] pw_shift,
    input  logic signed [DATA_W-1:0] pw_relu6_max,

    output logic [DIM_W-1:0] dw_ch_idx,
    output logic [DIM_W-1:0] pw_in_ch_idx,
    output logic [DIM_W-1:0] pw_out_ch_idx
);
    localparam int IN_ROW_W = (MAX_TILE_IN_H <= 1) ? 1 : $clog2(MAX_TILE_IN_H);
    localparam int IN_COL_W = (MAX_TILE_IN_W <= 1) ? 1 : $clog2(MAX_TILE_IN_W);

    typedef enum logic [2:0] {
        S_IDLE,
        S_DW_RUN,
        S_PW_LOAD,
        S_PW_RUN,
        S_DONE
    } state_t;

    state_t state;

    logic [DIM_W-1:0] in_img_h_reg;
    logic [DIM_W-1:0] in_img_w_reg;
    logic [DIM_W-1:0] out_img_h_reg;
    logic [DIM_W-1:0] out_img_w_reg;
    logic signed [DIM_W:0] tile_in_row_reg;
    logic signed [DIM_W:0] tile_in_col_reg;
    logic [DIM_W-1:0] tile_in_h_reg;
    logic [DIM_W-1:0] tile_in_w_reg;
    logic [DIM_W-1:0] tile_out_row_reg;
    logic [DIM_W-1:0] tile_out_col_reg;
    logic [DIM_W-1:0] tile_out_h_reg;
    logic [DIM_W-1:0] tile_out_w_reg;
    logic [DIM_W-1:0] in_channels_reg;
    logic [DIM_W-1:0] out_channels_reg;
    logic [ADDR_W-1:0] in_base_addr_reg;
    logic [ADDR_W-1:0] out_base_addr_reg;
    logic [ADDR_W-1:0] dw_buf_base_addr_reg;

    logic dw_start_pulse;
    logic pw_start_pulse;
    logic dw_active;
    logic pw_active;
    logic [DIM_W-1:0] pw_group_curr;
    logic pw_group_wait;
    logic dbg_pw_en;

    logic dw_reader_done;
    logic dw_writer_done;
    logic dw_rd_done_seen;
    logic dw_wr_done_seen;

    logic pw_reader_done;
    logic pw_writer_done;
    logic pw_rd_done_seen;
    logic pw_wr_done_seen;

    logic [ADDR_W-1:0] in_plane_stride;
    logic [ADDR_W-1:0] out_plane_stride;
    logic [ADDR_W-1:0] tile_area;

    logic [ADDR_W-1:0] in_base_ch;
    logic [ADDR_W-1:0] dw_buf_base_ch;
    logic [ADDR_W-1:0] out_base_ch;

    logic dw_in_valid;
    logic dw_in_ready;
    logic signed [DATA_W-1:0] dw_in_data;

    logic dw_out_valid;
    logic dw_out_ready;
    logic signed [DATA_W-1:0] dw_out_data;

    logic pw_in_valid;
    logic pw_in_ready;
    logic [DATA_W-1:0] pw_in_data;
    logic pw_in_first;
    logic pw_in_last;

    logic pw_acc_valid;
    logic pw_acc_ready;
    logic signed [ACC_W-1:0] pw_acc;

    logic pw_out_valid;
    logic pw_out_ready;
    logic signed [DATA_W-1:0] pw_out_data;

    assign busy = (state != S_IDLE);

    always_comb begin
        in_plane_stride = in_img_h_reg * in_img_w_reg;
        out_plane_stride = out_img_h_reg * out_img_w_reg;
        tile_area = tile_out_h_reg * tile_out_w_reg;

        in_base_ch = in_base_addr_reg + (dw_ch_idx * in_plane_stride);
        dw_buf_base_ch = dw_buf_base_addr_reg + (dw_ch_idx * tile_area);
        out_base_ch = out_base_addr_reg + (pw_out_ch_idx * out_plane_stride);
    end

    tile_reader #(
        .DATA_W(DATA_W),
        .ADDR_W(ADDR_W),
        .DIM_W(DIM_W)
    ) u_tile_reader (
        .clk(clk),
        .rst_n(rst_n),
        .start(dw_start_pulse),
        .cfg_img_h(in_img_h_reg),
        .cfg_img_w(in_img_w_reg),
        .cfg_base_addr(in_base_ch),
        .cfg_tile_in_row(tile_in_row_reg),
        .cfg_tile_in_col(tile_in_col_reg),
        .cfg_tile_in_h(tile_in_h_reg),
        .cfg_tile_in_w(tile_in_w_reg),
        .rd_en(in_rd_en),
        .rd_addr(in_rd_addr),
        .rd_data(in_rd_data),
        .out_valid(dw_in_valid),
        .out_ready(dw_in_ready),
        .out_data(dw_in_data),
        .done(dw_reader_done)
    );

    depthwise_stage #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .MAX_IMG_W(MAX_TILE_IN_W),
        .MAX_IMG_H(MAX_TILE_IN_H),
        .MUL_W(MUL_W),
        .BIAS_W(BIAS_W),
        .SHIFT_W(SHIFT_W)
    ) u_depthwise (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(dw_in_valid),
        .in_ready(dw_in_ready),
        .in_data(dw_in_data),
        .start(dw_start_pulse),
        .cfg_img_h(tile_in_h_reg[IN_ROW_W-1:0]),
        .cfg_img_w(tile_in_w_reg[IN_COL_W-1:0]),
        .cfg_stride(cfg_stride[IN_ROW_W-1:0]),
        .dw_weight_flat(dw_weight_flat),
        .dw_mul(dw_mul),
        .dw_bias(dw_bias),
        .dw_shift(dw_shift),
        .dw_relu6_max(dw_relu6_max),
        .out_valid(dw_out_valid),
        .out_ready(dw_out_ready),
        .out_data(dw_out_data)
    );

    tile_writer #(
        .DATA_W(DATA_W),
        .ADDR_W(ADDR_W),
        .DIM_W(DIM_W)
    ) u_dw_tile_writer (
        .clk(clk),
        .rst_n(rst_n),
        .start(dw_start_pulse),
        .cfg_img_h(tile_out_h_reg),
        .cfg_img_w(tile_out_w_reg),
        .cfg_base_addr(dw_buf_base_ch),
        .cfg_tile_out_row('0),
        .cfg_tile_out_col('0),
        .cfg_tile_out_h(tile_out_h_reg),
        .cfg_tile_out_w(tile_out_w_reg),
        .in_valid(dw_out_valid),
        .in_ready(dw_out_ready),
        .in_data(dw_out_data),
        .wr_en(dw_buf_wr_en),
        .wr_addr(dw_buf_wr_addr),
        .wr_data(dw_buf_wr_data),
        .done(dw_writer_done)
    );

    pw_tile_reader #(
        .DATA_W(DATA_W),
        .ADDR_W(ADDR_W),
        .DIM_W(DIM_W)
    ) u_pw_tile_reader (
        .clk(clk),
        .rst_n(rst_n),
        .start(pw_start_pulse),
        .cfg_tile_h(tile_out_h_reg),
        .cfg_tile_w(tile_out_w_reg),
        .cfg_channels(in_channels_reg),
        .cfg_base_addr(dw_buf_base_addr_reg),
        .rd_en(dw_buf_rd_en),
        .rd_addr(dw_buf_rd_addr),
        .rd_data(dw_buf_rd_data),
        .out_valid(pw_in_valid),
        .out_ready(pw_in_ready),
        .out_data(pw_in_data),
        .out_first_ch(pw_in_first),
        .out_last_ch(pw_in_last),
        .out_in_ch_idx(pw_in_ch_idx),
        .done(pw_reader_done)
    );

    pw_conv_1x1 #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W)
    ) u_pw_conv (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(pw_in_valid),
        .in_ready(pw_in_ready),
        .in_data(pw_in_data),
        .weight(pw_weight),
        .bias(pw_bias_acc),
        .first_in_ch(pw_in_first),
        .last_in_ch(pw_in_last),
        .out_valid(pw_acc_valid),
        .out_ready(pw_acc_ready),
        .out_acc(pw_acc)
    );

    requant_relu6 #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .MUL_W(MUL_W),
        .BIAS_W(BIAS_W),
        .SHIFT_W(SHIFT_W)
    ) u_pw_requant (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(pw_acc_valid),
        .in_ready(pw_acc_ready),
        .in_acc(pw_acc),
        .mul(pw_mul),
        .bias(pw_bias_requant),
        .shift(pw_shift),
        .relu6_max(pw_relu6_max),
        .relu6_en(1'b1),
        .out_valid(pw_out_valid),
        .out_ready(pw_out_ready),
        .out_q(pw_out_data)
    );

    tile_writer #(
        .DATA_W(DATA_W),
        .ADDR_W(ADDR_W),
        .DIM_W(DIM_W)
    ) u_out_tile_writer (
        .clk(clk),
        .rst_n(rst_n),
        .start(pw_start_pulse),
        .cfg_img_h(out_img_h_reg),
        .cfg_img_w(out_img_w_reg),
        .cfg_base_addr(out_base_ch),
        .cfg_tile_out_row(tile_out_row_reg),
        .cfg_tile_out_col(tile_out_col_reg),
        .cfg_tile_out_h(tile_out_h_reg),
        .cfg_tile_out_w(tile_out_w_reg),
        .in_valid(pw_out_valid),
        .in_ready(pw_out_ready),
        .in_data(pw_out_data),
        .wr_en(out_wr_en),
        .wr_addr(out_wr_addr),
        .wr_data(out_wr_data),
        .done(pw_writer_done)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 1'b0;
            dw_start_pulse <= 1'b0;
            pw_start_pulse <= 1'b0;
            dw_active <= 1'b0;
            pw_active <= 1'b0;
            dw_ch_idx <= '0;
            pw_out_ch_idx <= '0;
            in_img_h_reg <= '0;
            in_img_w_reg <= '0;
            out_img_h_reg <= '0;
            out_img_w_reg <= '0;
            tile_in_row_reg <= '0;
            tile_in_col_reg <= '0;
            tile_in_h_reg <= '0;
            tile_in_w_reg <= '0;
            tile_out_row_reg <= '0;
            tile_out_col_reg <= '0;
            tile_out_h_reg <= '0;
            tile_out_w_reg <= '0;
            in_channels_reg <= '0;
            out_channels_reg <= '0;
            in_base_addr_reg <= '0;
            out_base_addr_reg <= '0;
            dw_buf_base_addr_reg <= '0;
            dw_rd_done_seen <= 1'b0;
            dw_wr_done_seen <= 1'b0;
            pw_rd_done_seen <= 1'b0;
            pw_wr_done_seen <= 1'b0;
            pw_group_curr <= '0;
            pw_group_wait <= 1'b0;
        end else begin
            done <= 1'b0;
            dw_start_pulse <= 1'b0;
            pw_start_pulse <= 1'b0;

            if (start && state == S_IDLE) begin
                in_img_h_reg <= cfg_in_img_h;
                in_img_w_reg <= cfg_in_img_w;
                out_img_h_reg <= cfg_out_img_h;
                out_img_w_reg <= cfg_out_img_w;
                tile_in_row_reg <= cfg_tile_in_row;
                tile_in_col_reg <= cfg_tile_in_col;
                tile_in_h_reg <= cfg_tile_in_h;
                tile_in_w_reg <= cfg_tile_in_w;
                tile_out_row_reg <= cfg_tile_out_row;
                tile_out_col_reg <= cfg_tile_out_col;
                tile_out_h_reg <= cfg_tile_out_h;
                tile_out_w_reg <= cfg_tile_out_w;
                in_channels_reg <= cfg_in_channels;
                out_channels_reg <= cfg_out_channels;
                in_base_addr_reg <= cfg_in_base_addr;
                out_base_addr_reg <= cfg_out_base_addr;
                dw_buf_base_addr_reg <= cfg_dw_buf_base_addr;

                dw_ch_idx <= '0;
                pw_out_ch_idx <= '0;
                dw_rd_done_seen <= 1'b0;
                dw_wr_done_seen <= 1'b0;
                pw_rd_done_seen <= 1'b0;
                pw_wr_done_seen <= 1'b0;
                dw_active <= 1'b0;
                pw_active <= 1'b0;
                pw_group_curr <= '0;
                pw_group_wait <= 1'b0;
                state <= S_DW_RUN;
            end

            if (state == S_DW_RUN) begin
                if (!dw_active) begin
                    dw_start_pulse <= 1'b1;
                    dw_active <= 1'b1;
                    dw_rd_done_seen <= 1'b0;
                    dw_wr_done_seen <= 1'b0;
                end

                if (dw_reader_done) begin
                    dw_rd_done_seen <= 1'b1;
                end
                if (dw_writer_done) begin
                    dw_wr_done_seen <= 1'b1;
                end

                if (dw_active && dw_rd_done_seen && dw_wr_done_seen) begin
                    dw_active <= 1'b0;
                    if (dw_ch_idx == in_channels_reg - 1'b1) begin
                        pw_out_ch_idx <= '0;
                        pw_group_curr <= '0;
                        pw_group_wait <= 1'b0;
                        state <= S_PW_LOAD;
                    end else begin
                        dw_ch_idx <= dw_ch_idx + 1'b1;
                    end
                end
            end

            if (state == S_PW_LOAD) begin
                if (pw_group_ready) begin
                    state <= S_PW_RUN;
                    pw_group_wait <= 1'b0;
                end else begin
                    pw_group_wait <= 1'b1;
                end
            end

            if (state == S_PW_RUN) begin
                if (!pw_active) begin
                    pw_start_pulse <= 1'b1;
                    pw_active <= 1'b1;
                    pw_rd_done_seen <= 1'b0;
                    pw_wr_done_seen <= 1'b0;
                end

                if (pw_reader_done) begin
                    pw_rd_done_seen <= 1'b1;
                end
                if (pw_writer_done) begin
                    pw_wr_done_seen <= 1'b1;
                end

                if (pw_active && pw_rd_done_seen && pw_wr_done_seen) begin
                    pw_active <= 1'b0;
                    if (pw_out_ch_idx == out_channels_reg - 1'b1) begin
                        state <= S_DONE;
                    end else begin
                        pw_out_ch_idx <= pw_out_ch_idx + 1'b1;
                        if (((pw_out_ch_idx + 1'b1) % PW_GROUP) == 0) begin
                            pw_group_curr <= pw_group_curr + 1'b1;
                            state <= S_PW_LOAD;
                        end
                    end
                end
            end

            if (state == S_DONE) begin
                done <= 1'b1;
                dw_active <= 1'b0;
                pw_active <= 1'b0;
                state <= S_IDLE;
            end
        end
    end

    assign pw_group_idx = pw_group_curr;
    assign pw_group_req = (state == S_PW_LOAD) && !pw_group_ready;

    assign dbg_pw_en = $test$plusargs("DBG_PW_GROUP");

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
        end else if (dbg_pw_en) begin
            if (state == S_PW_LOAD && pw_group_ready) begin
                $display("PW_LOAD ready grp=%0d out_ch=%0d in_c=%0d out_c=%0d",
                         pw_group_curr, pw_out_ch_idx, in_channels_reg, out_channels_reg);
            end
            if (state == S_PW_LOAD && !pw_group_ready) begin
                $display("PW_LOAD wait grp=%0d out_ch=%0d in_c=%0d out_c=%0d",
                         pw_group_curr, pw_out_ch_idx, in_channels_reg, out_channels_reg);
            end
            if (state == S_PW_RUN && pw_start_pulse) begin
                $display("PW_RUN start grp=%0d out_ch=%0d", pw_group_curr, pw_out_ch_idx);
            end
            if (state == S_PW_RUN && pw_active && pw_rd_done_seen && pw_wr_done_seen) begin
                $display("PW_RUN done grp=%0d out_ch=%0d", pw_group_curr, pw_out_ch_idx);
            end
        end
    end
endmodule
