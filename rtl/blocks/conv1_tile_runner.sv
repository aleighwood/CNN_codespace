module conv1_tile_runner #(
    parameter int DATA_W = 8,
    parameter int ACC_W = 32,
    parameter int MUL_W = 16,
    parameter int BIAS_W = 32,
    parameter int SHIFT_W = 6,
    parameter int ADDR_W = 32,
    parameter int DIM_W = 16,
    parameter int MAX_TILE_IN_W = 33,
    parameter int MAX_TILE_IN_H = 33,
    parameter int MAX_TILE_OUT_W = 16,
    parameter int MAX_TILE_OUT_H = 16,
    parameter int OC_PAR = 4,
    parameter int STRIDE = 2
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

    input  logic [ADDR_W-1:0] cfg_in_base_addr,
    input  logic [ADDR_W-1:0] cfg_out_base_addr,

    output logic in_rd_en,
    output logic [ADDR_W-1:0] in_rd_addr,
    input  logic [DATA_W-1:0] in_rd_data,

    output logic out_wr_en,
    output logic [ADDR_W-1:0] out_wr_addr,
    output logic [DATA_W-1:0] out_wr_data,

    input  logic signed [OC_PAR*DATA_W*9-1:0] weight_flat_vec,
    input  logic signed [OC_PAR*ACC_W-1:0] bias_acc_vec,
    input  logic signed [OC_PAR*MUL_W-1:0] mul_vec,
    input  logic signed [OC_PAR*BIAS_W-1:0] bias_requant_vec,
    input  logic [OC_PAR*SHIFT_W-1:0] shift_vec,
    input  logic signed [OC_PAR*DATA_W-1:0] relu6_max_vec,

    output logic [DIM_W-1:0] ic_idx,
    output logic [DIM_W-1:0] oc_group_idx
);
    localparam int IN_ROW_W = (MAX_TILE_IN_H <= 1) ? 1 : $clog2(MAX_TILE_IN_H);
    localparam int IN_COL_W = (MAX_TILE_IN_W <= 1) ? 1 : $clog2(MAX_TILE_IN_W);
    localparam int OUT_PIX_MAX = MAX_TILE_OUT_H * MAX_TILE_OUT_W;
    localparam int OUT_IDX_W = (OUT_PIX_MAX <= 1) ? 1 : $clog2(OUT_PIX_MAX);

    typedef enum logic [2:0] {
        S_IDLE,
        S_INIT_PSUM,
        S_ACCUM,
        S_OUTPUT,
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
    logic [DIM_W-1:0] stride_reg;
    logic [ADDR_W-1:0] in_base_addr_reg;
    logic [ADDR_W-1:0] out_base_addr_reg;

    logic [OUT_IDX_W-1:0] psum_init_idx;
    logic [OUT_IDX_W-1:0] win_count;
    logic win_done;
    logic rd_done_seen;

    logic line_start;
    logic reader_start;

    logic lb_valid;
    logic lb_ready;
    logic signed [DATA_W*9-1:0] lb_window;
    logic [IN_ROW_W-1:0] lb_row;
    logic [IN_COL_W-1:0] lb_col;

    logic mac_valid;
    logic mac_ready;
    logic signed [OC_PAR*ACC_W-1:0] mac_acc_vec;


    logic reader_done;
    logic tr_valid;
    logic tr_ready;
    logic [DATA_W-1:0] tr_data;

    logic [ADDR_W-1:0] in_plane_stride;
    logic [ADDR_W-1:0] out_plane_stride;
    logic [ADDR_W-1:0] in_base_ch;
    logic [ADDR_W-1:0] out_base_ch;

    logic signed [ACC_W-1:0] psum_mem [0:OC_PAR-1][0:OUT_PIX_MAX-1];

    logic [OUT_IDX_W-1:0] out_stream_idx;
    logic out_stream_valid;
    logic [OUT_IDX_W-1:0] out_idx_pipe;
    logic out_idx_pipe_valid;
    logic [OUT_IDX_W-1:0] out_pix_count;
    logic [3:0] dbg_win_seen;
    logic [3:0] dbg_lb_seen;
    logic [3:0] dbg_wr_seen;

    logic [OUT_IDX_W-1:0] idx_pipe;
    logic idx_pipe_valid;
    logic [OUT_IDX_W-1:0] idx_calc;
    logic [OUT_IDX_W-1:0] idx_base;
    logic [DIM_W:0] out_pix_total;
    logic [DIM_W:0] out_pix_last;
    logic dbg_idx_en;
    logic out_stream_ready;

    logic [DIM_W-1:0] oc_idx_in_group;
    logic [DIM_W-1:0] oc_global_idx;

    logic signed [ACC_W-1:0] psum_read;
    logic [DIM_W-1:0] oc_out_idx;

    logic rq_valid;
    logic rq_in_ready;
    logic rq_out_ready;
    logic [DATA_W-1:0] rq_data;

    logic writer_start;

    assign busy = (state != S_IDLE);

    always_comb begin
        in_plane_stride = in_img_h_reg * in_img_w_reg;
        out_plane_stride = out_img_h_reg * out_img_w_reg;
        in_base_ch = in_base_addr_reg + (ic_idx * in_plane_stride);
        oc_global_idx = oc_group_idx * OC_PAR + oc_idx_in_group;
        out_base_ch = out_base_addr_reg + (oc_global_idx * out_plane_stride);
    end

    tile_reader #(
        .DATA_W(DATA_W),
        .ADDR_W(ADDR_W),
        .DIM_W(DIM_W)
    ) u_tile_reader (
        .clk(clk),
        .rst_n(rst_n),
        .start(reader_start),
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
        .out_valid(tr_valid),
        .out_ready(tr_ready),
        .out_data(tr_data),
        .done(reader_done)
    );

    line_buffer_3x3 #(
        .DATA_W(DATA_W),
        .MAX_IMG_W(MAX_TILE_IN_W),
        .MAX_IMG_H(MAX_TILE_IN_H)
    ) u_line_buffer (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(tr_valid),
        .in_ready(tr_ready),
        .in_data(tr_data),
        .start(line_start),
        .cfg_img_h(tile_in_h_reg[IN_ROW_W-1:0]),
        .cfg_img_w(tile_in_w_reg[IN_COL_W-1:0]),
        .cfg_stride(stride_reg[IN_ROW_W-1:0]),
        .out_valid(lb_valid),
        .out_ready(lb_ready),
        .window_flat(lb_window),
        .out_row(lb_row),
        .out_col(lb_col)
    );

    conv3x3_mac_vec #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .OC_PAR(OC_PAR)
    ) u_mac_vec (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(lb_valid),
        .in_ready(lb_ready),
        .window_flat(lb_window),
        .weight_flat_vec(weight_flat_vec),
        .out_valid(mac_valid),
        .out_ready(mac_ready),
        .out_acc_vec(mac_acc_vec)
    );

    assign mac_ready = 1'b1;

    assign out_pix_total = tile_out_h_reg * tile_out_w_reg;
    assign out_pix_last = out_pix_total - 1'b1;
    assign dbg_idx_en = $test$plusargs("DBG_CONV1_IDX");

    always_comb begin
        if (stride_reg == 2) begin
            idx_base = (lb_row >> 1) * tile_out_w_reg + (lb_col >> 1);
        end else begin
            idx_base = lb_row * tile_out_w_reg + lb_col;
        end
        idx_calc = idx_base;
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            idx_pipe <= '0;
            idx_pipe_valid <= 1'b0;
        end else begin
            if (lb_valid && lb_ready) begin
                idx_pipe <= idx_calc;
                idx_pipe_valid <= 1'b1;
            end else begin
                idx_pipe_valid <= 1'b0;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            dbg_win_seen <= '0;
            dbg_lb_seen <= '0;
        end else if (state != S_ACCUM || ic_idx != 0) begin
            dbg_win_seen <= '0;
            dbg_lb_seen <= '0;
        end else begin
            if (mac_valid && idx_pipe_valid) begin
                if (dbg_win_seen != 4'hf) begin
                    dbg_win_seen <= dbg_win_seen + 1'b1;
                end
            end
            if (lb_valid && lb_ready) begin
                if (dbg_lb_seen != 4'hf) begin
                    dbg_lb_seen <= dbg_lb_seen + 1'b1;
                end
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
        end else if (dbg_idx_en) begin
            if (state != S_OUTPUT) begin
                dbg_wr_seen <= '0;
            end
            if (state == S_ACCUM && mac_valid && idx_pipe_valid && (ic_idx == 0)) begin
                if ((dbg_win_seen < 8) ||
                    (out_pix_total > 8 && (win_count >= (out_pix_total - 8)))) begin
                    $display("CONV1_IDX ic=%0d win=%0d lb_row=%0d lb_col=%0d idx_base=%0d idx_calc=%0d idx_pipe=%0d out_pix=%0d",
                             ic_idx, win_count, lb_row, lb_col, idx_base, idx_calc, idx_pipe, out_pix_total);
                end
            end
            if (state == S_ACCUM && lb_valid && lb_ready && (ic_idx == 0) && (dbg_lb_seen < 8)) begin
                $display("CONV1_LB ic=%0d lb_row=%0d lb_col=%0d idx_base=%0d idx_calc=%0d out_pix=%0d",
                         ic_idx, lb_row, lb_col, idx_base, idx_calc, out_pix_total);
            end
            if (state == S_OUTPUT && rq_valid && rq_out_ready && out_idx_pipe_valid &&
                (oc_group_idx == 0) && (oc_idx_in_group == 0)) begin
                if ((out_pix_count < 8) ||
                    (out_pix_total > 8 && (out_pix_count >= (out_pix_total - 8)))) begin
                    $display("CONV1_OUT oc=%0d pix=%0d out_idx=%0d data=%0d",
                             oc_idx_in_group, out_pix_count, out_idx_pipe, $signed(rq_data));
                end
            end
            if (state == S_OUTPUT && out_wr_en && (dbg_wr_seen < 8)) begin
                $display("CONV1_WR oc=%0d pix=%0d out_idx=%0d addr=%0d data=%0d",
                         oc_idx_in_group, out_pix_count, out_idx_pipe,
                         out_wr_addr, $signed(out_wr_data));
                dbg_wr_seen <= dbg_wr_seen + 1'b1;
            end
        end
    end

    integer oc;
    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (oc = 0; oc < OC_PAR; oc = oc + 1) begin
                for (int i = 0; i < OUT_PIX_MAX; i = i + 1) begin
                    psum_mem[oc][i] = '0;
                end
            end
        end else begin
            if (state == S_INIT_PSUM) begin
                for (oc = 0; oc < OC_PAR; oc = oc + 1) begin
                    psum_mem[oc][psum_init_idx] <= bias_acc_vec[oc*ACC_W +: ACC_W];
                end
            end else if (state == S_ACCUM && mac_valid && idx_pipe_valid) begin
                for (oc = 0; oc < OC_PAR; oc = oc + 1) begin
                    psum_mem[oc][idx_pipe] <= psum_mem[oc][idx_pipe] +
                                              mac_acc_vec[oc*ACC_W +: ACC_W];
                end
            end
        end
    end

    assign out_stream_ready = rq_in_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_idx_pipe <= '0;
            out_idx_pipe_valid <= 1'b0;
        end else begin
            if (out_stream_valid && out_stream_ready) begin
                out_idx_pipe <= out_stream_idx;
                out_idx_pipe_valid <= 1'b1;
            end else if (rq_valid) begin
                out_idx_pipe_valid <= 1'b0;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_pix_count <= '0;
        end else if (state != S_OUTPUT) begin
            out_pix_count <= '0;
        end else if (rq_valid && rq_out_ready) begin
            if (out_pix_total != 0 && out_pix_count == out_pix_total - 1'b1) begin
                out_pix_count <= '0;
            end else begin
                out_pix_count <= out_pix_count + 1'b1;
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_stream_idx <= '0;
            out_stream_valid <= 1'b0;
            oc_idx_in_group <= '0;
            writer_start <= 1'b0;
        end else begin
            writer_start <= 1'b0;
            if (state == S_OUTPUT) begin
                if (!out_stream_valid && out_stream_ready) begin
                    out_stream_valid <= 1'b1;
                    out_stream_idx <= '0;
                    writer_start <= 1'b1;
                end

                if (out_stream_valid && out_stream_ready) begin
                    if (out_stream_idx == out_pix_last) begin
                        out_stream_idx <= '0;
                        out_stream_valid <= 1'b0;
                        if (oc_idx_in_group == OC_PAR - 1) begin
                            oc_idx_in_group <= '0;
                        end else begin
                            oc_idx_in_group <= oc_idx_in_group + 1'b1;
                        end
                    end else begin
                        out_stream_idx <= out_stream_idx + 1'b1;
                    end
                end
            end else begin
                out_stream_valid <= 1'b0;
                out_stream_idx <= '0;
                oc_idx_in_group <= '0;
                writer_start <= 1'b0;
            end
        end
    end

    always_comb begin
        psum_read = psum_mem[oc_idx_in_group][out_stream_idx];
    end
    assign oc_out_idx = oc_idx_in_group;

    requant_relu6 #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .MUL_W(MUL_W),
        .BIAS_W(BIAS_W),
        .SHIFT_W(SHIFT_W)
    ) u_requant (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(out_stream_valid),
        .in_ready(rq_in_ready),
        .in_acc(psum_read),
        .mul(mul_vec[oc_out_idx*MUL_W +: MUL_W]),
        .bias(bias_requant_vec[oc_out_idx*BIAS_W +: BIAS_W]),
        .shift(shift_vec[oc_out_idx*SHIFT_W +: SHIFT_W]),
        .relu6_max(relu6_max_vec[oc_out_idx*DATA_W +: DATA_W]),
        .relu6_en(1'b1),
        .out_valid(rq_valid),
        .out_ready(rq_out_ready),
        .out_q(rq_data)
    );

    tile_writer #(
        .DATA_W(DATA_W),
        .ADDR_W(ADDR_W),
        .DIM_W(DIM_W)
    ) u_out_writer (
        .clk(clk),
        .rst_n(rst_n),
        .start(writer_start),
        .cfg_img_h(out_img_h_reg),
        .cfg_img_w(out_img_w_reg),
        .cfg_base_addr(out_base_ch),
        .cfg_tile_out_row(tile_out_row_reg),
        .cfg_tile_out_col(tile_out_col_reg),
        .cfg_tile_out_h(tile_out_h_reg),
        .cfg_tile_out_w(tile_out_w_reg),
        .in_valid(rq_valid),
        .in_ready(rq_out_ready),
        .in_data(rq_data),
        .wr_en(out_wr_en),
        .wr_addr(out_wr_addr),
        .wr_data(out_wr_data),
        .done()
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 1'b0;
            ic_idx <= '0;
            oc_group_idx <= '0;
            psum_init_idx <= '0;
            win_count <= '0;
            win_done <= 1'b0;
            rd_done_seen <= 1'b0;
            reader_start <= 1'b0;
            line_start <= 1'b0;
        end else begin
            done <= 1'b0;
            reader_start <= 1'b0;
            line_start <= 1'b0;

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
                stride_reg <= cfg_stride;
                in_base_addr_reg <= cfg_in_base_addr;
                out_base_addr_reg <= cfg_out_base_addr;

                ic_idx <= '0;
                oc_group_idx <= '0;
                psum_init_idx <= '0;
                win_count <= '0;
                win_done <= 1'b0;
                rd_done_seen <= 1'b0;
                state <= S_INIT_PSUM;
                if (dbg_idx_en) begin
                    $display("CONV1_CFG tile_in_row=%0d tile_in_col=%0d tile_in_h=%0d tile_in_w=%0d tile_out_row=%0d tile_out_col=%0d tile_out_h=%0d tile_out_w=%0d",
                             cfg_tile_in_row, cfg_tile_in_col, cfg_tile_in_h, cfg_tile_in_w,
                             cfg_tile_out_row, cfg_tile_out_col, cfg_tile_out_h, cfg_tile_out_w);
                end
            end

            if (state == S_INIT_PSUM) begin
                if (psum_init_idx == (tile_out_h_reg * tile_out_w_reg) - 1'b1) begin
                    psum_init_idx <= '0;
                    win_count <= '0;
                    win_done <= 1'b0;
                    rd_done_seen <= 1'b0;
                    reader_start <= 1'b1;
                    line_start <= 1'b1;
                    state <= S_ACCUM;
                end else begin
                    psum_init_idx <= psum_init_idx + 1'b1;
                end
            end

            if (state == S_ACCUM) begin
                if (reader_done) begin
                    rd_done_seen <= 1'b1;
                end

                if (mac_valid && idx_pipe_valid) begin
                    if (win_count == (tile_out_h_reg * tile_out_w_reg) - 1'b1) begin
                        win_count <= '0;
                        win_done <= 1'b1;
                    end else begin
                        win_count <= win_count + 1'b1;
                    end
                end

                if (win_done && rd_done_seen) begin
                    win_done <= 1'b0;
                    rd_done_seen <= 1'b0;
                    if (ic_idx == in_channels_reg - 1'b1) begin
                        state <= S_OUTPUT;
                    end else begin
                        ic_idx <= ic_idx + 1'b1;
                        reader_start <= 1'b1;
                        line_start <= 1'b1;
                    end
                end
            end

            if (state == S_OUTPUT) begin
                if (oc_idx_in_group == OC_PAR - 1 &&
                    out_stream_idx == out_pix_last &&
                    out_stream_valid && out_stream_ready) begin
                    if ((oc_group_idx + 1) * OC_PAR >= out_channels_reg) begin
                        state <= S_DONE;
                    end else begin
                        oc_group_idx <= oc_group_idx + 1'b1;
                        ic_idx <= '0;
                        state <= S_INIT_PSUM;
                    end
                end
            end

            if (state == S_DONE) begin
                done <= 1'b1;
                state <= S_IDLE;
            end
        end
    end
endmodule
