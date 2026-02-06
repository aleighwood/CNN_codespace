module dws_tile_runner #(
    parameter int DATA_W = 8,
    parameter int ACC_W = 32,
    parameter int MUL_W = 32,
    parameter int BIAS_W = 32,
    parameter int SHIFT_W = 6,
    parameter int ADDR_W = 32,
    parameter int DIM_W = 16,
    parameter int MAX_TILE_IN_W = 33,
    parameter int MAX_TILE_IN_H = 33,
    parameter int MAX_TILE_OUT_W = 16,
    parameter int MAX_TILE_OUT_H = 16,
    parameter int PW_GROUP = 16,
    parameter int PW_OC_PAR = 16,
    parameter int PW_IC_PAR = 8
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
    input  logic signed [DATA_W-1:0] cfg_pad_value,

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

    output logic [PW_IC_PAR-1:0] dw_buf_rd_en,
    output logic [PW_IC_PAR*ADDR_W-1:0] dw_buf_rd_addr_vec,
    input  logic [PW_IC_PAR*DATA_W-1:0] dw_buf_rd_data_vec,

    output logic out_wr_en0,
    output logic [ADDR_W-1:0] out_wr_addr0,
    output logic [DATA_W-1:0] out_wr_data0,
    output logic out_wr_en1,
    output logic [ADDR_W-1:0] out_wr_addr1,
    output logic [DATA_W-1:0] out_wr_data1,

    input  logic signed [DATA_W*9-1:0] dw_weight_flat,
    input  logic signed [MUL_W-1:0] dw_mul,
    input  logic signed [BIAS_W-1:0] dw_bias,
    input  logic [SHIFT_W-1:0] dw_shift,
    input  logic signed [DATA_W-1:0] dw_relu6_max,
    input  logic signed [DATA_W-1:0] dw_relu6_min,

    input  logic signed [PW_OC_PAR*PW_IC_PAR*DATA_W-1:0] pw_weight_vec,
    input  logic signed [PW_OC_PAR*ACC_W-1:0] pw_bias_acc_vec,
    input  logic signed [PW_OC_PAR*MUL_W-1:0] pw_mul_vec,
    input  logic [PW_OC_PAR*SHIFT_W-1:0] pw_shift_vec,
    input  logic signed [PW_OC_PAR*DATA_W-1:0] pw_relu6_max_vec,
    input  logic signed [PW_OC_PAR*DATA_W-1:0] pw_relu6_min_vec,

    output logic [DIM_W-1:0] dw_ch_idx,
    output logic [DIM_W-1:0] pw_in_ch_idx,
    output logic [DIM_W-1:0] pw_out_ch_idx
);
    localparam int IN_ROW_W = (MAX_TILE_IN_H <= 1) ? 1 : $clog2(MAX_TILE_IN_H);
    localparam int IN_COL_W = (MAX_TILE_IN_W <= 1) ? 1 : $clog2(MAX_TILE_IN_W);
    localparam int OUT_PIX_MAX = MAX_TILE_OUT_H * MAX_TILE_OUT_W;
    localparam int OUT_IDX_W = (OUT_PIX_MAX <= 1) ? 1 : $clog2(OUT_PIX_MAX + 1);

    typedef enum logic [2:0] {
        S_IDLE,
        S_DW_RUN,
        S_PW_LOAD,
        S_PW_RUN,
        S_PW_WRITE,
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
    logic signed [DATA_W-1:0] pad_value_reg;
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
    logic [PW_IC_PAR*DATA_W-1:0] pw_in_data_vec;
    logic pw_in_first;
    logic pw_in_last;

    logic pw_acc_valid;
    logic pw_acc_ready;
    logic signed [PW_OC_PAR*ACC_W-1:0] pw_acc_vec;

    logic [PW_OC_PAR-1:0] pw_rq_valid_vec;
    logic [PW_OC_PAR-1:0] pw_rq_in_ready_vec;
    logic signed [PW_OC_PAR*DATA_W-1:0] pw_q_vec;
    logic pw_q_valid;

    logic [OUT_IDX_W-1:0] pw_pix_idx;
    logic pw_pix_done_seen;
    logic [DIM_W-1:0] pw_write_oc;
    logic [OUT_IDX_W-1:0] pw_write_idx;
    logic pw_writer_active;
    logic pw_writer_start;
    logic pw_write_valid;
    logic pw_write_ready;
    logic [DATA_W-1:0] pw_write_data;
    logic [DATA_W-1:0] pw_write_data1;
    logic pw_write_two;
    logic [DIM_W-1:0] pw_oc_limit;

    logic signed [DATA_W-1:0] pw_out_buf [0:PW_OC_PAR-1][0:OUT_PIX_MAX-1];

    integer oc;

    assign busy = (state != S_IDLE);

    always_comb begin
        in_plane_stride = in_img_h_reg * in_img_w_reg;
        out_plane_stride = out_img_h_reg * out_img_w_reg;
        tile_area = tile_out_h_reg * tile_out_w_reg;

        in_base_ch = in_base_addr_reg + (dw_ch_idx * in_plane_stride);
        dw_buf_base_ch = dw_buf_base_addr_reg + (dw_ch_idx * tile_area);
        out_base_ch = out_base_addr_reg + ((pw_out_ch_idx + pw_write_oc) * out_plane_stride);

        if (out_channels_reg > pw_out_ch_idx) begin
            if ((out_channels_reg - pw_out_ch_idx) >= PW_OC_PAR) begin
                pw_oc_limit = PW_OC_PAR[DIM_W-1:0];
            end else begin
                pw_oc_limit = out_channels_reg - pw_out_ch_idx;
            end
        end else begin
            pw_oc_limit = '0;
        end
    end

    logic [OUT_IDX_W-1:0] pw_pix_total;
    logic pw_pix_last;
    assign pw_pix_total = tile_out_h_reg * tile_out_w_reg;
    assign pw_pix_last = (pw_pix_idx == pw_pix_total - 1'b1);

    assign pw_write_valid = (state == S_PW_WRITE) && pw_writer_active;
    assign pw_write_data = pw_out_buf[pw_write_oc][pw_write_idx];
    assign pw_write_two = ((pw_write_idx + 1'b1) < pw_pix_total);
    assign pw_write_data1 = pw_out_buf[pw_write_oc][pw_write_idx + 1'b1];

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
        .cfg_pad_value(pad_value_reg),
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
        .dw_bias_acc(dw_bias),
        .dw_shift(dw_shift),
        .dw_relu6_max(dw_relu6_max),
        .dw_relu6_min(dw_relu6_min),
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

    pw_tile_reader_vec #(
        .DATA_W(DATA_W),
        .ADDR_W(ADDR_W),
        .DIM_W(DIM_W),
        .IC_PAR(PW_IC_PAR)
    ) u_pw_tile_reader (
        .clk(clk),
        .rst_n(rst_n),
        .start(pw_start_pulse),
        .cfg_tile_h(tile_out_h_reg),
        .cfg_tile_w(tile_out_w_reg),
        .cfg_channels(in_channels_reg),
        .cfg_base_addr(dw_buf_base_addr_reg),
        .rd_en(dw_buf_rd_en),
        .rd_addr_vec(dw_buf_rd_addr_vec),
        .rd_data_vec(dw_buf_rd_data_vec),
        .out_valid(pw_in_valid),
        .out_ready(pw_in_ready),
        .out_data_vec(pw_in_data_vec),
        .out_first_ch(pw_in_first),
        .out_last_ch(pw_in_last),
        .out_in_ch_idx(pw_in_ch_idx),
        .done(pw_reader_done)
    );

    pw_conv_1x1_vec #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .OC_PAR(PW_OC_PAR),
        .IC_PAR(PW_IC_PAR)
    ) u_pw_conv (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(pw_in_valid),
        .in_ready(pw_in_ready),
        .in_data_vec(pw_in_data_vec),
        .weight_vec(pw_weight_vec),
        .bias_vec(pw_bias_acc_vec),
        .first_in_ch(pw_in_first),
        .last_in_ch(pw_in_last),
        .out_valid(pw_acc_valid),
        .out_ready(pw_acc_ready),
        .out_acc_vec(pw_acc_vec)
    );

    assign pw_acc_ready = 1'b1;

    genvar rq_i;
    generate
        for (rq_i = 0; rq_i < PW_OC_PAR; rq_i = rq_i + 1) begin : gen_pw_rq
            requant_q31 #(
                .DATA_W(DATA_W),
                .ACC_W(ACC_W),
                .MUL_W(MUL_W),
                .SHIFT_W(SHIFT_W)
            ) u_pw_requant (
                .clk(clk),
                .rst_n(rst_n),
                .in_valid(pw_acc_valid),
                .in_ready(pw_rq_in_ready_vec[rq_i]),
                .in_acc(pw_acc_vec[rq_i*ACC_W +: ACC_W]),
                .mul_q31(pw_mul_vec[rq_i*MUL_W +: MUL_W]),
                .shift(pw_shift_vec[rq_i*SHIFT_W +: SHIFT_W]),
                .zp_out(pw_relu6_min_vec[rq_i*DATA_W +: DATA_W]),
                .relu6_max(pw_relu6_max_vec[rq_i*DATA_W +: DATA_W]),
                .relu6_en(1'b1),
                .out_valid(pw_rq_valid_vec[rq_i]),
                .out_ready(1'b1),
                .out_q(pw_q_vec[rq_i*DATA_W +: DATA_W])
            );
        end
    endgenerate

    assign pw_q_valid = pw_rq_valid_vec[0];

    tile_writer_2x #(
        .DATA_W(DATA_W),
        .ADDR_W(ADDR_W),
        .DIM_W(DIM_W)
    ) u_out_tile_writer (
        .clk(clk),
        .rst_n(rst_n),
        .start(pw_writer_start),
        .cfg_img_h(out_img_h_reg),
        .cfg_img_w(out_img_w_reg),
        .cfg_base_addr(out_base_ch),
        .cfg_tile_out_row(tile_out_row_reg),
        .cfg_tile_out_col(tile_out_col_reg),
        .cfg_tile_out_h(tile_out_h_reg),
        .cfg_tile_out_w(tile_out_w_reg),
        .in_valid(pw_write_valid),
        .in_two(pw_write_two),
        .in_ready(pw_write_ready),
        .in_data0(pw_write_data),
        .in_data1(pw_write_data1),
        .wr_en0(out_wr_en0),
        .wr_addr0(out_wr_addr0),
        .wr_data0(out_wr_data0),
        .wr_en1(out_wr_en1),
        .wr_addr1(out_wr_addr1),
        .wr_data1(out_wr_data1),
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
            pw_group_curr <= '0;
            pw_group_wait <= 1'b0;
            pw_pix_idx <= '0;
            pw_pix_done_seen <= 1'b0;
            pw_write_oc <= '0;
            pw_write_idx <= '0;
            pw_writer_active <= 1'b0;
            pw_writer_start <= 1'b0;
        end else begin
            done <= 1'b0;
            dw_start_pulse <= 1'b0;
            pw_start_pulse <= 1'b0;
            pw_writer_start <= 1'b0;

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
                pad_value_reg <= cfg_pad_value;
                in_base_addr_reg <= cfg_in_base_addr;
                out_base_addr_reg <= cfg_out_base_addr;
                dw_buf_base_addr_reg <= cfg_dw_buf_base_addr;

                dw_ch_idx <= '0;
                pw_out_ch_idx <= '0;
                dw_rd_done_seen <= 1'b0;
                dw_wr_done_seen <= 1'b0;
                pw_rd_done_seen <= 1'b0;
                dw_active <= 1'b0;
                pw_active <= 1'b0;
                pw_group_curr <= '0;
                pw_group_wait <= 1'b0;
                pw_pix_idx <= '0;
                pw_pix_done_seen <= 1'b0;
                pw_write_oc <= '0;
                pw_write_idx <= '0;
                pw_writer_active <= 1'b0;
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
                    pw_pix_idx <= '0;
                    pw_pix_done_seen <= 1'b0;
                end

                if (pw_reader_done) begin
                    pw_rd_done_seen <= 1'b1;
                end

                if (pw_q_valid) begin
                    for (oc = 0; oc < PW_OC_PAR; oc = oc + 1) begin
                        if (oc < pw_oc_limit) begin
                            pw_out_buf[oc][pw_pix_idx] <= pw_q_vec[oc*DATA_W +: DATA_W];
                        end
                    end
                    if (pw_pix_last) begin
                        pw_pix_done_seen <= 1'b1;
                    end else begin
                        pw_pix_idx <= pw_pix_idx + 1'b1;
                    end
                end

                if (pw_active && pw_rd_done_seen && pw_pix_done_seen) begin
                    pw_active <= 1'b0;
                    pw_writer_active <= 1'b0;
                    pw_write_oc <= '0;
                    pw_write_idx <= '0;
                    state <= S_PW_WRITE;
                end
            end

            if (state == S_PW_WRITE) begin
                if (!pw_writer_active) begin
                    pw_writer_start <= 1'b1;
                    pw_writer_active <= 1'b1;
                    pw_write_idx <= '0;
                end else if (pw_write_valid && pw_write_ready) begin
                    if (pw_write_two) begin
                        pw_write_idx <= pw_write_idx + 2'd2;
                    end else begin
                        pw_write_idx <= pw_write_idx + 1'b1;
                    end
                end

                if (pw_writer_done) begin
                    pw_writer_active <= 1'b0;
                    if (pw_write_oc + 1'b1 < pw_oc_limit) begin
                        pw_write_oc <= pw_write_oc + 1'b1;
                    end else begin
                        if (pw_out_ch_idx + pw_oc_limit >= out_channels_reg) begin
                            state <= S_DONE;
                        end else begin
                            pw_out_ch_idx <= pw_out_ch_idx + pw_oc_limit;
                            if (((pw_out_ch_idx + pw_oc_limit) % PW_GROUP) == 0) begin
                                pw_group_curr <= pw_group_curr + 1'b1;
                                state <= S_PW_LOAD;
                            end else begin
                                state <= S_PW_RUN;
                            end
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
    // Assert request during DW to prefetch group 0, and during PW_LOAD to wait for cache.
    // This avoids a combinational ready/req loop while overlapping weight load with DW.
    assign pw_group_req = (state == S_DW_RUN) || (state == S_PW_LOAD);

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
            if (state == S_PW_RUN && pw_active && pw_rd_done_seen && pw_pix_done_seen) begin
                $display("PW_RUN done grp=%0d out_ch=%0d", pw_group_curr, pw_out_ch_idx);
            end
        end
    end
endmodule
