module mobilenet_v1_ctrl #(
    parameter int DIM_W = 16,
    parameter int ADDR_W = 32,
    parameter int TILE_H = 16,
    parameter int TILE_W = 16
) (
    input  logic clk,
    input  logic rst_n,

    input  logic start,
    output logic busy,
    output logic done,

    input  logic [DIM_W-1:0] cfg_in_img_h,
    input  logic [DIM_W-1:0] cfg_in_img_w,

    input  logic [ADDR_W-1:0] cfg_fm_base0,
    input  logic [ADDR_W-1:0] cfg_fm_base1,
    input  logic [ADDR_W-1:0] cfg_dw_buf_base,

    output logic layer_is_conv1,
    output logic [DIM_W-1:0] layer_idx,

    output logic [DIM_W-1:0] cur_in_h,
    output logic [DIM_W-1:0] cur_in_w,
    output logic [DIM_W-1:0] cur_out_h,
    output logic [DIM_W-1:0] cur_out_w,
    output logic [DIM_W-1:0] cur_in_c,
    output logic [DIM_W-1:0] cur_out_c,
    output logic [DIM_W-1:0] cur_stride,

    output logic signed [DIM_W:0] tile_in_row,
    output logic signed [DIM_W:0] tile_in_col,
    output logic [DIM_W-1:0] tile_in_h,
    output logic [DIM_W-1:0] tile_in_w,
    output logic [DIM_W-1:0] tile_out_row,
    output logic [DIM_W-1:0] tile_out_col,
    output logic [DIM_W-1:0] tile_out_h,
    output logic [DIM_W-1:0] tile_out_w,

    output logic [ADDR_W-1:0] in_base_addr,
    output logic [ADDR_W-1:0] out_base_addr,
    output logic [ADDR_W-1:0] dw_buf_base_addr,

    output logic conv1_start,
    input  logic conv1_busy,
    input  logic conv1_done,

    output logic dws_start,
    input  logic dws_busy,
    input  logic dws_done
);
    localparam int NUM_LAYERS = 14;
    localparam int KERNEL = 3;
    localparam int PAD = 1;

    typedef enum logic [2:0] {
        S_IDLE,
        S_CFG,
        S_START,
        S_TILE,
        S_NEXT,
        S_DONE
    } state_t;

    state_t state;

    logic tile_cfg_valid;
    logic tile_cfg_ready;
    logic tile_start;
    logic tile_valid;
    logic tile_ready;
    logic tile_done;

    logic [DIM_W-1:0] stride_reg;
    logic [DIM_W-1:0] next_out_c;

    logic tile_active;
    logic tile_wait;

    logic base_sel;

    logic runner_busy;
    logic runner_done;

    assign layer_is_conv1 = (layer_idx == 0);

    always_comb begin
        next_out_c = 32;
        stride_reg = 2;
        case (layer_idx)
            0: begin
                next_out_c = 32;
                stride_reg = 2;
            end
            1: begin next_out_c = 64; stride_reg = 1; end
            2: begin next_out_c = 128; stride_reg = 2; end
            3: begin next_out_c = 128; stride_reg = 1; end
            4: begin next_out_c = 256; stride_reg = 2; end
            5: begin next_out_c = 256; stride_reg = 1; end
            6: begin next_out_c = 512; stride_reg = 2; end
            7: begin next_out_c = 512; stride_reg = 1; end
            8: begin next_out_c = 512; stride_reg = 1; end
            9: begin next_out_c = 512; stride_reg = 1; end
            10: begin next_out_c = 512; stride_reg = 1; end
            11: begin next_out_c = 512; stride_reg = 1; end
            12: begin next_out_c = 1024; stride_reg = 2; end
            13: begin next_out_c = 1024; stride_reg = 1; end
            default: begin next_out_c = 1024; stride_reg = 1; end
        endcase
    end

    assign runner_busy = layer_is_conv1 ? conv1_busy : dws_busy;
    assign runner_done = layer_is_conv1 ? conv1_done : dws_done;

    assign cur_stride = stride_reg;

    always_comb begin
        in_base_addr = base_sel ? cfg_fm_base1 : cfg_fm_base0;
        out_base_addr = base_sel ? cfg_fm_base0 : cfg_fm_base1;
        dw_buf_base_addr = cfg_dw_buf_base;
    end

    tile_ctrl #(
        .DIM_W(DIM_W)
    ) u_tile_ctrl (
        .clk(clk),
        .rst_n(rst_n),
        .cfg_valid(tile_cfg_valid),
        .cfg_ready(tile_cfg_ready),
        .cfg_img_h(cur_in_h),
        .cfg_img_w(cur_in_w),
        .cfg_tile_h(TILE_H[DIM_W-1:0]),
        .cfg_tile_w(TILE_W[DIM_W-1:0]),
        .cfg_stride(stride_reg),
        .cfg_pad(PAD[DIM_W-1:0]),
        .cfg_kernel(KERNEL[DIM_W-1:0]),
        .start(tile_start),
        .tile_valid(tile_valid),
        .tile_ready(tile_ready),
        .tile_out_row(tile_out_row),
        .tile_out_col(tile_out_col),
        .tile_out_h(tile_out_h),
        .tile_out_w(tile_out_w),
        .tile_in_row(tile_in_row),
        .tile_in_col(tile_in_col),
        .tile_in_h(tile_in_h),
        .tile_in_w(tile_in_w),
        .done(tile_done)
    );

    assign busy = (state != S_IDLE);

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 1'b0;
            layer_idx <= '0;
            cur_in_h <= '0;
            cur_in_w <= '0;
            cur_out_h <= '0;
            cur_out_w <= '0;
            cur_in_c <= '0;
            cur_out_c <= '0;
            tile_cfg_valid <= 1'b0;
            tile_start <= 1'b0;
            tile_ready <= 1'b0;
            tile_active <= 1'b0;
            tile_wait <= 1'b0;
            conv1_start <= 1'b0;
            dws_start <= 1'b0;
            base_sel <= 1'b0;
        end else begin
            done <= 1'b0;
            tile_cfg_valid <= 1'b0;
            tile_start <= 1'b0;
            tile_ready <= 1'b0;
            conv1_start <= 1'b0;
            dws_start <= 1'b0;
            if (tile_wait) begin
                tile_wait <= 1'b0;
            end

            if (start && state == S_IDLE) begin
                layer_idx <= '0;
                cur_in_h <= cfg_in_img_h;
                cur_in_w <= cfg_in_img_w;
                cur_in_c <= 3;
                cur_out_c <= 32;
                base_sel <= 1'b0;
                state <= S_CFG;
            end

            if (state == S_CFG) begin
                cur_out_c <= next_out_c;
                cur_out_h <= (cur_in_h + (PAD << 1) - KERNEL) / stride_reg + 1'b1;
                cur_out_w <= (cur_in_w + (PAD << 1) - KERNEL) / stride_reg + 1'b1;

                if (tile_cfg_ready) begin
                    tile_cfg_valid <= 1'b1;
                    state <= S_START;
                end
            end

            if (state == S_START) begin
                tile_start <= 1'b1;
                tile_active <= 1'b0;
                state <= S_TILE;
            end

            if (state == S_TILE) begin
                if (tile_valid && !tile_active && !runner_busy && !tile_wait) begin
                    tile_active <= 1'b1;
                    if (layer_is_conv1) begin
                        conv1_start <= 1'b1;
                    end else begin
                        dws_start <= 1'b1;
                    end
                end

                if (runner_done && tile_active) begin
                    tile_active <= 1'b0;
                    tile_ready <= 1'b1;
                    tile_wait <= 1'b1;
                end

                if (tile_done && !tile_active) begin
                    state <= S_NEXT;
                end
            end

            if (state == S_NEXT) begin
                cur_in_h <= cur_out_h;
                cur_in_w <= cur_out_w;
                cur_in_c <= cur_out_c;
                base_sel <= ~base_sel;

                if (layer_idx == NUM_LAYERS - 1) begin
                    state <= S_DONE;
                end else begin
                    layer_idx <= layer_idx + 1'b1;
                    state <= S_CFG;
                end
            end

            if (state == S_DONE) begin
                done <= 1'b1;
                state <= S_IDLE;
            end
        end
    end
endmodule
