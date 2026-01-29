module gap_runner #(
    parameter int DATA_W = 8,
    parameter int ACC_W = 32,
    parameter int MUL_W = 16,
    parameter int BIAS_W = 32,
    parameter int SHIFT_W = 6,
    parameter int ADDR_W = 32,
    parameter int DIM_W = 16
) (
    input  logic clk,
    input  logic rst_n,

    input  logic start,
    output logic busy,
    output logic done,

    input  logic [DIM_W-1:0] cfg_in_h,
    input  logic [DIM_W-1:0] cfg_in_w,
    input  logic [DIM_W-1:0] cfg_in_c,
    input  logic [ADDR_W-1:0] cfg_in_base,
    input  logic [ADDR_W-1:0] cfg_out_base,

    output logic in_rd_en,
    output logic [ADDR_W-1:0] in_rd_addr,
    input  logic signed [DATA_W-1:0] in_rd_data,

    output logic out_wr_en,
    output logic [ADDR_W-1:0] out_wr_addr,
    output logic signed [DATA_W-1:0] out_wr_data,

    output logic [DIM_W-1:0] gap_ch_idx,
    input  logic signed [MUL_W-1:0] gap_mul,
    input  logic signed [BIAS_W-1:0] gap_bias,
    input  logic [SHIFT_W-1:0] gap_shift
);
    typedef enum logic [2:0] {
        S_IDLE,
        S_READ,
        S_QUANT,
        S_WRITE,
        S_NEXT,
        S_DONE
    } state_t;

    state_t state;

    logic [DIM_W-1:0] h_reg;
    logic [DIM_W-1:0] w_reg;
    logic [DIM_W-1:0] c_reg;
    logic [ADDR_W-1:0] in_base_reg;
    logic [ADDR_W-1:0] out_base_reg;

    logic [DIM_W-1:0] row;
    logic [DIM_W-1:0] col;

    logic signed [ACC_W-1:0] acc;

    logic rq_in_valid;
    logic rq_in_ready;
    logic rq_out_valid;
    logic rq_out_ready;
    logic signed [DATA_W-1:0] rq_out;

    assign busy = (state != S_IDLE);

    requant_relu6 #(
        .DATA_W(DATA_W),
        .ACC_W(ACC_W),
        .MUL_W(MUL_W),
        .BIAS_W(BIAS_W),
        .SHIFT_W(SHIFT_W)
    ) u_requant (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(rq_in_valid),
        .in_ready(rq_in_ready),
        .in_acc(acc),
        .mul(gap_mul),
        .bias(gap_bias),
        .shift(gap_shift),
        .relu6_max({DATA_W{1'b1}}),
        .relu6_en(1'b0),
        .out_valid(rq_out_valid),
        .out_ready(rq_out_ready),
        .out_q(rq_out)
    );

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            done <= 1'b0;
            gap_ch_idx <= '0;
            row <= '0;
            col <= '0;
            acc <= '0;
            h_reg <= '0;
            w_reg <= '0;
            c_reg <= '0;
            in_base_reg <= '0;
            out_base_reg <= '0;
        end else begin
            done <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        h_reg <= cfg_in_h;
                        w_reg <= cfg_in_w;
                        c_reg <= cfg_in_c;
                        in_base_reg <= cfg_in_base;
                        out_base_reg <= cfg_out_base;
                        gap_ch_idx <= '0;
                        row <= '0;
                        col <= '0;
                        acc <= '0;
                        state <= S_READ;
                    end
                end
                S_READ: begin
                    acc <= acc + in_rd_data;
                    if (col == w_reg - 1'b1) begin
                        col <= '0;
                        if (row == h_reg - 1'b1) begin
                            row <= '0;
                            state <= S_QUANT;
                        end else begin
                            row <= row + 1'b1;
                        end
                    end else begin
                        col <= col + 1'b1;
                    end
                end
                S_QUANT: begin
                    if (rq_in_ready) begin
                        state <= S_WRITE;
                    end
                end
                S_WRITE: begin
                    if (rq_out_valid) begin
                        state <= S_NEXT;
                    end
                end
                S_NEXT: begin
                    if (gap_ch_idx == c_reg - 1'b1) begin
                        state <= S_DONE;
                    end else begin
                        gap_ch_idx <= gap_ch_idx + 1'b1;
                        row <= '0;
                        col <= '0;
                        acc <= '0;
                        state <= S_READ;
                    end
                end
                S_DONE: begin
                    done <= 1'b1;
                    state <= S_IDLE;
                end
                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end

    always_comb begin
        in_rd_en = 1'b0;
        in_rd_addr = '0;
        out_wr_en = 1'b0;
        out_wr_addr = '0;
        out_wr_data = '0;
        rq_in_valid = 1'b0;
        rq_out_ready = 1'b0;

        case (state)
            S_READ: begin
                in_rd_en = 1'b1;
                in_rd_addr = in_base_reg + (gap_ch_idx * (h_reg * w_reg)) + (row * w_reg) + col;
            end
            S_QUANT: begin
                rq_in_valid = 1'b1;
                rq_out_ready = 1'b1;
            end
            S_WRITE: begin
                rq_out_ready = 1'b1;
                if (rq_out_valid) begin
                    out_wr_en = 1'b1;
                    out_wr_addr = out_base_reg + gap_ch_idx;
                    out_wr_data = rq_out;
                end
            end
            default: begin
            end
        endcase
    end
endmodule
