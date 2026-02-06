module pw_conv_1x1_vec #(
    parameter int DATA_W = 8,
    parameter int ACC_W = 32,
    parameter int OC_PAR = 16,
    parameter int IC_PAR = 8
) (
    input  logic clk,
    input  logic rst_n,

    input  logic in_valid,
    output logic in_ready,
    input  logic signed [IC_PAR*DATA_W-1:0] in_data_vec,
    input  logic signed [OC_PAR*IC_PAR*DATA_W-1:0] weight_vec,
    input  logic signed [OC_PAR*ACC_W-1:0] bias_vec,
    input  logic first_in_ch,
    input  logic last_in_ch,

    output logic out_valid,
    input  logic out_ready,
    output logic signed [OC_PAR*ACC_W-1:0] out_acc_vec
);
    logic signed [OC_PAR*ACC_W-1:0] acc_reg;
    logic signed [OC_PAR*ACC_W-1:0] acc_next;

    assign in_ready = !out_valid || out_ready;

    integer i;
    integer j;
    always_comb begin
        acc_next = acc_reg;
        for (i = 0; i < OC_PAR; i = i + 1) begin
            logic signed [ACC_W-1:0] acc_base;
            acc_base = first_in_ch ? bias_vec[i*ACC_W +: ACC_W] : acc_reg[i*ACC_W +: ACC_W];
            acc_next[i*ACC_W +: ACC_W] = acc_base;
            for (j = 0; j < IC_PAR; j = j + 1) begin
                logic signed [DATA_W-1:0] w_ij;
                logic signed [DATA_W-1:0] x_j;
                logic signed [ACC_W-1:0] w_ext;
                logic signed [ACC_W-1:0] x_ext;
                w_ij = weight_vec[(i*IC_PAR + j)*DATA_W +: DATA_W];
                x_j = in_data_vec[j*DATA_W +: DATA_W];
                w_ext = {{(ACC_W-DATA_W){w_ij[DATA_W-1]}}, w_ij};
                x_ext = {{(ACC_W-DATA_W){x_j[DATA_W-1]}}, x_j};
                acc_next[i*ACC_W +: ACC_W] = acc_next[i*ACC_W +: ACC_W] +
                                             (x_ext * w_ext);
            end
        end
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_reg <= '0;
            out_valid <= 1'b0;
            out_acc_vec <= '0;
        end else begin
            if (out_valid && out_ready) begin
                out_valid <= 1'b0;
            end

            if (in_valid && in_ready) begin
                if (last_in_ch) begin
                    out_acc_vec <= acc_next;
                    out_valid <= 1'b1;
                    acc_reg <= '0;
                end else begin
                    acc_reg <= acc_next;
                end
            end
        end
    end
endmodule
