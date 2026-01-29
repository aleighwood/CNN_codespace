module pw_conv_1x1 #(
    parameter int DATA_W = 8,
    parameter int ACC_W = 32
) (
    input  logic clk,
    input  logic rst_n,

    input  logic in_valid,
    output logic in_ready,
    input  logic signed [DATA_W-1:0] in_data,
    input  logic signed [DATA_W-1:0] weight,
    input  logic signed [ACC_W-1:0] bias,
    input  logic first_in_ch,
    input  logic last_in_ch,

    output logic out_valid,
    input  logic out_ready,
    output logic signed [ACC_W-1:0] out_acc
);
    logic signed [ACC_W-1:0] acc_reg;
    logic signed [ACC_W-1:0] acc_next;

    assign in_ready = !out_valid || out_ready;

    always_comb begin
        if (first_in_ch) begin
            acc_next = bias;
        end else begin
            acc_next = acc_reg;
        end
        acc_next = acc_next + ($signed(in_data) * $signed(weight));
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            acc_reg <= '0;
            out_valid <= 1'b0;
            out_acc <= '0;
        end else begin
            if (out_valid && out_ready) begin
                out_valid <= 1'b0;
            end

            if (in_valid && in_ready) begin
                acc_reg <= acc_next;
                if (last_in_ch) begin
                    out_acc <= acc_next;
                    out_valid <= 1'b1;
                    acc_reg <= '0;
                end
            end
        end
    end
endmodule
