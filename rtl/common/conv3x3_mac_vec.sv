module conv3x3_mac_vec #(
    parameter int DATA_W = 8,
    parameter int ACC_W = 32,
    parameter int OC_PAR = 4
) (
    input  logic clk,
    input  logic rst_n,

    input  logic in_valid,
    output logic in_ready,
    input  logic signed [DATA_W*9-1:0] window_flat,
    input  logic signed [OC_PAR*DATA_W*9-1:0] weight_flat_vec,

    output logic out_valid,
    input  logic out_ready,
    output logic signed [OC_PAR*ACC_W-1:0] out_acc_vec
);
    localparam int WIN_ELEMS = 9;

    logic signed [ACC_W-1:0] sum_comb [0:OC_PAR-1];
    integer oc_comb;
    integer oc_ff;
    integer k;

    always_comb begin
        for (oc_comb = 0; oc_comb < OC_PAR; oc_comb = oc_comb + 1) begin
            sum_comb[oc_comb] = '0;
            for (k = 0; k < WIN_ELEMS; k = k + 1) begin
                sum_comb[oc_comb] += $signed(window_flat[k*DATA_W +: DATA_W]) *
                                     $signed(weight_flat_vec[(oc_comb*WIN_ELEMS + k)*DATA_W +: DATA_W]);
            end
        end
    end

    assign in_ready = !out_valid || out_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_acc_vec <= '0;
        end else begin
            if (out_valid && out_ready) begin
                out_valid <= 1'b0;
            end

            if (in_valid && in_ready) begin
                for (oc_ff = 0; oc_ff < OC_PAR; oc_ff = oc_ff + 1) begin
                    out_acc_vec[oc_ff*ACC_W +: ACC_W] <= sum_comb[oc_ff];
                end
                out_valid <= 1'b1;
            end
        end
    end
endmodule
