module requant_q31 #(
    parameter int DATA_W = 8,
    parameter int ACC_W = 32,
    parameter int MUL_W = 32,
    parameter int SHIFT_W = 6
) (
    input  logic clk,
    input  logic rst_n,

    input  logic in_valid,
    output logic in_ready,
    input  logic signed [ACC_W-1:0] in_acc,

    input  logic signed [MUL_W-1:0] mul_q31,
    input  logic signed [SHIFT_W-1:0] shift,
    input  logic signed [DATA_W-1:0] zp_out,
    input  logic signed [DATA_W-1:0] relu6_max,
    input  logic relu6_en,

    output logic out_valid,
    input  logic out_ready,
    output logic signed [DATA_W-1:0] out_q
);
    localparam int Q_MIN = -(1 << (DATA_W - 1));
    localparam int Q_MAX = (1 << (DATA_W - 1)) - 1;

    function automatic signed [ACC_W-1:0] srdhm(
        input signed [ACC_W-1:0] a,
        input signed [MUL_W-1:0] b
    );
        logic signed [63:0] ab;
        logic signed [63:0] nudge;
        logic signed [63:0] res;
        begin
            ab = $signed(a) * $signed(b);
            if (ab >= 0) begin
                nudge = 64'sd1073741824; // 1 << 30
            end else begin
                nudge = -64'sd1073741823; // 1 - (1 << 30)
            end
            res = (ab + nudge) >>> 31;
            if (res > 64'sd2147483647) begin
                res = 64'sd2147483647;
            end else if (res < -64'sd2147483648) begin
                res = -64'sd2147483648;
            end
            srdhm = res[ACC_W-1:0];
        end
    endfunction

    function automatic signed [ACC_W-1:0] rdivp(
        input signed [ACC_W-1:0] x,
        input int unsigned shift_amt
    );
        logic signed [ACC_W-1:0] mask;
        logic signed [ACC_W-1:0] remainder;
        logic signed [ACC_W-1:0] threshold;
        begin
            if (shift_amt == 0) begin
                rdivp = x;
            end else begin
                mask = ({{(ACC_W){1'b0}}} | ((1 <<< shift_amt) - 1));
                remainder = x & mask;
                threshold = (mask >>> 1);
                if (x < 0) begin
                    threshold = threshold + 1;
                end
                rdivp = (x >>> shift_amt) + ((remainder > threshold) ? 1 : 0);
            end
        end
    endfunction

    logic signed [ACC_W-1:0] scaled;
    logic signed [ACC_W-1:0] with_zp;
    logic signed [ACC_W-1:0] clipped;
    logic signed [DATA_W-1:0] quantized;
    int unsigned shift_amt;

    always_comb begin
        shift_amt = $unsigned(shift);
        scaled = rdivp(srdhm(in_acc, mul_q31), shift_amt);

        with_zp = scaled + $signed(zp_out);

        if (relu6_en) begin
            if (with_zp < $signed(zp_out)) begin
                clipped = $signed(zp_out);
            end else if (with_zp > $signed(relu6_max)) begin
                clipped = $signed(relu6_max);
            end else begin
                clipped = with_zp;
            end
        end else begin
            clipped = with_zp;
        end

        if (clipped > Q_MAX) begin
            quantized = $signed(Q_MAX);
        end else if (clipped < Q_MIN) begin
            quantized = $signed(Q_MIN);
        end else begin
            quantized = clipped[DATA_W-1:0];
        end
    end

    assign in_ready = !out_valid || out_ready;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            out_valid <= 1'b0;
            out_q <= '0;
        end else begin
            if (out_valid && out_ready) begin
                out_valid <= 1'b0;
            end

            if (in_valid && in_ready) begin
                out_q <= quantized;
                out_valid <= 1'b1;
            end
        end
    end
endmodule
