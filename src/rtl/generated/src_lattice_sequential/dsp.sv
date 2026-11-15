
// -----------------------------------------------------------------------------
// dsp.sv - Generic MAC for Xilinx Artix-7 / Nexys A7
//
// first_mac = 1 : P <= C + A*B
// first_mac = 0 : P <= P + A*B
//
// - Signed multiplication
// - Parameterized accumulator width
// - Synchronous reset
// - Clock enable
// - One registered MAC operation per enabled clock cycle
// -----------------------------------------------------------------------------

module dsp #(
    parameter int A_WIDTH = 16,
    parameter int B_WIDTH = 16,
    parameter int P_WIDTH = 48
)(
    input  logic                         clk,
    input  logic                         rstn,
    input  logic                         ce,
    input  logic                         first_mac,
    input  logic signed [A_WIDTH-1:0]    A,
    input  logic signed [B_WIDTH-1:0]    B,
    input  logic signed [P_WIDTH-1:0]    C,
    output logic signed [P_WIDTH-1:0]    P
);

    logic signed [P_WIDTH-1:0] product;
    logic signed [P_WIDTH-1:0] addend;

    // Signed multiplication, extended to accumulator width.
    always_comb begin
        product = $signed(A) * $signed(B);
        addend  = first_mac ? C : P;
    end

    // Multiply-accumulate with clock enable.
    always_ff @(posedge clk) begin
        if (!rstn) begin
            P <= '0;
        end
        else if (ce) begin
            P <= addend + product;
        end
    end

endmodule
