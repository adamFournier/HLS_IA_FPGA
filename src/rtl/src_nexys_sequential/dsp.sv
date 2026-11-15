
module dsp #(
    parameter int A_WIDTH = 16,
    parameter int B_WIDTH = 16,
    parameter int P_WIDTH = 32
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

    always_comb begin
        product = $signed(A) * $signed(B);
        addend  = first_mac ? C : P;
    end

    always_ff @(posedge clk) begin
        if (!rstn)
            P <= '0;
        else if (ce)
            P <= addend + product;
    end

endmodule
