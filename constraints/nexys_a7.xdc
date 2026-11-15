
# ==================================================
# Nexys A7-100T - Contraintes pour SOC_MLP
# ==================================================

# Horloge 100 MHz
set_property PACKAGE_PIN E3 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
create_clock -period 10.000 -name sys_clk [get_ports clk]

# UART RX : reception depuis le PC
set_property PACKAGE_PIN C4 [get_ports uart_rx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_rx]

# UART TX : transmission vers le PC
set_property PACKAGE_PIN D4 [get_ports uart_tx]
set_property IOSTANDARD LVCMOS33 [get_ports uart_tx]
