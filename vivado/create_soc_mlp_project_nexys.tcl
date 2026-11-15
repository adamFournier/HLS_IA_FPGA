# Auto-generated Nexys A7 Artix-7 Vivado project (UART MLP)
# Topology: 784 -> 64 -> 32 -> 10
# Architecture: sequential
set project_name {SOC_MLP_nexys_a7_sequential}
set part_name {xc7a100tcsg324-1}
set top_name {SOC_MLP}
set project_dir {C:/Users/adams/Documents/MLP_HLS-main/vivado/build/SOC_MLP_nexys_a7_sequential}
set rtl_dir {C:/Users/adams/Documents/MLP_HLS-main/src/rtl/generated/src_nexys_sequential}
set xdc_path {C:/Users/adams/Documents/MLP_HLS-main/constraints/nexys_a7.xdc}
set header_path {C:/Users/adams/Documents/MLP_HLS-main/src/rtl/generated/src_nexys_sequential/scales.vh}
set run_flow 0
foreach arg $argv {
    if {$arg eq "--run"} {set run_flow 1}
}

# Never silently delete an existing Vivado project.
if {[file exists [file join $project_dir "$project_name.xpr"]]} {
    error "Project already exists: $project_dir. Rename/move it or use a new project_name."
}
file mkdir $project_dir
create_project $project_name $project_dir -part $part_name
set_property target_language Verilog [current_project]
set_property include_dirs [list $rtl_dir] [get_filesets sources_1]

set rtl_files {
    {C:/Users/adams/Documents/MLP_HLS-main/src/rtl/generated/src_nexys_sequential/dsp.sv}
    {C:/Users/adams/Documents/MLP_HLS-main/src/rtl/generated/src_nexys_sequential/mem_a.sv}
    {C:/Users/adams/Documents/MLP_HLS-main/src/rtl/generated/src_nexys_sequential/mem_b.sv}
    {C:/Users/adams/Documents/MLP_HLS-main/src/rtl/generated/src_nexys_sequential/mlp_sequential.sv}
    {C:/Users/adams/Documents/MLP_HLS-main/src/rtl/generated/src_nexys_sequential/mlp_sequential_ctrl.sv}
    {C:/Users/adams/Documents/MLP_HLS-main/src/rtl/generated/src_nexys_sequential/mlp_sequential_datapath.sv}
    {C:/Users/adams/Documents/MLP_HLS-main/src/rtl/generated/src_nexys_sequential/rom_act_all.sv}
    {C:/Users/adams/Documents/MLP_HLS-main/src/rtl/generated/src_nexys_sequential/rom_bias_all.sv}
    {C:/Users/adams/Documents/MLP_HLS-main/src/rtl/generated/src_nexys_sequential/rom_weight_all.sv}
    {C:/Users/adams/Documents/MLP_HLS-main/src/rtl/generated/src_nexys_sequential/soc_mlp.sv}
    {C:/Users/adams/Documents/MLP_HLS-main/src/rtl/generated/src_nexys_sequential/uart_input.sv}
    {C:/Users/adams/Documents/MLP_HLS-main/src/rtl/generated/src_nexys_sequential/uart_output.sv}
    {C:/Users/adams/Documents/MLP_HLS-main/src/rtl/generated/src_nexys_sequential/uart_rx.sv}
    {C:/Users/adams/Documents/MLP_HLS-main/src/rtl/generated/src_nexys_sequential/uart_tx.sv}
}
foreach src $rtl_files {
    if {![file isfile $src]} {error "Missing RTL source: $src"}
    add_files -norecurse -fileset sources_1 $src
    set_property file_type SystemVerilog [get_files $src]
}
add_files -norecurse -fileset sources_1 $header_path
set_property file_type {Verilog Header} [get_files $header_path]
if {![file isfile $xdc_path]} {error "Missing XDC file: $xdc_path"}
add_files -norecurse -fileset constrs_1 $xdc_path
set_property top $top_name [get_filesets sources_1]
update_compile_order -fileset sources_1
puts "INFO: Nexys A7 Vivado project created: $project_dir"

if {$run_flow} {
    launch_runs synth_1 -jobs 8
    wait_on_run synth_1
    if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
        error "Synthesis did not complete successfully"
    }
    if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} {
        error "Synthesis failed: [get_property STATUS [get_runs synth_1]]"
    }
    launch_runs impl_1 -to_step write_bitstream -jobs 8
    wait_on_run impl_1
    if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
        error "Implementation/bitstream did not complete successfully"
    }
}
