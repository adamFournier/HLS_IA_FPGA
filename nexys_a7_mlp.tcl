# -----------------------------------------------------------------------------
# Vivado project generator for Nexys A7 + sequential SOC_MLP (UART)
# Neural network: 784 -> 64 -> 32 -> 10
# Based on the Lattice source list; requires Artix-7-compatible RTL.
# -----------------------------------------------------------------------------

# CONFIGURATION: Nexys A7-100T (for 50T use xc7a50tcsg324-1)
set project_name "SOC_MLP_nexys_a7"
set part_name    "xc7a100tcsg324-1"
set top_name     "SOC_MLP"
set run_flow     0                 ;# 0: create project only; 1: synth + impl + bitstream
set jobs         4

# Locate repository root from the TCL file location.
set script_dir [file dirname [file normalize [info script]]]
set repo_root ""
set candidate $script_dir
for {set level 0} {$level < 7} {incr level} {
    if {[file isdirectory [file join $candidate "src" "rtl" "generated"]]} {
        set repo_root $candidate
        break
    }
    set parent [file dirname $candidate]
    if {$parent eq $candidate} { break }
    set candidate $parent
}
if {$repo_root eq ""} {
    error "Project root not found. Put this TCL somewhere inside MLP_HLS-main."
}

# Reuse the Lattice UART RTL as a starting point; NOT the Lattice toolchain.
set rtl_dir [file normalize [file join $repo_root "src" "rtl" "generated" "src_lattice_sequential"]]
set quant_dir [file normalize [file join $repo_root "data" "quantized_data" "fashion_mnist_q16" "lattice" "sequential"]]
set project_dir [file normalize [file join $repo_root "vivado" "build" $project_name]]

# Create a Nexys A7 XDC matching the actual SOC_MLP top ports.
set xdc_path [file normalize [file join $repo_root "constraints" "nexys_a7.xdc"]]

# RTL modules to compile. Names are from your generated Lattice TCL.
set rtl_files {
    dsp.sv
    mem_a.sv
    mem_b.sv
    mlp_sequential.sv
    mlp_sequential_ctrl.sv
    mlp_sequential_datapath.sv
    rom_act_all.sv
    rom_bias_all.sv
    rom_weight_all.sv
    soc_mlp.sv
    uart_input.sv
    uart_output.sv
    uart_rx.sv
    uart_tx.sv
}

# Check source files before creating the project.
if {![file isdirectory $rtl_dir]} { error "RTL directory not found: $rtl_dir" }
foreach filename $rtl_files {
    if {![file isfile [file join $rtl_dir $filename]]} {
        error "Missing RTL file: [file join $rtl_dir $filename]"
    }
}
if {![file isfile [file join $rtl_dir "scales.vh"]]} {
    error "Missing scales.vh in $rtl_dir"
}

puts "INFO: Project root : $repo_root"
puts "INFO: FPGA         : $part_name"
puts "INFO: RTL          : $rtl_dir"
puts "INFO: Project      : $project_dir"

file mkdir [file dirname $project_dir]
if {[file exists [file join $project_dir "$project_name.xpr"]]} {
    error "Project already exists: $project_dir. Rename project_name or move the existing project."
}

create_project -force $project_name $project_dir -part $part_name
set_property target_language Verilog [current_project]
set_property simulator_language Mixed [current_project]
set_property include_dirs [list $rtl_dir] [get_filesets sources_1]

# Add SystemVerilog files.
foreach filename $rtl_files {
    set path [file normalize [file join $rtl_dir $filename]]
    add_files -norecurse -fileset sources_1 $path
    set_property file_type SystemVerilog [get_files $path]
    puts "INFO: Added RTL: $filename"
}

# Add Verilog header with architecture and quantization constants.
set header [file normalize [file join $rtl_dir "scales.vh"]]
add_files -norecurse -fileset sources_1 $header
set_property file_type {Verilog Header} [get_files $header]

# Add ROM initialization files, first looking in generated RTL, then quantized data.
foreach filename {weight_all.mem bias_all.mem act_all.mem} {
    set path ""
    foreach directory [list $rtl_dir $quant_dir] {
        set candidate [file normalize [file join $directory $filename]]
        if {[file isfile $candidate]} {
            set path $candidate
            break
        }
    }
    if {$path ne ""} {
        add_files -norecurse -fileset sources_1 $path
        puts "INFO: Added ROM data: $path"
    } else {
        puts "WARNING: $filename not found (act_all.mem may be unnecessary for analytic LeakyReLU)."
    }
}

# Physical pin and clock constraints for Nexys A7.
if {[file isfile $xdc_path]} {
    add_files -fileset constrs_1 $xdc_path
    puts "INFO: Added XDC: $xdc_path"
} else {
    puts "WARNING: XDC not found: $xdc_path"
    puts "WARNING: Create XDC with clock, reset, UART pins before generating a bitstream."
    if {$run_flow} {
        error "run_flow=1 requires an XDC file: $xdc_path"
    }
}

set_property top $top_name [get_filesets sources_1]
update_compile_order -fileset sources_1

# The generator does not create Zynq/DMA IP: Nexys A7 is Artix-7.
# SOC_MLP must itself expose a suitable clock/reset and UART interface.

if {$run_flow} {
    puts "INFO: Starting synthesis..."
    launch_runs synth_1 -jobs $jobs
    wait_on_run synth_1
    if {[get_property STATUS [get_runs synth_1]] ne "synth_design Complete!"} {
        puts "WARNING: Check synthesis run status/log before continuing."
    }
    puts "INFO: Starting implementation and bitstream..."
    launch_runs impl_1 -to_step write_bitstream -jobs $jobs
    wait_on_run impl_1
}

puts "INFO: Vivado project created: [file join $project_dir "$project_name.xpr"]"
puts "INFO: Next: check top ports, XDC constraints, memories, and synthesis messages."
