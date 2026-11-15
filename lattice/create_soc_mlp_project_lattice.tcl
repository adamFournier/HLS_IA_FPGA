# -----------------------------------------------------------------------------
# Auto-generated Lattice Radiant project script.
# Layout    : mem_layout.json
# Topology  : 784 -> 64 -> 32 -> 10
# Mode      : sequential
# -----------------------------------------------------------------------------

set script_dir [file dirname [file normalize [info script]]]
set repo_root  [file normalize [file join $script_dir ".."]]

set project_name "SOC_MLP_lattice_sequential"
set impl_name    "impl_1"
set device       "iCE40UP5K-SG48I"
set synthesis    "synplify"
set top_name     "SOC_MLP"
set run_flow     0

foreach arg $argv {
    if {$arg eq "--run"} { set run_flow 1 }
}

set project_dir [file normalize [file join $repo_root "lattice" "build" $project_name]]
set rtl_dir     [file normalize [file join $repo_root "src/rtl/generated/src_lattice_sequential"]]

puts "INFO: repo_root    = $repo_root"
puts "INFO: project_name = $project_name"
puts "INFO: project_dir  = $project_dir"
puts "INFO: impl_name    = $impl_name"
puts "INFO: device       = $device"
puts "INFO: synthesis    = $synthesis"
puts "INFO: top          = $top_name"
puts "INFO: mode         = sequential"
puts "INFO: rtl_dir      = $rtl_dir"

if {[file exists $project_dir]} {
    file delete -force $project_dir
}
file mkdir $project_dir

# Radiant command names can differ slightly between versions. The script keeps
# the generated source list explicit and stable so it can also be reused manually.
prj_create -name $project_name -impl $impl_name -dev $device -synthesis $synthesis -dir $project_dir

set rtl_files {
    scales.vh
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

foreach src $rtl_files {
    set src_path [file normalize [file join $rtl_dir $src]]
    puts "INFO: add source $src_path"
    prj_add_source $src_path
}

set pdc_files {
    src/rtl/src_lattice_sequential/contraints.pdc
}

foreach pdc $pdc_files {
    if {[file pathtype $pdc] eq "absolute"} {
        set pdc_path [file normalize $pdc]
    } else {
        set pdc_path [file normalize [file join $repo_root $pdc]]
    }
    if {[file exists $pdc_path]} {
        puts "INFO: add PDC $pdc_path"
        prj_add_source $pdc_path
    }
}

# Set the intended top when supported by the installed Radiant version.
catch {prj_set_impl_opt -impl $impl_name top $top_name}
catch {prj_set_top $top_name}

if {$run_flow} {
    puts "INFO: Running implementation flow"
    prj_run Synthesis -impl $impl_name
    prj_run Map       -impl $impl_name
    prj_run PAR       -impl $impl_name
    prj_run Export    -impl $impl_name
}
