# MLP Inference on FPGA

**PyTorch · Fixed-point Quantization · SystemVerilog · FPGA/SoC**

This project implements a workflow from PyTorch MLP training to automatically generated FPGA hardware. It explores **sequential and pipelined RTL architectures** using fixed-point quantization, including experiments with Fashion-MNIST.

## Workflow

```text
Notebook 01: Train the MLP (PyTorch)
                 ↓
Notebook 02: Quantize the model and generate RTL
                 ↓
Notebook 03: Generate the FPGA project Tcl script
                 ↓
       Run Tcl in Vivado / Radiant
                 ↓
   FPGA project creation and implementation
                 ↓
Notebook 04: Transfer data and validate on board
```

## How to run

1. **Train** — Run `notebooks/01_train_MLP.ipynb` to train and save the PyTorch model.
2. **Quantize + generate RTL** — Run `notebooks/02_quantize_MLP.ipynb`. Choose the FPGA target and `sequential` or `pipeline` architecture. It exports quantized parameters, test vectors, and SystemVerilog RTL.
3. **Generate Tcl** — Run the corresponding `notebooks/03_generate_TCL_*.ipynb` notebook to generate the FPGA project creation script.
4. **Create the FPGA project** — Open **Vivado** (for AMD/Xilinx targets) and execute the generated Tcl script in its Tcl Console:

   ```tcl
   source vivado/create_soc_mlp_project_vivado.tcl
   ```

   This creates and configures the project with the generated RTL and required integration files. To run the full Vivado flow where supported:

   ```tcl
   source vivado/create_soc_mlp_project_vivado.tcl -tclargs --run
   ```

   For a Lattice target, use the generated Radiant Tcl script in **Lattice Radiant** instead.
5. **Test** — Simulate the design, then use the appropriate `04_push_data_*` notebook/script for board-level testing when hardware is available.

**Note:** The generated Tcl script is run in **Vivado**, not Vitis HLS, for the Xilinx FPGA project.

## Architectures and targets

- **Sequential RTL:** shared computation resources across layers.
- **Pipelined RTL:** layer-level hardware parallelism for higher potential throughput.
- **Targets:** Nexys A7, Ultra96-V2 and Lattice iCE40UP5K.
- **Interfaces:** UART or AXI-Stream/DMA depending on the target.

## Repository structure

| Directory | Content |
|---|---|
| `notebooks/` | Training, quantization, Tcl generation and validation |
| `src/py/` | Python model and quantization code |
| `src/rtl/` | SystemVerilog templates and testbenches |
| `src/tcl/` | FPGA project-generation scripts |
| `vivado/`, `lattice/` | Vendor tool integration |

## Setup

```bash
pip install -r requirements.txt
jupyter lab
```

Configure the dataset/model paths and target before running the notebooks. FPGA synthesis and implementation require the corresponding vendor tools.

**Status:** Ongoing project; not all architecture/board combinations have been validated on physical hardware.

**Author:** [Adam Fournier](https://www.linkedin.com/in/adam-fournier-6b625132a/) — ENSEEIHT, Toulouse INP
