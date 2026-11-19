# MLP Hardware Acceleration on FPGA

**From PyTorch training to fixed-point inference and generated SystemVerilog**

This project explores the implementation of multilayer perceptrons (MLPs) on FPGA, connecting machine-learning development in Python with digital hardware design. The workflow covers model training, numerical quantization, hardware generation, simulation, and target-specific integration.

The focus is on the trade-offs between **numerical precision, hardware resources, latency, and throughput**, particularly when choosing between sequential and pipelined inference architectures.

## Project overview

```text
Dataset
   |
   v
PyTorch MLP training
   |
   v
Fixed-point calibration and quantization
   |
   +--> Software / quantized reference outputs
   |
   v
Memory initialization and SystemVerilog generation
   |
   +----------------------+ 
   |                      |
   v                      v
Sequential RTL       Pipelined RTL
(shared compute)     (layer-level hardware)
   |                      |
   +-----------+----------+
               |
               v
RTL simulation and FPGA tool flow
               |
               v
Target integration, testing and performance analysis
```

## Technical highlights

- **ML model development:** PyTorch training and evaluation of configurable MLP topologies, including a Fashion-MNIST classification workflow.
- **Quantization:** configurable fixed-point representation for weights, biases, and activations, with calibration and overflow checks.
- **Hardware generation:** automated export of model parameters, memory files, configuration headers, test vectors, and SystemVerilog RTL.
- **Architecture exploration:** sequential and pipelined implementations with different compute-resource and throughput trade-offs.
- **Verification:** simulation with reference outputs generated from the quantized model.
- **FPGA integration:** target-specific project generation and host-to-FPGA interfaces.

## Hardware architectures

| | Sequential | Pipelined |
|---|---|---|
| Organization | Shared computation across layers | Dedicated hardware organized by layers |
| Main objective | Reduce hardware replication | Increase processing concurrency |
| Design trade-off | More time-multiplexed computation | Higher potential throughput at a resource cost |
| Project support | SystemVerilog generation and testbench | SystemVerilog generation and testbench |

Actual latency, throughput, and resource utilization depend on the network topology, numeric format, device, and implementation results; this repository does not claim a single universal performance improvement.

## FPGA targets

The project's target profiles include:

| Platform | FPGA / SoC | Toolchain | Data interface |
|---|---|---|---|
| **Nexys A7** | AMD/Xilinx Artix-7 | Vivado | UART |
| **Ultra96-V2** | AMD/Xilinx Zynq UltraScale+ MPSoC | Vivado | AXI-Stream / DMA |
| **Lattice iCE40UP5K** | Lattice FPGA | Radiant | UART |

Target support means that configuration and generation paths exist in the project; successful synthesis, implementation, or board-level validation should be assessed separately for each configuration.

## Repository structure

```text
notebooks/
  01_train_MLP.ipynb                 Model training
  02_quantize_MLP.ipynb              Quantization and RTL export
  03_generate_TCL_*.ipynb            FPGA project-generation workflows
  04_push_data_*                     Board communication / tests

src/
  py/                                Training, quantization and utilities
  rtl/                               SystemVerilog templates and testbenches
  tcl/                               FPGA project-generation scripts

vivado/                               AMD/Xilinx integration
lattice/                              Lattice integration
data/                                 Local datasets and generated artifacts
```

Some files under `data/` and generated build directories may be intentionally excluded from the public release.

## Example: Fashion-MNIST quantization

The current quantization notebook includes a Fashion-MNIST configuration for a trained PyTorch MLP:

```python
config = {
    "name": "fashion_mnist_q16",
    "model_path": "data/fashion_mnist/mlp_fashion_mnist.pt",
    "dataset_path": "data/fashion_mnist.npz",
    "output_root": "data/quantized_data",
    "target": "nexys_a7",
    "architecture": "sequential",
    "weight_bits": 8,
    "bias_bits": 16,
    "act_bits": 16,
    "generate_rtl": True,
}
```

The full notebook configuration also specifies calibration, DSP accumulator constraints, test samples, activation look-up tables, and reporting. Change the target and architecture according to the desired experiment and available hardware.

## Getting started

**Requirements:** Python 3.10+, PyTorch, NumPy, pandas, scikit-learn, and Jupyter. FPGA generation and deployment also require the relevant vendor tools.

```bash
git clone https://github.com/adamFournier/HLS_IA_FPGA.git
cd HLS_IA_FPGA

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
jupyter lab
```

Then follow the notebooks in order:

1. **Train** the MLP in `notebooks/01_train_MLP.ipynb`, using a suitable local dataset and configuration.
2. **Quantize and export** using `notebooks/02_quantize_MLP.ipynb`. Check the dataset paths, model checkpoint, FPGA target, and architecture.
3. **Generate the FPGA project** using the appropriate `03_generate_TCL_*` notebook or the scripts in `src/tcl/`.
4. **Simulate and evaluate** the resulting RTL using the generated test vectors and the testbenches in `src/rtl/tb/`.
5. **Validate on hardware** where the relevant board, software image, and toolchain are available.

The exact setup may vary by target. Model checkpoints and datasets are not necessarily included in the repository.

## Verification and evaluation

The project is designed to compare the results of different stages:

- floating-point PyTorch inference;
- quantized software/reference inference;
- generated RTL simulation;
- FPGA outputs, where board validation has been performed.

Useful measurements include output error, numeric overflow, inference latency, throughput, and hardware resource utilization. Performance results should always be identified by **model, architecture, quantization, board, and tool version** rather than compared without context.

## Technologies

**Python · PyTorch · NumPy · SystemVerilog · RTL · FPGA · Vivado · Lattice Radiant · AXI-Stream · DMA · UART · Fixed-point arithmetic**

## Project status

This is an evolving engineering project. The repository includes training and quantization code, RTL generation paths, FPGA integration scripts, and testbench infrastructure. Not every combination of dataset, target, and architecture is necessarily validated on physical hardware.

## Author

**Adam Fournier**  
Engineering student, ENSEEIHT — Toulouse INP  
Embedded AI · FPGA/SoC Design · Digital Signal Processing

[LinkedIn](https://www.linkedin.com/in/adam-fournier-6b625132a/)

---

*Before reusing the code or publishing third-party datasets and hardware assets, check the applicable licences and permissions.*
