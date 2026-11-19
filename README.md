# MLP Inference on FPGA

**PyTorch · Quantization · SystemVerilog · FPGA/SoC**

This project explores the hardware implementation of multilayer perceptrons (MLPs), from PyTorch training and fixed-point quantization to RTL generation and FPGA integration. A Fashion-MNIST classification model is used to study the trade-offs between numerical accuracy, latency, throughput, and hardware resources.

## Workflow

```text
Dataset → PyTorch MLP → Fixed-point quantization
                          ↓
                 SystemVerilog generation
                          ↓
                Sequential / Pipelined RTL
                          ↓
                Simulation → FPGA integration
```

## Key features

- Configurable MLP training and evaluation with PyTorch.
- Quantization of weights, biases, and activations; export of memory files and test vectors.
- Automatic SystemVerilog generation for **sequential** (shared computation) and **pipelined** architectures.
- RTL testbenches and software-reference comparisons.
- Target-specific flows for **Nexys A7**, **Ultra96-V2**, and **Lattice iCE40UP5K**, using UART or AXI-Stream/DMA interfaces as appropriate.

## Repository

| Path | Purpose |
|---|---|
| `notebooks/` | Training, quantization, project generation, and board tests |
| `src/py/` | Python training and quantization modules |
| `src/rtl/` | RTL templates and testbenches |
| `src/tcl/` | FPGA project-generation scripts |
| `vivado/`, `lattice/` | Vendor-specific integration |

## Getting started

Requires Python 3.10+ and the relevant FPGA vendor tools for hardware implementation.

```bash
git clone https://github.com/adamFournier/HLS_IA_FPGA.git
cd HLS_IA_FPGA
pip install -r requirements.txt
jupyter lab
```

Start with `notebooks/01_train_MLP.ipynb`, then `notebooks/02_quantize_MLP.ipynb`. Configure the dataset, model path, target FPGA, and architecture before generating the RTL.

**Status:** Ongoing project. Tool flows and validation depend on the selected configuration; physical-board results are not claimed for every target.

**Author:** [Adam Fournier](https://www.linkedin.com/in/adam-fournier-6b625132a/) — ENSEEIHT, Toulouse INP
