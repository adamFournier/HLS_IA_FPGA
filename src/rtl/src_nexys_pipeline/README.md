# Nexys A7 pipelined UART MLP

Independent port of the original Lattice layer pipeline onto the Nexys A7
(Artix-7 at 100 MHz). The existing sequential implementation is not replaced.

- External top ports: `clk`, `uart_rx`, `uart_tx`.
- Each MLP layer has its own MAC (48-bit accumulator), synchronous per-layer
  weight/bias ROMs and a ping-pong buffer at each layer boundary.
- ROM weights, biases and activation LUTs come from the per-layer pipeline
  `mem_layout.json` exported by notebook 02, not the sequential layout.
- The UART protocol is 784 signed INT16 little-endian input samples, then
  ten signed INT16 little-endian output scores and one 32-bit little-endian
  hardware phase-cycle counter.
- The first `NUM_LINEAR - 1` outputs fill the pipeline and are discarded by
  the host. The host appends the same number of flush input frames.
- Default top UART rate is 2,000,000 baud. Match the PC command baud rate.

Set the same selectors in notebooks 02 and 03:

```python
'name': 'fashion_mnist_q16',
'target': 'nexys_a7',
'architecture': 'pipeline',
```

Run notebook 02 to export the quantized weights and generate RTL. Alternatively
run the generator explicitly:

```bash
python src/tcl/remodel_nexys_project.py --repo-root . \
  --layout data/quantized_data/fashion_mnist_q16/nexys_a7/pipeline/mem_layout.json \
  --fpga-mode pipeline
```

Generated RTL: `src/rtl/generated/src_nexys_pipeline/`

Generated Vivado script: `vivado/create_soc_mlp_project_nexys_pipeline.tcl`

Vivado project: `vivado/build/SOC_MLP_nexys_a7_pipeline/`

PC test after programming pipeline bitstream:

```bash
python notebooks/04_push_data_nexys.py --fpga-mode pipeline --port COM5 --baud 2000000 --max-samples 100
```

The reported hardware cycle count is the **pipeline phase duration**, not
the complete first-image latency. The Python test remains stop-and-wait; it
does not benchmark a fully streamed multi-frame transport.

The generated SV and Vivado build have **not** yet been compiled, simulated
or tested on the physical Nexys board. Confirm exact score vectors, frame
alignment, resource usage and timing before treating it as validated.
