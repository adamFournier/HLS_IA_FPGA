#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Send Fashion-MNIST INT16 images to a Nexys A7 FPGA over UART.

Standalone adaptation of 04_push_data_nexys.py, preserving the UART framing,
serial helpers and CSV reporting. Target RTL: sequential 784 -> 64 -> 32 -> 10.

PC -> FPGA: 784 signed int16 pixels, little-endian, no frame delimiter.
FPGA -> PC: 10 signed int16 class scores, little-endian, no delimiter.
The predicted class is argmax of the 10 received scores.

Prerequisite: Program the Nexys A7 with the matching Vivado bitstream.
First test: python notebooks/04_push_data_nexys.py --list-ports
            python notebooks/04_push_data_nexys.py --port COM5 --max-samples 1
"""

from __future__ import annotations

import argparse
import os
import re
import struct
import sys
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import serial
    import serial.tools.list_ports
except ImportError as exc:
    raise SystemExit("Missing pyserial module. Install it with: pip install pyserial") from exc


DEFAULT_INPUT_WORDS = 784
DEFAULT_OUTPUT_WORDS = 10
DEFAULT_NUM_LINEAR = 1

# ============================================================
# USER CONFIGURATION
# ============================================================

# Set PROJECT_ROOT_OVERRIDE to an absolute Path if automatic
# project-root detection is not suitable on your machine.
PROJECT_ROOT_OVERRIDE: Optional[Path] = None

# Dataset selection.
# To switch activation folders, normally only change DATASET_PROFILE.
DATASET_BASE_DIR = Path("data/quantized_data")
DATASET_PROFILE = "fashion_mnist_q16/nexys_a7/sequential"
DATASET_FILENAME = "fashion_mnist_quantized.npz"

# FPGA / generated RTL selection.
FPGA_TARGET = "nexys_a7"
DEFAULT_FPGA_MODE = "sequential"
RTL_GENERATED_DIR = Path("src/rtl/generated")
SCALES_FILENAME = "scales.vh"

# Results.
RESULT_DIR = Path("Result")


# ============================================================
# Project paths
# ============================================================

def script_path() -> Path:
    """Return the absolute path of this script."""
    try:
        return Path(__file__).resolve()
    except NameError:
        return Path.cwd().resolve() / "04_push_data_nexys.py"


def candidate_roots() -> list[Path]:
    """Possible project roots, used only when PROJECT_ROOT_OVERRIDE is None."""
    sp = script_path()
    cwd = Path.cwd().resolve()

    roots = [
        cwd,
        cwd.parent,
        sp.parent,
        sp.parent.parent,
        sp.parent.parent.parent,
    ]

    unique: list[Path] = []
    seen: set[Path] = set()

    for root in roots:
        root = root.resolve()
        if root not in seen:
            unique.append(root)
            seen.add(root)

    return unique


def resolve_project_root(explicit: Optional[Path] = None) -> Path:
    """Resolve the repository/project root in a deterministic order.

    Priority:
      1. --project-root
      2. PROJECT_ROOT_OVERRIDE
      3. automatic detection of a directory containing both src/ and data/
    """
    if explicit is not None:
        root = explicit.expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"--project-root is not a directory: {root}")
        return root

    if PROJECT_ROOT_OVERRIDE is not None:
        root = Path(PROJECT_ROOT_OVERRIDE).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(
                f"PROJECT_ROOT_OVERRIDE is not a directory: {root}"
            )
        return root

    for root in candidate_roots():
        if (root / "src").is_dir() and (root / "data").is_dir():
            return root

    searched = "\n".join(f"  - {p}" for p in candidate_roots())
    raise FileNotFoundError(
        "Unable to determine the project root automatically.\n"
        "Expected a directory containing both src/ and data/.\n"
        f"Searched:\n{searched}\n\n"
        "Either set PROJECT_ROOT_OVERRIDE at the top of the script "
        "or use --project-root."
    )


def configured_dataset_path(project_root: Path, fpga_mode: str = DEFAULT_FPGA_MODE) -> Path:
    """Dataset selected by the USER CONFIGURATION block."""
    profile = DATASET_PROFILE
    if fpga_mode == 'pipeline' and profile.endswith('/sequential'):
        profile = profile[:-len('sequential')] + 'pipeline'
    return (project_root / DATASET_BASE_DIR / profile / DATASET_FILENAME).resolve()


def resolve_dataset_path(
    explicit_dataset: Optional[Path],
    dataset_dir: Optional[Path],
    project_root: Path,
    fpga_mode: str = DEFAULT_FPGA_MODE,
) -> Path:
    """Resolve the dataset path without ambiguous recursive searching.

    Priority:
      1. --dataset FILE
      2. --dataset-dir DIR
      3. USER CONFIGURATION:
           DATASET_BASE_DIR / DATASET_PROFILE / DATASET_FILENAME

    With --dataset-dir, both of these are accepted:
      DIR/fashion_mnist_quantized.npz
      DIR/fashion_mnist_quantized.npz
    """
    if explicit_dataset is not None:
        dataset = explicit_dataset.expanduser().resolve()

        if not dataset.is_file():
            raise FileNotFoundError(
                f"--dataset does not point to an existing file:\n  {dataset}"
            )

        return dataset

    if dataset_dir is not None:
        directory = dataset_dir.expanduser().resolve()

        if not directory.is_dir():
            raise NotADirectoryError(
                f"--dataset-dir does not point to a directory:\n  {directory}"
            )

        candidates = [
            directory / DATASET_FILENAME,
            directory / DATASET_PROFILE / DATASET_FILENAME,
        ]

        for dataset in candidates:
            if dataset.is_file():
                return dataset.resolve()

        expected = "\n".join(f"  - {p}" for p in candidates)
        raise FileNotFoundError(
            "--dataset-dir was provided, but the configured dataset "
            "was not found.\n"
            f"Expected one of:\n{expected}\n"
            "Edit DATASET_PROFILE / DATASET_FILENAME at the top of the script "
            "or pass --dataset explicitly."
        )

    dataset = configured_dataset_path(project_root, fpga_mode)

    if not dataset.is_file():
        raise FileNotFoundError(
            "Configured dataset not found.\n"
            f"Expected:\n  {dataset}\n\n"
            "Current configuration:\n"
            f"  DATASET_BASE_DIR = {DATASET_BASE_DIR}\n"
            f"  DATASET_PROFILE  = {DATASET_PROFILE!r}\n"
            f"  DATASET_FILENAME = {DATASET_FILENAME!r}\n\n"
            "Change these values at the top of the script, "
            "or use --dataset."
        )

    return dataset


def configured_scales_path(project_root: Path, fpga_mode: str) -> Path:
    """Return the scales.vh path implied by target + FPGA mode."""
    generated_target = "nexys" if FPGA_TARGET == "nexys_a7" else FPGA_TARGET
    generated_subdir = f"src_{generated_target}_{fpga_mode}"
    return (
        project_root
        / RTL_GENERATED_DIR
        / generated_subdir
        / SCALES_FILENAME
    ).resolve()


def resolve_scales_vh(
    explicit: Optional[Path],
    project_root: Path,
    fpga_mode: str,
) -> Optional[Path]:
    """Resolve scales.vh.

    --scales has priority. Otherwise use the deterministic generated-RTL path.
    Missing scales.vh is not fatal because the script can infer dimensions
    from the dataset, but the exact expected path is printed.
    """
    if explicit is not None:
        scales = explicit.expanduser().resolve()

        if not scales.is_file():
            raise FileNotFoundError(
                f"--scales does not point to an existing file:\n  {scales}"
            )

        return scales

    scales = configured_scales_path(project_root, fpga_mode)

    if scales.is_file():
        return scales

    print(
        "WARNING: configured scales.vh was not found:\n"
        f"  {scales}\n"
        "Dataset dimensions will be used where possible."
    )
    return None


def resolve_out_dir(explicit: Optional[Path], project_root: Path) -> Path:
    """Resolve output directory."""
    if explicit is not None:
        return explicit.expanduser().resolve()

    return (project_root / RESULT_DIR).resolve()


# ============================================================
# SystemVerilog define helpers
# ============================================================

def parse_sv_define_int(path: Optional[Path], name: str) -> Optional[int]:
    if path is None or not path.exists():
        return None

    text = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(rf"`define\s+{re.escape(name)}\s+(-?\d+)", text)
    return int(m.group(1)) if m else None


def read_num_linear(scales_path: Optional[Path], default: int = DEFAULT_NUM_LINEAR) -> int:
    value = parse_sv_define_int(scales_path, "NUM_LINEAR")
    return int(default) if value is None else int(value)


def infer_words_from_scales(scales_path: Optional[Path]) -> Tuple[int, int]:
    n0 = parse_sv_define_int(scales_path, "N0")
    num_linear = read_num_linear(scales_path, DEFAULT_NUM_LINEAR)

    nout = None
    if num_linear is not None:
        nout = parse_sv_define_int(scales_path, f"N{num_linear}")

    # Fallback for fixed six-layer exports that define N6 directly.
    if nout is None:
        nout = parse_sv_define_int(scales_path, "N6")

    return (
        DEFAULT_INPUT_WORDS if n0 is None else int(n0),
        DEFAULT_OUTPUT_WORDS if nout is None else int(nout),
    )


# ============================================================
# Dataset helpers
# ============================================================

def pick_npz_array(ds, preferred_names, ndim=None, required=True):
    for name in preferred_names:
        if name in ds.files:
            arr = ds[name]
            if ndim is None or arr.ndim == ndim:
                return name, arr
    if required:
        raise KeyError(
            f"Missing NPZ key. Tried: {preferred_names}. "
            f"Available keys: {list(ds.files)}"
        )
    return None, None


def get_npz_scalar(ds, names, default=None):
    for name in names:
        if name in ds.files:
            return int(np.asarray(ds[name]).reshape(-1)[0])
    return default


def q_to_float(x_int: np.ndarray, scale_exp: int) -> np.ndarray:
    return np.asarray(x_int, dtype=np.float64) * (2.0 ** (-int(scale_exp)))


def load_dataset(dataset_npz: Path) -> dict:
    """Read the Fashion-MNIST NPZ produced by src/py/quantization.py.

    Expected shapes:
      x_int   : (N, 784) signed INT16 pixels
      y       : (N,) class labels 0..9
      pred_int: (N, 10) optional software reference class scores
    """
    with np.load(dataset_npz, allow_pickle=False) as ds:
        print("Available NPZ keys:", list(ds.files))
        x_int_key, x_raw = pick_npz_array(
            ds, ["x_int", "X_int", "inputs_int", "input_int"], ndim=2
        )
        x_float_key, x_float = pick_npz_array(
            ds, ["x", "x_float", "X_float"], ndim=2, required=False
        )
        label_key, labels = pick_npz_array(
            ds, ["y", "labels", "targets", "y_true"], ndim=1, required=True
        )
        pred_int_key, pred_int = pick_npz_array(
            ds, ["pred_board_int", "pred_int", "prediction_int"],
            ndim=2, required=False
        )
        pred_float_key, pred_float = pick_npz_array(
            ds, ["pred_float", "pred_board_float", "prediction_float"],
            ndim=2, required=False
        )
        input_exp = get_npz_scalar(ds, ["input_scale_exp", "input_exp", "x_scale_exp"])
        output_exp = get_npz_scalar(ds, ["pred_scale_exp", "output_scale_exp", "output_exp"])

        raw_pixels = np.asarray(x_raw)
        if raw_pixels.size == 0 or raw_pixels.ndim != 2:
            raise ValueError("x_int must be a nonempty 2-D array")
        if not np.issubdtype(raw_pixels.dtype, np.integer):
            raise ValueError("x_int must contain quantized integer pixels")
        if np.any(raw_pixels < -32768) or np.any(raw_pixels > 32767):
            raise ValueError("x_int contains pixels outside signed INT16 range")
        x = raw_pixels.astype(np.int16)
        y = np.asarray(labels, dtype=np.int64).copy()
        if y.shape != (len(x),):
            raise ValueError("y must be a 1-D class label per input image")
        if np.any(y < 0) or np.any(y > 9):
            raise ValueError("Fashion-MNIST labels must be in [0, 9]")
        if x.shape[1] != 784:
            raise ValueError(f"Expected Fashion-MNIST vectors of 784 pixels; got {x.shape[1]}")

        reference = None
        if pred_int is not None:
            pred_data = np.asarray(pred_int)
            if pred_data.shape != (len(x), 10):
                raise ValueError(f"pred_int must have shape ({len(x)}, 10), got {pred_data.shape}")
            if np.any(pred_data < -32768) or np.any(pred_data > 32767):
                raise ValueError("pred_int contains values outside signed INT16 range")
            reference = pred_data.astype(np.int16).copy()

        float_reference = None
        if pred_float is not None:
            float_data = np.asarray(pred_float, dtype=np.float32)
            if float_data.shape != (len(x), 10):
                raise ValueError(f"pred_float must have shape ({len(x)}, 10)")
            float_reference = float_data.copy()
        x_float_copy = np.asarray(x_float, dtype=np.float32).copy() if x_float is not None else None
        if x_float_copy is not None and x_float_copy.shape != x.shape:
            raise ValueError("Float and quantized input data disagree in shape")

    print(f"Input    : {x_int_key}, shape={x.shape}, dtype={x.dtype}")
    print(f"Labels   : {label_key}, shape={y.shape}, classes=0..9")
    print(f"Reference: {pred_int_key}, shape={None if reference is None else reference.shape}")
    print(f"Float ref: {pred_float_key}")
    print(f"Exp input/output: {input_exp} / {output_exp}")
    return {
        "X": x,
        "X_float": x_float_copy,
        "Y_labels": y,
        "Y_pred_int_ref": reference,
        "Y_pred_float_ref": float_reference,
        "input_scale_exp": input_exp,
        "pred_scale_exp": output_exp,
    }


# ============================================================
# UART helpers
# ============================================================

def ensure_int16_values(values: Iterable[int], expected_len: Optional[int] = None) -> list[int]:
    vals = [int(v) for v in values]
    if expected_len is not None and len(vals) != int(expected_len):
        raise ValueError(f"Expected {expected_len} int16 values, got {len(vals)}: {vals}")
    bad = [v for v in vals if v < -32768 or v > 32767]
    if bad:
        raise ValueError(f"Values outside int16 range [-32768, 32767]: {bad}")
    return vals


def pack_int16_words(values: Iterable[int], expected_len: int) -> bytes:
    vals = ensure_int16_values(values, expected_len=expected_len)
    return struct.pack("<" + f"{expected_len}h", *vals)


def unpack_int16_words(payload: bytes, expected_len: int) -> list[int]:
    expected_bytes = 2 * int(expected_len)
    if len(payload) != expected_bytes:
        raise ValueError(f"Invalid payload: got {len(payload)} bytes instead of {expected_bytes}")
    return list(struct.unpack("<" + f"{expected_len}h", payload))


def int16_words_hex(values: Iterable[int]) -> str:
    """Return int16 words as 4-hex-digit unsigned words, e.g. -3243 -> f355."""
    return " ".join(f"{(int(v) & 0xFFFF):04x}" for v in values)


def list_serial_ports() -> None:
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        print("No serial port detected.")
        return
    print("Detected serial ports:")
    for p in ports:
        print(f"  {p.device:16s} {p.description}")


def drain_uart_input(ser: serial.Serial, quiet_time_s: float = 0.2) -> bytes:
    junk = bytearray()
    deadline = time.time() + float(quiet_time_s)
    while time.time() < deadline:
        chunk = ser.read(1024)
        if chunk:
            junk.extend(chunk)
            deadline = time.time() + float(quiet_time_s)
    return bytes(junk)


def open_uart(port: str, baud: int, timeout_s: float, startup_delay_s: float) -> serial.Serial:
    try:
        ser = serial.Serial(
            port=port,
            baudrate=baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.05,
            write_timeout=timeout_s,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
        )
    except serial.SerialException as exc:
        print(f"Cannot open {port}: {exc}", file=sys.stderr)
        if "Permission denied" in str(exc):
            print("\nPossible Linux fix: sudo usermod -a -G dialout $USER then log out/in.", file=sys.stderr)
        print("\nAvailable ports:", file=sys.stderr)
        list_serial_ports()
        raise

    ser.reset_input_buffer()
    ser.reset_output_buffer()

    if startup_delay_s > 0:
        time.sleep(float(startup_delay_s))

    junk = drain_uart_input(ser, quiet_time_s=0.1)
    if junk:
        print("Ignored bytes after open:", junk.hex(" "))

    ser.reset_input_buffer()
    ser.reset_output_buffer()

    print("UART port opened :", ser.port)
    print("Baudrate         :", ser.baudrate)
    print("UART format      : 8N1")
    return ser


def read_exact(ser: serial.Serial, n_bytes: int, timeout_s: float) -> bytes:
    deadline = time.time() + float(timeout_s)
    buf = bytearray()

    while len(buf) < n_bytes and time.time() < deadline:
        chunk = ser.read(n_bytes - len(buf))
        if chunk:
            buf.extend(chunk)

    if len(buf) != n_bytes:
        raise TimeoutError(
            f"RX timeout: got {len(buf)} bytes, expected {n_bytes}. "
            f"Received bytes: {bytes(buf).hex(' ')}"
        )

    return bytes(buf)


def transaction(
    ser: serial.Serial,
    values: Iterable[int],
    input_words: int,
    output_words: int,
    timeout_s: float,
    reset_input_buffer: bool = True,
) -> tuple[list[int], bytes, bytes, int]:
    tx_values = ensure_int16_values(values, expected_len=input_words)
    tx_bytes = pack_int16_words(tx_values, input_words)
    rx_nbytes = 2 * int(output_words) + 4  # includes compute-cycle counter

    # This script keeps the known-good transaction style:
    # one TX frame, then one RX frame.
    # Since the UART protocol has no SOF/EOF, clearing old RX bytes before
    # sending a new frame avoids mixing a stale response with the next one.
    # Do not clear the input buffer between write() and read_exact().
    if reset_input_buffer:
        n_waiting = ser.in_waiting
        # if n_waiting:
        #     print(f"WARNING: RX buffer had {n_waiting} byte(s) before TX; clearing them.")
        ser.reset_input_buffer()

    ser.write(tx_bytes)
    ser.flush()

    rx_bytes = read_exact(ser, rx_nbytes, timeout_s)
    rx_values = unpack_int16_words(rx_bytes[:-4], output_words)
    cycles = struct.unpack("<I", rx_bytes[-4:])[0]

    return rx_values, tx_bytes, rx_bytes, cycles


def warmup_transactions(
    ser: serial.Serial,
    n: int,
    input_words: int,
    output_words: int,
    timeout_s: float,
) -> None:
    if n <= 0:
        return

    dummy = [0] * input_words
    print(f"UART warmup: {n} ignored dummy transaction(s)")

    for k in range(n):
        try:
            rx_values, _, rx_bytes, cycles = transaction(
                ser,
                dummy,
                input_words=input_words,
                output_words=output_words,
                timeout_s=timeout_s,
                reset_input_buffer=True,
            )
            print(f"  warmup {k + 1}/{n}: ignored RX bytes = {rx_bytes.hex(' ')} | RX int16 = {rx_values}")
        except TimeoutError as exc:
            print(f"  warmup {k + 1}/{n}: ignored timeout ({exc})")
            ser.reset_input_buffer()

    ser.reset_input_buffer()


# ============================================================
# CSV helpers
# ============================================================

def add_float_columns(cols: dict[str, object], prefix: str, arr, n_rows: int, width: int) -> None:
    if arr is None:
        for j in range(width):
            cols[f"{prefix}_{j}"] = np.full(n_rows, np.nan, dtype=np.float32)
    else:
        for j in range(width):
            cols[f"{prefix}_{j}"] = np.asarray(arr[:n_rows, j], dtype=np.float32)


def add_int_columns(cols: dict[str, object], prefix: str, arr, n_rows: int, width: int) -> None:
    if arr is None:
        for j in range(width):
            cols[f"{prefix}_{j}"] = np.full(n_rows, np.nan, dtype=np.float32)
    else:
        for j in range(width):
            cols[f"{prefix}_{j}"] = np.asarray(arr[:n_rows, j], dtype=np.int32)


# ============================================================
# Dataset execution
# ============================================================

def resolve_serial_port(explicit_port: Optional[str]) -> str:
    """Select a connected serial port; never guess a COM number."""
    if explicit_port:
        return explicit_port
    ports = list(serial.tools.list_ports.comports())
    if len(ports) == 1:
        print(f"Auto-detected serial port: {ports[0].device} ({ports[0].description})")
        return ports[0].device
    list_serial_ports()
    raise ValueError(
        "Please choose the Nexys UART port with --port COMx "
        "(use --list-ports to identify it)."
    )


def inference_only_cycles(
    raw_frame_cycles: Iterable[int],
    *,
    n_images: int,
    n_layers: int,
    fpga_mode: str,
) -> tuple[np.ndarray, int]:
    """Reconstruct inference-only timing from hardware counters.

    The sequential RTL returns the full compute time per image. The pipeline
    RTL returns one active global-phase duration per output frame. Image i
    traverses phases i .. i+n_layers-1, hence its inference-only latency is
    the sum of those measured phase durations. Pauses between phases (notably
    UART input/output and host round trips) are deliberately excluded.

    Total batch inference time includes pipeline fill/drain phases and is
    therefore the sum of ALL received phase durations, not the sum of the
    overlapping per-image latencies.
    """
    measured = np.asarray(list(raw_frame_cycles), dtype=np.uint64)
    if n_images <= 0 or n_layers <= 0:
        raise ValueError("n_images and n_layers must be positive")

    if fpga_mode == "pipeline":
        expected = n_images + n_layers - 1
        if measured.size != expected:
            raise ValueError(f"Expected {expected} pipeline counters, got {measured.size}")
        image_cycles = np.asarray(
            [np.sum(measured[i : i + n_layers], dtype=np.uint64)
             for i in range(n_images)],
            dtype=np.uint64,
        )
    elif fpga_mode == "sequential":
        if measured.size != n_images:
            raise ValueError(f"Expected {n_images} sequential counters, got {measured.size}")
        image_cycles = measured.copy()
    else:
        raise ValueError(f"Unknown FPGA mode: {fpga_mode}")

    batch_cycles = int(np.sum(measured, dtype=np.uint64))
    if batch_cycles <= 0:
        raise ValueError("All FPGA cycle counters are zero: cannot measure inference")
    return image_cycles, batch_cycles


def run_dataset(args: argparse.Namespace) -> int:
    if args.progress_every <= 0:
        raise ValueError("--progress-every must be positive")
    if args.max_samples == 0:
        raise ValueError("--max-samples must be positive, or -1 for all samples")

    root = resolve_project_root(args.project_root)
    dataset_npz = resolve_dataset_path(args.dataset, args.dataset_dir, root, args.fpga_mode)
    scales_path = resolve_scales_vh(args.scales, root, args.fpga_mode)
    if scales_path is None:
        raise FileNotFoundError(
            f"Generated Nexys scales.vh missing: run quantization for nexys_a7/{args.fpga_mode} first"
        )
    inferred_input_words, inferred_output_words = infer_words_from_scales(scales_path)
    data = load_dataset(dataset_npz)
    x = data["X"]
    labels = data["Y_labels"]
    pred_ref = data["Y_pred_int_ref"]
    pred_float = data["Y_pred_float_ref"]
    input_size, output_size = x.shape[1], 10

    input_words = input_size if args.input_words is None else int(args.input_words)
    output_words = output_size if args.output_words is None else int(args.output_words)
    if (input_words, output_words) != (784, 10):
        raise ValueError("The current Fashion-MNIST design requires exactly 784 input and 10 output INT16 words")
    if (inferred_input_words, inferred_output_words) != (784, 10):
        raise ValueError(
            f"scales.vh topology mismatch: N0={inferred_input_words}, "
            f"N{read_num_linear(scales_path)}={inferred_output_words}; expected 784 and 10"
        )

    n_test = len(x) if args.max_samples is None or args.max_samples < 0 else min(args.max_samples, len(x))
    if n_test == 0:
        raise ValueError("No dataset samples to transmit")
    x_test = x[:n_test]
    labels_test = labels[:n_test]
    num_layers = read_num_linear(scales_path)
    pipeline_delay = max(0, num_layers - 1) if args.fpga_mode == "pipeline" else 0
    if args.fpga_mode == "pipeline" and args.warmup_transactions:
        raise ValueError("Pipeline alignment requires --warmup-transactions 0")
    if args.fpga_mode == "pipeline":
        flush = np.full((pipeline_delay, input_size), args.flush_value, dtype=np.int16)
        x_stream = np.vstack([x_test, flush])
    else:
        x_stream = x_test
    n_stream = len(x_stream)

    print("\nNexys A7 / Fashion-MNIST UART configuration")
    print("  dataset        :", dataset_npz)
    print("  scales         :", scales_path)
    print("  topology       : 784 -> 64 -> 32 -> 10 (" + args.fpga_mode + ")")
    print("  pipeline_delay :", pipeline_delay, "frame(s)")
    print("  samples        :", n_test)
    print("  transmitted    :", n_stream, "frames, including flush")
    print("  bitrate        :", args.baud, "baud, 8N1")
    print("  bytes/image    :", input_words * 2, "TX,", output_words * 2 + 4, "RX (10 scores + 32-bit cycles)")
    print("  wire format    : signed INT16, little-endian, no delimiter")

    if args.dry_run:
        example = pack_int16_words(x_test[0], input_words)
        print("\nDRY RUN: FPGA communication not started")
        print("  first label    :", int(labels_test[0]))
        print("  first 16 pixels:", x_test[0, :16].tolist())
        print("  first 32 bytes :", example[:32].hex(" "))
        print("  byte count     :", len(example))
        return 0

    port = resolve_serial_port(args.port)
    out_dir = resolve_out_dir(args.out_dir, root)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_raw = np.zeros((n_stream, output_words), dtype=np.int16)
    tx_hex, rx_hex = [], []
    fpga_cycles = []

    started = time.time()
    with open_uart(port, args.baud, args.timeout, args.startup_delay) as ser:
        warmup_transactions(
            ser, n=args.warmup_transactions, input_words=input_words,
            output_words=output_words, timeout_s=args.timeout,
        )
        for i in range(n_stream):
            try:
                received, tx_bytes, rx_bytes, cycles = transaction(
                    ser, x_stream[i], input_words=input_words,
                    output_words=output_words, timeout_s=args.timeout,
                    reset_input_buffer=args.reset_input_buffer_before_tx,
                )
            except TimeoutError as exc:
                raise TimeoutError(
                    f"Sample {i}: {exc}. Check that the Nexys bitstream is programmed, "
                    "the UART port is correct, baud rates match, and reset completed."
                ) from exc
            y_raw[i] = received
            tx_hex.append(tx_bytes.hex())
            rx_hex.append(rx_bytes.hex())
            fpga_cycles.append(cycles)
            if i < args.preview:
                if args.fpga_mode == "sequential":
                    pred = int(np.argmax(y_raw[i]))
                    msg = f"image {i}: label={labels_test[i]}, FPGA_class={pred}, scores={y_raw[i].tolist()}"
                    if pred_ref is not None:
                        msg += f", reference_class={int(np.argmax(pred_ref[i]))}"
                    msg += f", MLP_compute={cycles / 100_000:.3f} ms ({cycles} cycles at 100 MHz)"
                    print(msg)
                else:
                    tag = "flush" if i >= n_test else "input"
                    print(f"frame {i}: {tag}, raw_class={int(np.argmax(y_raw[i]))}, phase_cycles={cycles} (one phase only)")
            if (i + 1) % args.progress_every == 0 or i + 1 == n_stream:
                print(f"  {i + 1}/{n_stream} stream frames processed")
    elapsed = time.time() - started

    # The FPGA sends one counter per output frame. In pipeline mode these
    # counters describe *phases*, not complete inference for that frame.
    raw_phase_cycles = np.asarray(fpga_cycles, dtype=np.uint64)
    image_inference_cycles, batch_inference_cycles = inference_only_cycles(
        raw_phase_cycles, n_images=n_test, n_layers=num_layers,
        fpga_mode=args.fpga_mode,
    )
    image_inference_ms = image_inference_cycles.astype(np.float64) / 100_000.0
    batch_inference_ms = batch_inference_cycles / 100_000.0
    batch_throughput = n_test * 100_000_000.0 / batch_inference_cycles
    aligned_phase_cycles = raw_phase_cycles[pipeline_delay:pipeline_delay + n_test]

    y_fpga = y_raw[pipeline_delay:pipeline_delay + n_test]
    rx_hex = rx_hex[pipeline_delay:pipeline_delay + n_test]
    tx_hex = tx_hex[:n_test]
    fpga_classes = np.argmax(y_fpga, axis=1).astype(np.int64)
    correct = fpga_classes == labels_test

    print("\n=== FPGA INFERENCE ONLY (UART and Python excluded) ===")
    print(f"Image latency : {np.mean(image_inference_ms):.3f} ms/image "
          f"(min={np.min(image_inference_ms):.3f}, max={np.max(image_inference_ms):.3f})")
    print(f"Total inference time ({n_test} images): {batch_inference_ms:.3f} ms "
          f"({batch_inference_cycles} active clock cycles)")
    print(f"Batch throughput ({n_test} images): {batch_throughput:.1f} images/s")
    if args.fpga_mode == "pipeline":
        mean_phase_cycles = float(np.mean(raw_phase_cycles))
        steady_throughput = 100_000_000.0 / mean_phase_cycles
        print(f"Pipeline steady-state throughput: {steady_throughput:.1f} images/s")
        print(f"Measured average pipeline phase: {mean_phase_cycles / 100_000.0:.3f} ms")
        print("NOTE: per-image latency is reconstructed from the measured phase "
              "counters with UART-induced idle gaps removed; it is not a "
              "wall-clock entry-to-exit measurement.")
    else:
        steady_throughput = batch_throughput
    print(f"\nFPGA classification accuracy: {correct.mean():.2%} ({correct.sum()}/{n_test})")
    print(f"UART + PC total elapsed (NOT inference-only): {elapsed:.2f} s")
    for i in range(min(args.preview, n_test)):
        print(f"image {i}: FPGA_class={int(fpga_classes[i])}, "
              f"inference_latency={image_inference_ms[i]:.3f} ms "
              f"({int(image_inference_cycles[i])} cycles)")

    ref_class, matches_ref, errors = None, None, None
    if pred_ref is not None:
        ref_subset = pred_ref[:n_test]
        ref_class = np.argmax(ref_subset, axis=1)
        matches_ref = fpga_classes == ref_class
        errors = y_fpga.astype(np.int32) - ref_subset.astype(np.int32)
        print(f"Class agreement vs INT16 software reference: {matches_ref.mean():.2%}")
        print(f"Exact 10-score vectors: {np.all(errors == 0, axis=1).sum()}/{n_test}")
        print(f"Max absolute score difference: {np.abs(errors).max()}")
    else:
        ref_subset = None
        print("No INT16 software reference found: only classification labels are compared")

    if pd is None:
        print("pandas not installed; CSV not written (pip install pandas)")
        return 0

    cols: dict[str, object] = {
        "sample": np.arange(n_test, dtype=np.int32),
        "target_class": labels_test,
        "fpga_predicted_class": fpga_classes,
        "fpga_correct": correct,
        "reference_predicted_class": ref_class if ref_class is not None else np.full(n_test, np.nan),
        "fpga_matches_reference": matches_ref if matches_ref is not None else np.full(n_test, np.nan),
    }
    add_float_columns(cols, "input_float", data["X_float"], n_test, input_size)
    add_int_columns(cols, "input_quantized", x_test, n_test, input_size)
    add_float_columns(cols, "pred_float", pred_float, n_test, output_size)
    add_int_columns(cols, "pred_quantized", ref_subset, n_test, output_size)
    add_int_columns(cols, "fpga_quantized", y_fpga, n_test, output_size)
    add_int_columns(cols, "error_fpga_pred_quantized", errors, n_test, output_size)
    exp = data["pred_scale_exp"]
    fpga_float = q_to_float(y_fpga, exp).astype(np.float32) if exp is not None else None
    add_float_columns(cols, "fpga_float", fpga_float, n_test, output_size)
    # Backward-compatible names now refer to full image inference, not a
    # single pipeline phase. Keep the raw counter in its own dedicated field.
    cols["mlp_cycles"] = image_inference_cycles
    cols["mlp_compute_ms"] = image_inference_ms
    cols["inference_latency_cycles"] = image_inference_cycles
    cols["inference_latency_ms"] = image_inference_ms
    cols["batch_inference_cycles"] = batch_inference_cycles
    cols["batch_inference_ms"] = batch_inference_ms
    cols["batch_throughput_images_per_s"] = batch_throughput
    cols["steady_state_throughput_images_per_s"] = steady_throughput
    cols["raw_output_phase_cycles"] = aligned_phase_cycles
    cols["tx_words_hex"] = tx_hex
    cols["rx_words_hex"] = rx_hex
    cols["transport"] = args.transport_label
    cols["fpga_mode"] = args.fpga_mode
    cols["pipeline_delay"] = pipeline_delay
    cols["raw_output_frame"] = np.arange(pipeline_delay, pipeline_delay + n_test)
    cols["mlp_timer_meaning"] = (
        "image_latency_reconstructed_sum_of_phase_active_cycles"
        if args.fpga_mode == "pipeline" else "image_compute_cycles"
    )
    cols["num_linear"] = read_num_linear(scales_path)
    cols["input_scale_exponent"] = (-1 if data["input_scale_exp"] is None else data["input_scale_exp"])
    cols["output_scale_exponent"] = (-1 if exp is None else exp)

    csv_path = (
        args.out_csv.expanduser().resolve()
        if args.out_csv is not None
        else out_dir / "soc_mlp_nexys_a7_fashion_mnist_results.csv"
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(cols)
    df.to_csv(csv_path, index=False)
    print("Results CSV:", csv_path)
    print(df[["sample", "target_class", "fpga_predicted_class", "fpga_correct",
              "reference_predicted_class", "fpga_matches_reference"]].head(10).to_string(index=False))

    if args.save_raw_csv:
        raw = pd.DataFrame({"sample": np.arange(n_test), "tx_words_hex": tx_hex, "rx_words_hex": rx_hex})
        for j in range(output_words):
            raw[f"fpga_score_{j}"] = y_fpga[:, j]
        raw_path = csv_path.with_name(csv_path.stem + "_raw.csv")
        raw.to_csv(raw_path, index=False)
        print("Raw UART CSV:", raw_path)
    return 0


# ============================================================
# Main
# ============================================================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Send Fashion-MNIST INT16 images to Nexys A7 MLP via UART."
    )

    # Project / generated files.
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help=(
            "Repository root. Default: PROJECT_ROOT_OVERRIDE or automatic "
            "detection of a directory containing src/ and data/."
        ),
    )
    parser.add_argument(
        "--fpga-mode",
        choices=["sequential", "pipeline"],
        default=DEFAULT_FPGA_MODE,
    )
    parser.add_argument(
        "--num-linear-fallback",
        type=int,
        default=DEFAULT_NUM_LINEAR,
    )
    parser.add_argument(
        "--scales",
        type=Path,
        default=None,
        help=(
            "Explicit scales.vh. Default is computed from "
            "FPGA_TARGET + --fpga-mode."
        ),
    )

    # UART.
    parser.add_argument("--port", default=None, help="UART serial port, e.g. COM5 on Windows; auto-select if only one port")
    parser.add_argument("--baud", type=int, default=115200, help="UART baudrate")
    parser.add_argument("--timeout", type=float, default=10.0, help="RX timeout in seconds")
    parser.add_argument("--startup-delay", type=float, default=0.5, help="Delay after opening the port")
    parser.add_argument("--list-ports", action="store_true", help="List serial ports then exit")

    # Dataset.
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help=(
            "Explicit quantized NPZ file. Overrides DATASET_PROFILE "
            "and DATASET_FILENAME."
        ),
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=None,
        help=(
            "Dataset directory. The script looks only for the configured "
            "DATASET_FILENAME, optionally under DATASET_PROFILE."
        ),
    )

    # Outputs.
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for CSV")
    parser.add_argument("--out-csv", type=Path, default=None, help="Complete output CSV path")

    # Frame dimensions.
    parser.add_argument(
        "--input-words",
        type=int,
        default=None,
        help="Default: dataset input width.",
    )
    parser.add_argument(
        "--output-words",
        type=int,
        default=None,
        help="Default: dataset output width.",
    )

    # Run control.
    parser.add_argument("--max-samples", type=int, default=10, help="Images to test (default: 10). -1 = full dataset")
    parser.add_argument("--preview", type=int, default=10, help="Number of first aligned samples to print")
    parser.add_argument("--progress-every", type=int, default=100, help="Print progress every N stream frames")

    parser.add_argument("--warmup-transactions", type=int, default=0, help="Optional ignored dummy transactions before dataset (default: 0)")
    parser.add_argument("--flush-value", type=int, default=0, help="Dummy value used for pipeline flush frames")
    parser.add_argument("--save-raw-csv", action="store_true", help="Also save raw stream TX/RX frames")

    parser.add_argument(
        "--reset-input-buffer-before-tx",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Clear stale RX bytes before each new TX frame.",
    )

    parser.add_argument(
        "--transport-label",
        default="uart_lsb_first_int16_no_sof_send_then_read",
        help="String stored in the CSV transport column.",
    )

    parser.add_argument("--dry-run", action="store_true", help="Verify and pack a sample without opening UART")
    args = parser.parse_args(argv)

    if args.list_ports:
        list_serial_ports()
        return 0

    # Print the user-facing configuration before any UART access.
    print("Configured defaults:")
    print("  DATASET_BASE_DIR =", DATASET_BASE_DIR)
    print("  DATASET_PROFILE  =", DATASET_PROFILE)
    print("  DATASET_FILENAME =", DATASET_FILENAME)
    print("  FPGA_TARGET      =", FPGA_TARGET)
    print("  DEFAULT_FPGA_MODE=", DEFAULT_FPGA_MODE)
    print("  RESULT_DIR       =", RESULT_DIR)

    return run_dataset(args)


if __name__ == "__main__":
    raise SystemExit(main())
