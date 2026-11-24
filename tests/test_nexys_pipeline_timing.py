"""Regression tests for Nexys A7 FPGA inference timing and UART packets.

Run with:
    py -m unittest discover -s tests -p test_nexys_pipeline_timing.py

These tests check Python parsing/math only; Vivado synthesis and real board
timing must still be validated on the Nexys A7.
"""
from __future__ import annotations

import importlib.util
import struct
import unittest
from pathlib import Path

import numpy as np


RUNNER = Path(__file__).resolve().parents[1] / "notebooks" / "04_push_data_nexys.py"
spec = importlib.util.spec_from_file_location("nexys_runner", RUNNER)
runner = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(runner)


class FakeSerial:
    def __init__(self, packet: bytes):
        self.packet = packet
        self.writes = []
        self.in_waiting = 0

    def reset_input_buffer(self):
        pass

    def write(self, data):
        self.writes.append(bytes(data))

    def flush(self):
        pass

    def read(self, count):
        chunk, self.packet = self.packet[:count], self.packet[count:]
        return chunk


class TestInferenceTiming(unittest.TestCase):
    def test_real_per_image_pipeline_latencies_and_batch(self):
        # Three layers: phase lengths 10,20,30,40,50 cycles. Image 0
        # completes 3 cycles into phase 2; image 1 completes 5 cycles
        # into phase 3; image 2 completes 7 cycles into phase 4.
        # The phase-end counter is deliberately different from the
        # image's measured completion time.
        image, batch = runner.inference_only_cycles(
            [0, 0, 33, 55, 77], [10, 20, 30, 40, 50],
            n_images=3, n_layers=3, fpga_mode="pipeline",
        )
        np.testing.assert_array_equal(image, [33, 55, 77])
        self.assertEqual(batch, 107)  # 10+20+30+40+7, not 150

    def test_sequential_is_unchanged(self):
        image, batch = runner.inference_only_cycles(
            [11, 13, 17], [11, 13, 17],
            n_images=3, n_layers=3, fpga_mode="sequential",
        )
        np.testing.assert_array_equal(image, [11, 13, 17])
        self.assertEqual(batch, 41)

    def test_pipeline_packet_uses_two_hardware_counters(self):
        packet = struct.pack("<10hII", *range(10), 100920, 50300)
        self.assertEqual(len(packet), 28)
        fake = FakeSerial(packet)
        scores, sent, received, latency, phase = runner.transaction(
            fake, [0] * 784, input_words=784, output_words=10,
            timeout_s=0.1, fpga_mode="pipeline",
        )
        self.assertEqual(scores, list(range(10)))
        self.assertEqual((latency, phase), (100920, 50300))
        self.assertEqual(len(sent), 784 * 2)
        self.assertEqual(received, packet)

    def test_sequential_packet_remains_24_bytes(self):
        packet = struct.pack("<10hI", *range(10), 1234)
        fake = FakeSerial(packet)
        scores, _sent, received, latency, phase = runner.transaction(
            fake, [0] * 784, input_words=784, output_words=10,
            timeout_s=0.1, fpga_mode="sequential",
        )
        self.assertEqual(scores, list(range(10)))
        self.assertEqual((latency, phase), (1234, 1234))
        self.assertEqual(received, packet)


if __name__ == "__main__":
    unittest.main()
