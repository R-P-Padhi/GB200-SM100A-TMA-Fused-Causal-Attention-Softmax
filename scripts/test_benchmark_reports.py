"""Benchmark parser tests. All sample numbers below are synthetic fixtures."""
import unittest

from summarize_bench import parse_log, parse_record


class BenchmarkParserTests(unittest.TestCase):
    def test_cpu_record(self):
        record = parse_record(
            "model=cpu_reference pattern=random kernel=cpu_reference rows=64 cols=128 "
            "n=5 min_ms=0.1 median_ms=0.2 p95_ms=0.3 p99_ms=0.3 "
            "stddev_ms=0.04 cv_pct=20.0 approx_logical_GBps=1.2"
        )
        self.assertEqual(record["source"], "cpu")
        self.assertEqual(record["rows"], 64)
        self.assertEqual(record["p99_ms"], 0.3)

    def test_cuda_event_record(self):
        record = parse_record(
            "BENCH category=shape clock=cuda_event kernel=online rows=512 cols=512 "
            "n=40 min_ms=0.01 median_ms=0.02 p95_ms=0.03 p99_ms=0.04"
        )
        self.assertEqual(record["category"], "shape")
        self.assertEqual(record["source"], "gpu")
        self.assertEqual(record["median_ms"], 0.02)

    def test_concurrency_record(self):
        record = parse_record(
            "BENCH category=concurrency clock=host_wall streams=8 calls_per_stream=4 "
            "kernel=online rows=512 cols=512 n=20 batch_p50_ms=2.5 "
            "batch_p95_ms=3.5 batch_p99_ms=4.0 kernels_per_s=12800.0 "
            "throughput_speedup=3.2"
        )
        self.assertEqual(record["streams"], 8)
        self.assertEqual(record["throughput_speedup"], 3.2)

    def test_headings_are_not_measurements(self):
        self.assertIsNone(parse_record("BENCH hardware=GB200 warning=no_counters"))
        self.assertIsNone(parse_record("Timing: host wall-clock"))
        self.assertIsNone(parse_record("BENCH complete: CUDA events are device timing"))

    def test_fails_closed_on_empty(self):
        with self.assertRaisesRegex(ValueError, "no timing records"):
            parse_log("hello\n")

    def test_rejects_negative_and_nan(self):
        for value in ["-1", "nan", "inf"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_record(
                    f"BENCH category=shape kernel=online rows=32 cols=32 median_ms={value}"
                )

    def test_requires_shape_and_kernel(self):
        with self.assertRaises(ValueError):
            parse_record("BENCH category=shape rows=32 cols=32 median_ms=1")
        with self.assertRaises(ValueError):
            parse_record("BENCH category=shape kernel=online rows=0 cols=32 median_ms=1")


if __name__ == "__main__":
    unittest.main()
