import time
import unittest

from utilities.perf_monitor import PerfMonitor


class TestPerfMonitor(unittest.TestCase):
    def test_format_duration(self):
        """Performs basic validation of the _format_duration method."""

        monitor = PerfMonitor("test")

        self.assertEqual(monitor._format_duration(1), "1.00 ns")
        self.assertEqual(monitor._format_duration(2500), "2.50 μs")
        self.assertEqual(monitor._format_duration(3000000), "3.00 ms")
        self.assertEqual(monitor._format_duration(1230000000), "1.23 s")
        self.assertEqual(monitor._format_duration(120000000000), "2.00 min")

    def test_perf_monitor(self):
        """Performs basic validation of the PerfTracker and its output_str."""

        tracker = PerfMonitor("TestOfTime")

        self.assertRegex(tracker.summary_str(), "TestOfTime performance: N=0")

        with tracker.sample():
            time.sleep(1)

        self.assertRegex(
            tracker.summary_str(),
            r"TestOfTime performance: N=1 \| Min=1.[0-9]{2} s \| Max=1.[0-9]{2} s \| Median=1.[0-9]{2} s \| P90=1.[0-9]{2} s",
        )

        with tracker.sample():
            time.sleep(4)

        self.assertRegex(
            tracker.summary_str(),
            r"TestOfTime performance: N=2 \| Min=1.[0-9]{2} s \| Max=4.[0-9]{2} s \| Median=2.[0-9]{2} s \| P90=3.[0-9]{2} s",
        )

    def test_perf_monitor_min(self):
        """Performs basic validation of the min method."""

        tracker = PerfMonitor("TestMin")
        tracker.set_samples_for_testing([1_000_000_000, 4_000_000_000])

        self.assertEqual(tracker.min(), 1.0)

    def test_perf_monitor_max(self):
        """Performs basic validation of the max method."""
        tracker = PerfMonitor("TestMax")
        tracker.set_samples_for_testing([1_000_000_000, 4_000_000_000])
        self.assertEqual(tracker.max(), 4.0)

    def test_perf_monitor_median(self):
        """Performs basic validation of the median method."""
        tracker = PerfMonitor("TestMedian")
        tracker.set_samples_for_testing([1_000_000_000, 2_000_000_000, 3_000_000_000])
        self.assertEqual(tracker.median(), 2.0)

    def test_perf_monitor_percentile(self):
        """Performs basic validation of the percentile method."""
        tracker = PerfMonitor("TestPercentile")
        tracker.set_samples_for_testing(
            [1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000]
        )
        self.assertEqual(tracker.percentile(90), 3.7)


if __name__ == "__main__":
    unittest.main()
