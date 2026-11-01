"""Unit tests for the toolkit. Run with:  python -m unittest -v   (or pytest)."""
import csv
import tempfile
import unittest
from pathlib import Path

from stats_toolkit import (mean, median, mode, variance, std_dev, quartiles,
                           iqr_outliers, load_column, StatisticsAnalyzer)


class TestCentralTendency(unittest.TestCase):
    def test_mean_of_known_values(self):
        self.assertAlmostEqual(mean([2, 4, 6, 8]), 5.0)

    def test_median_even_and_odd_lengths(self):
        self.assertEqual(median([3, 1, 2]), 2)          # odd  -> middle value
        self.assertEqual(median([4, 1, 3, 2]), 2.5)     # even -> average of middle two

    def test_mode_with_ties(self):
        self.assertEqual(mode([1, 1, 2, 2, 3]), [1, 2])
        self.assertEqual(mode([5, 5, 5, 1]), [5])

    def test_mode_none_when_all_values_appear_once(self):
        self.assertEqual(mode([1, 2, 3]), [])

    def test_empty_input_raises(self):
        for func in (mean, median, mode, variance, std_dev, quartiles, iqr_outliers):
            with self.assertRaises(ValueError):
                func([])


class TestSpread(unittest.TestCase):
    DATA = [2, 4, 4, 4, 5, 5, 7, 9]    # mean 5, sum of squared gaps 32

    def test_sample_vs_population_variance(self):
        self.assertAlmostEqual(variance(self.DATA, sample=False), 4.0)      # 32 / 8
        self.assertAlmostEqual(variance(self.DATA, sample=True), 32 / 7)    # 32 / 7

    def test_std_dev_population(self):
        self.assertAlmostEqual(std_dev(self.DATA, sample=False), 2.0)

    def test_std_dev_single_value(self):
        self.assertEqual(std_dev([5], sample=False), 0.0)
        with self.assertRaises(ValueError):
            std_dev([5], sample=True)       # n - 1 = 0, undefined

    def test_quartiles_known_values(self):
        self.assertEqual(quartiles([1, 2, 3, 4, 5, 6, 7, 8, 9]), (3, 5, 7))
        q1, q2, q3 = quartiles([1, 2, 3, 4])
        self.assertAlmostEqual(q1, 1.75)
        self.assertAlmostEqual(q2, 2.5)
        self.assertAlmostEqual(q3, 3.25)

class TestOutliers(unittest.TestCase):
    def test_no_outliers_in_uniform_data(self):
        self.assertEqual(iqr_outliers([10, 10, 10, 10, 10])["outliers"], [])
        self.assertEqual(iqr_outliers([1, 2, 3, 4, 5])["outliers"], [])

    def test_detects_obvious_outlier(self):
        result = iqr_outliers([10, 11, 12, 13, 14, 100])
        self.assertAlmostEqual(result["iqr"], 2.5)
        self.assertAlmostEqual(result["lower"], 7.5)
        self.assertAlmostEqual(result["upper"], 17.5)
        self.assertEqual(result["outliers"], [100])

    def test_detects_low_outlier(self):
        self.assertEqual(iqr_outliers([1, 50, 51, 52, 53, 54])["outliers"], [1])

class TestLoading(unittest.TestCase):
    def test_invalid_and_missing_values_are_rejected(self):
        rows = [["1", "A", "5"], ["2", "A", ""], ["3", "A", "abc"], ["4", "A", "-"],
                ["5", "A", "2.5"], ["6", "A", "12,40O"], ["7", "A", "-3"]]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "t.csv"
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["date", "agent_id", "amount"])
                writer.writerows(rows)
            values, rejected = load_column(path, "amount")
        self.assertEqual(values, [5.0, 2.5])
        self.assertEqual([n for n, _ in rejected], [3, 4, 5, 7, 8])   # header is row 1
        self.assertEqual(rejected[3][1], "12,40O")

class TestAnalyzer(unittest.TestCase):
    def test_without_outliers_returns_new_analyzer(self):
        analyzer = StatisticsAnalyzer([10, 11, 12, 13, 14, 100], "x")
        clean = analyzer.without_outliers()
        self.assertEqual(clean.summary()["count"], 5)
        self.assertEqual(analyzer.summary()["count"], 6)     # original untouched
        self.assertAlmostEqual(clean.summary()["mean"], 12.0)

if __name__ == "__main__":
    unittest.main()
