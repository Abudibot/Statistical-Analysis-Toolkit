"""
Statistical Analysis Toolkit
RULE: implement every calculation yourself with plain Python.
NumPy may be imported ONLY in compare_with_numpy(), to verify your results.
Do not use the statistics module.

Choices made:
  * Quartile method : linear interpolation, position = (n - 1) * p
                      (the same as NumPy's default method="linear").
  * Mode            : every value tied for the highest count; if every
                      value appears only once the answer is [] ("none").
  * Variance        : sample (n - 1) by default - the data is a sample of days.
  * Row numbers     : spreadsheet style. The header is row 1, first data row is 2.
"""
import csv
import io
import math
import unittest
from pathlib import Path

DATA_PATH = Path("data") / "agent_deposits.csv"
OUTPUT_PATH = Path("output") / "stats_report.txt"
PLACEHOLDERS = {"-", "n/a", "na", "null", "none", "nan"}

def _parse_value(raw: str):
    """Return (number, None) if raw is trustworthy, else (None, reason)."""
    text = raw.strip()
    if text == "":
        return None, "missing value"
    if text.lower() in PLACEHOLDERS:
        return None, "placeholder, not a number"
    if "_" in text:                      # Python would accept 1_000; we do not
        return None, "not a valid number"
    try:
        number = float(text)
    except ValueError:
        return None, "not a valid number"   # e.g. '12,40O' - we do NOT guess
    if not math.isfinite(number):
        return None, "not a finite number"
    if number < 0:
        return None, "negative deposit is impossible"
    return number, None

def _read_rows(path: Path, column: str):
    """Return (records, rejected). A record is (value, date, agent_id)."""
    records, rejected = [], []
    with open(path, newline="", encoding="utf-8-sig") as file:  
        reader = csv.DictReader(file)
        if column not in (reader.fieldnames or []):
            raise ValueError(f"column '{column}' not found in {path}")
        for row_number, row in enumerate(reader, start=2):   # header = row 1
            raw = row[column] or ""
            number, reason = _parse_value(raw)
            if reason:
                rejected.append((row_number, raw, reason))
            else:
                records.append((number, row.get("date", ""), row.get("agent_id", "")))
    return records, rejected

def load_column(path: Path, column: str) -> tuple[list[float], list[tuple[int, str]]]:
    """Return (valid_values, rejected) where rejected holds (row_number, raw_value) pairs."""
    records, rejected = _read_rows(path, column)
    return [r[0] for r in records], [(n, raw) for n, raw, _ in rejected]

def _require(values: list[float]) -> None:
    if len(values) == 0:
        raise ValueError("input is empty")

def mean(values: list[float]) -> float:
    _require(values)
    return sum(values) / len(values)

def _median_sorted(s: list[float]) -> float:
    """Median of data that is ALREADY sorted."""
    n = len(s)
    middle = n // 2
    if n % 2 == 1:
        return s[middle]
    return (s[middle - 1] + s[middle]) / 2

def median(values: list[float]) -> float:
    _require(values)
    return _median_sorted(sorted(values))

def mode(values: list[float]) -> list[float]:
    """Return every value that occurs most often (a list, because there can be ties or none).

    If every value appears exactly once there is no mode worth reporting -> [].
    """
    _require(values)
    counts = {}
    for v in values:
        counts[v] = counts.get(v, 0) + 1
    highest = max(counts.values())
    if highest == 1:
        return []
    return sorted(v for v, c in counts.items() if c == highest)

      # variance and standaard deviation
def variance(values: list[float], sample: bool = True) -> float:
    """Sample variance (n - 1) by default; population variance (n) when sample is False."""
    _require(values)
    n = len(values)
    if sample and n < 2:
        raise ValueError("sample variance needs at least 2 values")
    m = mean(values)
    squared_gaps = sum((v - m) ** 2 for v in values)
    return squared_gaps / (n - 1 if sample else n)

def std_dev(values: list[float], sample: bool = True) -> float:
    return math.sqrt(variance(values, sample))

     # quartiles and IQR outliers
def _percentile_sorted(s: list[float], p: float) -> float:
    """Linear interpolation between neighbours: position = (n - 1) * p."""
    position = (len(s) - 1) * p
    low = math.floor(position)
    high = math.ceil(position)
    return s[low] + (s[high] - s[low]) * (position - low)

def _quartiles_sorted(s: list[float]) -> tuple[float, float, float]:
    return _percentile_sorted(s, 0.25), _median_sorted(s), _percentile_sorted(s, 0.75)

def quartiles(values: list[float]) -> tuple[float, float, float]:
    """Return (Q1, Q2, Q3) using linear interpolation (position = (n-1)*p).

    This is NumPy's default method="linear", so compare_with_numpy uses the same one.
    """
    _require(values)
    return _quartiles_sorted(sorted(values))

def _outlier_info(s: list[float], k: float) -> dict:
    q1, _, q3 = _quartiles_sorted(s)
    iqr = q3 - q1
    lower, upper = q1 - k * iqr, q3 + k * iqr
    return {
        "iqr": iqr,
        "lower": lower,
        "upper": upper,
        "outliers": [v for v in s if v < lower or v > upper],
    }

def iqr_outliers(values: list[float], k: float = 1.5) -> dict:
    """Return the IQR, the lower and upper fences, and the values outside them."""
    _require(values)
    return _outlier_info(sorted(values), k)

    # the analyzer class
class StatisticsAnalyzer:
    """Wraps one numeric column and exposes every statistic above as a method."""

    def __init__(self, values: list[float], name: str = "values",
                 labels: list[tuple[str, str]] | None = None,
                 rejected: list[tuple[int, str, str]] | None = None) -> None:
        _require(values)
        self.name = name
        self.values = list(values)
        self.sorted_values = sorted(values)          # sorted ONCE, reused everywhere
        self.labels = labels or [("", "")] * len(values)   # (date, agent_id) per value
        self.rejected = rejected or []               # (row_number, raw, reason)
        self.rows_read = len(values) + len(self.rejected)
        self._summary = None                         # cache
        self._clean = None                           # cache

    @classmethod
    def from_csv(cls, path: Path, column: str) -> "StatisticsAnalyzer":
        records, rejected = _read_rows(path, column)
        values = [r[0] for r in records]
        labels = [(r[1], r[2]) for r in records]
        return cls(values, column, labels, rejected)

    def summary(self) -> dict:
        if self._summary is None:
            s = self.sorted_values
            q1, q2, q3 = _quartiles_sorted(s)
            fences = _outlier_info(s, 1.5)
            modes = mode(self.values)
            self._summary = {
                "count": len(s),
                "mean": mean(s),
                "median": q2,
                "mode": modes,
                "variance": variance(s) if len(s) > 1 else None,
                "std_dev": std_dev(s) if len(s) > 1 else None,
                "min": s[0],
                "max": s[-1],
                "q1": q1,
                "q3": q3,
                "iqr": fences["iqr"],
                "lower": fences["lower"],
                "upper": fences["upper"],
                "outliers": fences["outliers"],
            }
        return self._summary

    def outlier_records(self) -> list[tuple[float, str, str]]:
        """Outliers with their (date, agent_id), so the risk team can trace them."""
        info = self.summary()
        found = [(v, d, a) for v, (d, a) in zip(self.values, self.labels)
                 if v < info["lower"] or v > info["upper"]]
        return sorted(found)

    def without_outliers(self) -> "StatisticsAnalyzer":
        if self._clean is None:
            info = self.summary()
            kept = [(v, lab) for v, lab in zip(self.values, self.labels)
                    if info["lower"] <= v <= info["upper"]]
            self._clean = StatisticsAnalyzer([v for v, _ in kept], self.name,
                                             [lab for _, lab in kept])
        return self._clean

         #  report 
    def report(self) -> str:
        a = self.summary()
        b = self.without_outliers().summary()

        def num(x):
            return "n/a" if x is None else f"{x:,.2f}"

        def mode_text(info):
            return ", ".join(f"{m:,.2f}" for m in info["mode"]) if info["mode"] else "none"

        def row(label, left, right):
            return f"{label:<20}{left:<26}{right}"

        lines = [f"=== Statistics report: {self.name} ===",
                 f"Rows read         : {self.rows_read}",
                 f"Valid values      : {len(self.values)}",
                 f"Rejected values   : {len(self.rejected)}"]
        for row_number, raw, reason in self.rejected:
            lines.append(f"  row {row_number}: '{raw}' ({reason})")
        lines += [
            "",
            row("", "all data", "outliers removed"),
            row("count", a["count"], b["count"]),
            row("mean", num(a["mean"]), num(b["mean"])),
            row("median", num(a["median"]), num(b["median"])),
            row("mode", mode_text(a), mode_text(b)),
            row("variance (sample)", num(a["variance"]), num(b["variance"])),
            row("std dev (sample)", num(a["std_dev"]), num(b["std_dev"])),
            row("min / max", f"{num(a['min'])} / {num(a['max'])}",
                f"{num(b['min'])} / {num(b['max'])}"),
            row("Q1 / Q3", f"{num(a['q1'])} / {num(a['q3'])}",
                f"{num(b['q1'])} / {num(b['q3'])}"),
            row("IQR", num(a["iqr"]), num(b["iqr"])),
            "",
            f"Outlier fences      : lower {num(a['lower'])}, upper {num(a['upper'])}",
        ]
        records = self.outlier_records()
        if records:
            text = ", ".join(f"{v:,.2f} ({d}, {ag})" for v, d, ag in records)
        else:
            text = "none"
        lines.append(f"Outliers            : {text}")
        lines += ["", "Note for the risk team:", self._risk_note(a, b, records)]
        return "\n".join(lines)

    def _risk_note(self, a: dict, b: dict, records: list) -> str:
        if not records:
            return ("No deposit fell outside the IQR fences, so no day looks unusual. "
                    "The mean and median stay close, which suggests steady activity. "
                    "Rejected rows should still be chased up as missing data.")
        high = [r for r in records if r[0] > a["upper"]]
        low = [r for r in records if r[0] < a["lower"]]
        parts = []
        if high:
            parts.append(f"{len(high)} very high deposit(s) (largest {high[-1][0]:,.2f} KES) "
                         f"are far above the typical ceiling of {a['upper']:,.2f} KES and "
                         "could point to structuring, a batched or duplicated entry, "
                         "or one-off bulk customer activity.")
        if low:
            parts.append(f"{len(low)} very low deposit(s) (smallest {low[0][0]:,.2f} KES) "
                         "may mean a failed upload, a closed outlet that day, or "
                         "money held back from deposit.")
        parts.append(f"Removing these {len(records)} days moves the mean from {a['mean']:,.2f} "
                     f"to {b['mean']:,.2f} KES, so they distort the average and each "
                     "should be checked against agent records before any conclusion.")
        # keep it to exactly three sentences
        while len(parts) < 3:
            parts.insert(0, "Outliers are unusual values, not proof of wrongdoing.")
        return " ".join(parts)

      # NumPy comparison (the ONLY place numpy is imported)
def compare_with_numpy(analyzer: StatisticsAnalyzer) -> dict[str, bool]:
    """Import numpy HERE, recompute each statistic and return {name: matches}."""
    import numpy as np

    data = np.array(analyzer.values)
    a = analyzer.summary()

    def close(mine, theirs):          # tolerance, never ==
        return math.isclose(mine, float(theirs), rel_tol=1e-9, abs_tol=1e-9)

    return {
        "mean": close(a["mean"], np.mean(data)),
        "median": close(a["median"], np.median(data)),
        "variance": close(a["variance"], np.var(data, ddof=1)),    # sample -> ddof=1
        "std dev": close(a["std_dev"], np.std(data, ddof=1)),
        # method="linear" is the same quartile rule we implemented
        "Q1 / Q3": close(a["q1"], np.percentile(data, 25, method="linear"))
                   and close(a["q3"], np.percentile(data, 75, method="linear")),
    }

def numpy_section(analyzer: StatisticsAnalyzer) -> str:
    results = compare_with_numpy(analyzer)
    word = lambda ok: "MATCH" if ok else "MISMATCH"
    lines = ["=== NumPy verification ===",
             f"mean      {word(results['mean'])}",
             f"median    {word(results['median'])}",
             f"variance  {word(results['variance'])}  (ddof=1)",
             f"std dev   {word(results['std dev'])}",
             f"Q1 / Q3   {word(results['Q1 / Q3'])}  (method=linear)"]
    if not all(results.values()):
        lines.append("MISMATCH must be explained: check ddof (n-1 vs n) and quartile method.")
    return "\n".join(lines)

def tests_section() -> str:
    suite = unittest.defaultTestLoader.loadTestsFromName("test_stats_toolkit")
    stream = io.StringIO()
    unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    output = [ln for ln in stream.getvalue().splitlines() if ln.strip()]
    ran = next((ln for ln in output if ln.startswith("Ran ")), "Ran 0 tests")
    status = output[-1] if output else "FAILED"
    return "=== Unit tests ===\n" + ran + "\n" + status

if __name__ == "__main__":
    analyzer = StatisticsAnalyzer.from_csv(DATA_PATH, "daily_deposits_kes")
    full_report = "\n\n".join([analyzer.report(), numpy_section(analyzer), tests_section()])
    print(full_report)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(full_report + "\n", encoding="utf-8")
