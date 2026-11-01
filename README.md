# Day 3 - Statistical Analysis Toolkit

A statistics toolkit written in plain Python (no `statistics` module; NumPy is used only to
verify results) that analyses daily agent deposits in KES from `data/agent_deposits.csv`.

## Approach
1. `load_column` reads the CSV and separates trustworthy numbers from rejected values.
2. Hand-written functions compute mean, median, mode, variance, std dev, quartiles and IQR outliers.
3. `StatisticsAnalyzer` wraps a column, sorts it once, caches results, and builds the report.
4. `compare_with_numpy` (the only place NumPy is imported) checks every statistic with a tolerance.
5. `test_stats_toolkit.py` holds 14 small hand-checkable unit tests.

## How to run
```bash
python -m venv venv
source venv/Scripts/activate      # Git Bash on Windows 
pip install -r requirements.txt
python stats_toolkit.py           # prints and saves output/stats_report.txt
python -m unittest -v             # runs the unit tests
```

## My choices
- ### Sample vs population variance:
 sample (n - 1). The 35 valid days are a sample of the
  network's ongoing activity, not every day that will ever exist, so n - 1 avoids
  underestimating spread. `sample=False` gives the population (n) version.
- ### Quartile method:
 linear interpolation, position = (n - 1) * p. This equals NumPy's default
  `method="linear"`, so the NumPy comparison uses the same rule. Q2 is the median.
- ### Mode:
 returns every tied value as a list. If every value appears once, it returns `[]`,
  shown as "none" in the report.
- ### Invalid and missing values:
 blanks, `-`, `n/a`, `12,40o` (letter o), negatives, NaN and inf
  are rejected, never guessed. Each rejection is listed in the report with its row number
  (header = row 1), raw value and reason.
- ### Outliers:
 IQR rule, fences at Q1 - 1.5*IQR and Q3 + 1.5*IQR. Outliers are reported with
  date and agent, not deleted; `without_outliers()` returns a second analyzer for comparison.
- ### Efficiency:
 data is sorted once in the constructor; `summary()` is cached.
- ### NumPy check:
 uses `math.isclose` (not `==`) with ddof=1 and `method="linear"`. Any
  mismatch would be due to ddof (n vs n - 1) or a different quartile method.

## Assumptions
- Amounts in `daily_deposits_kes` are in Kenyan shillings, one row per agent per day.
- The 35 valid days are a **sample**, not the full population, so sample variance (n - 1) is the default.
- A deposit cannot be negative, so negatives are rejected. Zero is accepted as a valid value.
- Text that is not a clean number (blank, `-`, `n/a`, `12,40o`) cannot be trusted, so it is rejected
  rather than corrected. In particular, `12,40o` is not assumed to mean 12,400.
- Rows are treated as independent observations; repeated agents and dates are not adjusted for.
- An extreme value is flagged as an outlier by the 1.5 x IQR rule, but it is **not** assumed to be an
  error. It stays in the "all data" column and is only excluded in "outliers removed".
- Row numbers in the report count the CSV header as row 1, so the first data row is row 2.
- Only the `daily_deposits_kes` column is analysed; `transactions` is out of scope.

## What the outliers might mean
Four days fall outside the fences: three very high deposits (96,500; 88,400; 29,060.99) and one
very low (1,200). Highs could be structuring, batched or duplicate entries, or bulk customer
activity; the low could be a failed upload or a closed outlet. Three of the four belong to AG-104,
so that agent deserves a review. Removing them drops the mean from 22,400.00 to 18,349.65 KES,
while the median barely moves, so the median is the safer "typical day" figure.

## Results (see `output/stats_report.txt`)
40 rows read, 35 valid, 5 rejected. All statistics MATCH NumPy.
