# Temporal Functions

[Back to documentation index](index.md)

`lazybear` provides temporal functions for date-based calculations. These functions return lazy expressions (`Expr`)
that
compile to sqlalchemy expressions and are evaluated by the database server.
## Contents

- [`calc_age(birth_date, reference_date)`](#calc_age)
    - [Parameters](#parameters)
    - [Examples](#examples)
    - [Behavior and edge cases](#behavior-and-edge-cases)
    - [Database support](#database-support)
- [`date_add(value, *, days=0, months=0, years=0)`](#date_add)
- [`date_diff(start_date, end_date)`](#date_diff)
- [`date_trunc(value, unit)`](#date_trunc)
- [`date_part(value, part)`](#date_part)
- [`between(value, start, end)`](#between)
- [`in_interval(event_date, reference_date, ndays=90)`](#in_interval)
- [`in_window(event_date, reference_date, *, start_days, end_days)`](#in_window)
- [`intervals_overlap(a_start, a_end, b_start, b_end)`](#intervals_overlap)

---
<a name="calc_age"></a>
## `calc_age(birth_date, reference_date)`

[Back to contents](#contents)

Calculate age in **completed years** at a reference date. The calculation subtracts one year if the birthday has not yet
occurred in the reference year. It does not simply subtract the birth year from the reference year.

Returns a LazyBear `Expr`, which can be used with `select`, `with_columns`, and `filter`.

### Parameters

- `birth_date`: Date column or expression, such as `col('birth_date')`.
- `reference_date`: Date column or expression, such as `col('index_date')`, or a Python `datetime.date` value.

Both arguments are required. Use `col()` to reference a column by name; a plain string is treated as a literal value.

### Examples

Use another column as the reference date, for example to calculate age at a study index date:

```python
import lazybear as lb

cohort_df = lb.scan_table('patients', eng)

result = (
    cohort_df
    .with_columns(
        age=lb.calc_age(lb.col('birth_date'), lb.col('index_date')),
    )
    .select('patient_id', 'index_date', 'age')
    .collect()
)
```

Use a fixed reference date:

```python
from datetime import date

import lazybear as lb
result = (
    cohort_df
    .select(
        'patient_id',
        lb.calc_age(
            lb.col('birth_date'),
            date(2026, 10, 21),
        ).alias('age'),
    )
    .collect()
)
```

A Python `date` is passed to sqlalchemy as a bound parameter. You can also explicitly use `lb.lit(date(2026, 10, 21))`.

Filter a cohort to patients aged 18 or older at their individual index dates:

```python
import lazybear as lb
adults = (
    cohort_df
    .filter(
        lb.calc_age(lb.col('birth_date'), lb.col('index_date')) >= 18
    )
    .select('patient_id', 'birth_date', 'index_date')
    .collect()
)
```
### Behavior and edge cases

With a reference date of October 21, 2026:

| Birth date   |    Age |
|--------------|-------:|
| `2000-10-20` |     26 |
| `2000-10-21` |     26 |
| `2000-10-22` |     25 |
| `2026-10-21` |      0 |
| `2026-11-01` |     -1 |
| `NULL`       | `NULL` |

- **Missing dates:** The result is SQL `NULL` if either date is `NULL`.
- **Leap-year birthdays:** February 29 birthdays advance on March 1 in non-leap years. For example, a person born on
  February 29, 2000 is 24 on February 28, 2025, and 25 on March 1, 2025.
- **Future birth dates:** A birth date after the reference date can produce a negative value. The function does not
  validate dates or replace negative ages with `NULL`.
- **Datetime values:** The calculation uses the year, month, and day. If timestamps have time-zone information, ensure
  that dates are interpreted in the intended time zone before calculating age.
### Database support

`calc_age()` uses sqlalchemy's dialect-aware `extract()` and `case()` operations rather than database-specific
age functions. The expression is designed to work across sqlite, postgresql, sql server, oracle, and mysql/mariadb.
SQL compilation is tested for these dialects; sqlite execution tests verify the calculation and its edge cases.
Actual execution should also be tested against each database backend used in production, since successful SQL
compilation does not guarantee server compatibility.

As with other LazyBear expressions, computation is deferred until an execution method such as `collect()` is called.
See [Expressions & Namespaces](expressions.md) for additional expression operators and functions.

---
<a name="date_add"></a>
## `date_add(value, *, days=0, months=0, years=0)`

[Back to contents](#contents)

Add or subtract calendar days, months, or years from a date. Offsets can be integers, LazyBear expressions, or
sqlalchemy columns. All offsets default to zero, and are applied in **years, months, days** order. Negative values
subtract dates. For reliable study windows, prefer day offsets: **month-end rollover behavior differs between databases**.

```python
import lazybear as lb

follow_up = lb.date_add(lb.col('index_date'), days=90)
lookback = lb.date_add(lb.col('index_date'), days=-365)
next_year = lb.date_add(lb.col('index_date'), years=1)
```

An omitted or zero offset leaves the date unchanged. Missing dates produce SQL `NULL`. The operation is intended for
calendar dates; timestamp/time-of-day preservation is not guaranteed across database backends.

---
<a name="date_diff"></a>
## `date_diff(start_date, end_date)`

[Back to contents](#contents)

Calculate the signed difference **in calendar days**: `end_date - start_date`. Time-of-day is ignored and a missing
start or end date produces SQL `NULL`.

```python
import lazybear as lb

result = (
    cohort_df
    .with_columns(
        days_to_diagnosis=lb.date_diff(lb.col('index_date'), lb.col('diagnosis_date')),
    )
    .collect()
)
```

The value is positive for dates after the start date, zero for the same calendar day, and negative for earlier dates.

---
<a name="date_trunc"></a>
## `date_trunc(value, unit)`

[Back to contents](#contents)

Return the first date in the calendar period containing `value`. Supported `unit` strings are `'day'`, `'month'`,
`'quarter'`, and `'year'`. The result is a **date**, not a timestamp.

```python
import lazybear as lb

monthly_counts = (
    cohort_df
    .with_columns(month=lb.date_trunc(lb.col('diagnosis_date'), 'month'))
    .group_by('month')
    .agg(n=('patient_id', 'count'))
    .collect()
)
```

For example, truncating `2026-08-19` to a month gives `2026-08-01`; truncating to a quarter gives `2026-07-01`.
A missing date returns SQL `NULL`, and an unsupported unit raises `ValueError` when building the expression.

---
<a name="date_part"></a>
## `date_part(value, part)`

[Back to contents](#contents)

Extract the `'year'`, `'month'`, `'day'`, `'quarter'`, or `'weekday'` as an integer. Quarters run from 1 to 4.
Weekday uses **ISO numbering (Monday = 1, Sunday = 7)**, independent of the database server's weekday settings.

```python
import lazybear as lb

result = (
    cohort_df
    .select(
        lb.date_part(lb.col('diagnosis_date'), 'year').alias('diagnosis_year'),
        lb.date_part(lb.col('diagnosis_date'), 'quarter').alias('diagnosis_quarter'),
        lb.date_part(lb.col('diagnosis_date'), 'weekday').alias('weekday'),
    )
    .collect()
)
```

Missing dates produce SQL `NULL` for all parts, including quarter. Unsupported parts raise `ValueError`.

---
<a name="between"></a>
## `between(value, start, end)`

[Back to contents](#contents)

An **inclusive** range predicate: `start <= value AND value <= end`. Equivalent to `col('date').between(start, end)`:

```python
import lazybear as lb

cohort_df.filter(
  lb.between(lb.col('diagnosis_date'), lb.col('start_date'), lb.col('end_date')),
)
cohort_df.filter(
  lb.col('diagnosis_date').between(lb.col('start_date'), lb.col('end_date')),
)
```

The function and the expression method use SQL `BETWEEN`; missing values give SQL `NULL`.

---
<a name="in_interval"></a>
## `in_interval(event_date, reference_date, ndays=90)`

[Back to contents](#contents)

Check whether an event occurs within a relative **inclusive** calendar-day window. A positive `ndays` checks from
reference day through `reference + ndays`; a negative value checks from `reference + ndays` through reference day.
Zero includes only the reference day.

```python
import lazybear as lb

# Days 0 through 90 after treatment (91 included calendar dates).
within_followup = lb.in_interval(lb.col('diagnosis_date'), lb.col('treatment_date'), ndays=90)

# Days -365 through 0 before index date.
prior_diagnosis = lb.in_interval(lb.col('diagnosis_date'), lb.col('index_date'), ndays=-365)
```

`ndays` must be a Python integer; booleans are rejected. Both boundaries are inclusive, so use `in_window()` for
windows that exclude the reference day.

---
<a name="in_window"></a>
## `in_window(event_date, reference_date, *, start_days, end_days)`

[Back to contents](#contents)

Check whether an event occurs within an arbitrary relative inclusive date window. Both offsets are required,
may be negative, and `start_days` must not exceed `end_days`.

```python
import lazybear as lb

# Days 1–90 after index: exclude day zero.
follow_up = lb.in_window(lb.col('outcome_date'), lb.col('index_date'), start_days=1, end_days=90)

# 365 days before through 30 days after index.
near_index = lb.in_window(lb.col('diagnosis_date'), lb.col('index_date'), start_days=-365, end_days=30)
```

An invalid order raises `ValueError`. Both endpoints are inclusive; missing dates evaluate to SQL `NULL`.

---
<a name="intervals_overlap"></a>
## `intervals_overlap(a_start, a_end, b_start, b_end)`

[Back to contents](#contents)

Check whether two nonempty, **half-open** date intervals overlap: `[a_start, a_end)` and `[b_start, b_end)`.
An endpoint touching the other interval's start does **not** count as overlap. Empty intervals do not overlap.

```python
import lazybear as lb

cohort_df.filter(
    lb.intervals_overlap(
        lb.col('admission_date'), lb.col('discharge_date'),
        lb.col('exposure_start'), lb.col('exposure_end'),
    )
)
```

This convention suits intervals whose end denotes the first day **not** included. If dates such as a medication's
last covered day are stored inclusively, add one day to the inclusive end before calling `intervals_overlap()`.
NULL endpoints follow SQL three-valued logic; filter predicates only retain rows evaluating to SQL TRUE.

### Database support

These functions are designed for SQLite, PostgreSQL, SQL Server, Oracle, and MySQL/MariaDB. `date_add()`,
`date_diff()`, `date_trunc()`, and the ISO weekday calculation use database-specific SQL compilation; ordinary
range predicates use sqlalchemy operators. The test suite checks SQL compilation for all five dialects and execution
against SQLite. Real-server integration tests are recommended for every production backend.

[Back to documentation index](index.md)
