# Temporal Functions

[Back to documentation index](index.md)

`lazybear` provides temporal functions for date-based calculations. These functions return lazy expressions (`Expr`)
that
compile to SQLAlchemy expressions and are evaluated by the database server.

## Contents

- [`calc_age(birth_date, reference_date)`](#calc_agebirth_date-reference_date)
    - [Parameters](#parameters)
    - [Examples](#examples)
    - [Behavior and edge cases](#behavior-and-edge-cases)
    - [Database support](#database-support)

---

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

A Python `date` is passed to SQLAlchemy as a bound parameter. You can also explicitly use `lb.lit(date(2026, 10, 21))`.

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

[Back to documentation index](index.md)
