# DataTypes (Type Aliases)

[Back to index](index.md)

Many of `lazybear`'s functions use three type aliases to describe the inputs they accept. These are **type hints**, not new runtime types. Use `lb.col()` for column references and `lb.lit()` for explicit literal values whenever possible.

```python
from datetime import date, datetime
from typing import Any, TypeAlias

import sqlalchemy as sa

from lazybear.expressions import Expr

ColumnInput: TypeAlias = Expr | sa.ColumnElement[Any] | str
DateInput: TypeAlias = ColumnInput | date | datetime
OffsetInput: TypeAlias = int | Expr | sa.ColumnElement[Any]
```


## `ColumnInput`

A column reference or expression. Used for arguments that are expected to refer to a column, such as `calc_age()`'s `birth_date`.

```python
lb.col('birth_date')  # lazybear Expr (recommended)
patients.c.birth_date  # sqlalchemy ColumnElement
'birth_date'  # Column name as str (accepted, but warns)
```

A string column name is converted to `lb.col('birth_date')` by `str_to_col()`, with a warning encouraging explicit `lb.col()` usage.

## `DateInput`

A date-valued expression, column, or literal. Includes every `ColumnInput` form, plus Python `date` and `datetime` values.

```python
lb.col('index_date')  # lazybear Expr referencing a column
lb.lit(date(2026, 10, 21))  # lazybear Expr containing a literal date (recommended)
patients.c.index_date  # sqlalchemy ColumnElement
'index_date'  # String column name (warns)
'2026-10-21'  # Date string (warns, but converted to a literal date)
date(2026, 10, 21)  # Python date (accepted without warning)
datetime(2026, 10, 21, 14, 30)  # Python datetime (warns: time is discarded)
```

When `date_to_col_or_lit()` is used, strings that look like dates become date literals; other strings become column references. Both string forms generate guidance warnings. Python `date` values are accepted directly, while Python `datetime` values are converted to dates with a warning.

For example:

```python
lb.calc_age(lb.col('birth_date'), date(2026, 10, 21))
lb.date_diff(lb.col('index_date'), lb.lit(date(2026, 12, 31)))
```

## `OffsetInput`

An integer or integer-valued expression, used for the `days`, `months`, and `years` arguments of `date_add()`.

```python
90  # Python int (fixed offset)
lb.col('followup_days')  # lazybear Expr (per-row offset)
patients.c.followup_days  # sqlalchemy ColumnElement (per-row offset)
```

For example:

```python
lb.date_add(lb.col('index_date'), days=90)
lb.date_add(lb.col('index_date'), days=lb.col('followup_days'))
```

**Note:** `in_interval(ndays=...)` and `in_window(start_days=..., end_days=...)` accept integer offsets only, not arbitrary `OffsetInput` expressions. `bool` is not a valid offset despite being a subclass of `int` in Python.
