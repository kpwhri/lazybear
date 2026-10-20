# `LazyBearFrame`

[Back to API reference](../api.md)

`LazyBearFrame` is the main object representing a lazy sql query.

It is immutable: every transformation returns a new `LazyBearFrame`.

lazybear is intended to feel familiar to polars users, but it is not intended to be feature complete. The goal is to
push filtering, projection, joins, grouping, and limiting into the database before collecting data locally for analysis.

## Contents

- [Properties](#properties)
    - [`columns`](#columns)
    - [`engine`](#engine)
- [`select`](#selectitems)
- [`filter`](#filterpredicates-constraints)
- [`with_columns`](#with_columnsexprs-named)
- [`drop`](#dropcolumns-stricttrue)
- [`sort`](#sortby-more_by-descendingfalse)
- [`order_by`](#order_bykeys)
- [`limit`](#limitn)

---

## Properties

[Back to contents](#contents)

### `columns`

Return a list of column names in the current frame.

```python
users.columns
```

Example:

```python
from lazybear import scan_table

users = scan_table('users', engine)

print(users.columns)
```

```python
['id', 'name', 'age']
```

---

### `engine`

Return the sqlalchemy engine this frame is bound to.

```python
users.engine
```

Example:

```python
from lazybear import scan_table

users = scan_table('users', engine)

assert users.engine is engine
```

---

## `select(*items)`

[Back to contents](#contents)

Project columns or expressions.

### Parameters

- `*items`: Columns or expressions to select.
    - `str`: select a column by name.
    - `list[str]`: select multiple columns.
    - `(alias, expr)`: select an expression with an output name.
    - `AliasedExpr`: select an expression created with `.alias(...)`.

### Examples

Select columns by name:

```python
users.select('id', 'name')
```

Select from a list:

```python
users.select(['id', 'name'])
```

Select an expression with an alias:

```python
from lazybear import col

users.select(
    'id',
    ('age_next_year', col('age') + 1),
)
```

Use `.alias(...)`:

```python
from lazybear import col

users.select(
    'id',
    (col('age') + 1).alias('age_next_year'),
)
```

Collect selected columns:

```python
df = (
    users
    .select('id', 'name')
    .collect()
)
```

---

## `filter(*predicates, **constraints)`

[Back to contents](#contents)

Filter rows using one or more boolean expressions and/or equality constraints.

This is similar to polars `filter`, with support for keyword constraints.

### Parameters

- `*predicates`: One or more lazybear boolean expressions. Multiple predicates are combined with `AND`.
- `**constraints`: Column equality filters written as keyword arguments.

### Notes

- Use `lazybear.col`, not `polars.col` or `pl.col`.
- Keyword constraints are shorthand for equality filters.
- Predicates and constraints can be mixed.

### Examples

Filter with one predicate:

```python
from lazybear import col

users.filter(col('age') >= 30)
```

Filter with multiple predicates:

```python
from lazybear import col

users.filter(
    col('age') >= 30,
    col('name') != 'Ukko',
)
```

Use keyword constraints:

```python
users.filter(age=41, name='Tellervo')
```

This is equivalent to:

```python
from lazybear import col

users.filter(
    col('age') == 41,
    col('name') == 'Tellervo',
)
```

Mix predicates and constraints:

```python
from lazybear import col

users.filter(
    col('id') > 1,
    age=41,
)
```

Use null checks:

```python
from lazybear import col

heroes.filter(col('nickname').is_null())
```

Collect filtered rows:

```python
from lazybear import col

df = (
    users
    .filter(col('age') >= 30)
    .sort('id')
    .collect()
)
```

---

## `with_columns(*exprs, **named)`

[Back to contents](#contents)

Add or replace columns while keeping existing columns.

### Parameters

- `*exprs`: Positional expressions. These must be aliased expressions or `(name, expr)` tuples.
- `**named`: Keyword expressions where the keyword becomes the output column name.

### Examples

Add a column with a keyword expression:

```python
from lazybear import col

users.with_columns(
    age_next_year=col('age') + 1,
)
```

Add a column with a tuple:

```python
from lazybear import col

users.with_columns(
    ('age_next_year', col('age') + 1),
)
```

Add multiple columns:

```python
from lazybear import col

users.with_columns(
    age_next_year=col('age') + 1,
    is_older=col('age') >= 40,
)
```

Replace an existing column:

```python
from lazybear import col

users.with_columns(
    age=col('age') + 1,
)
```

---

## `drop(*columns, strict=True)`

[Back to contents](#contents)

Remove one or more columns.

### Parameters

- `*columns`: Column names to remove. Accepts varargs or a single list/tuple of column names.
- `strict`: Validate that all column names exist in the current schema. If `True`, raise when any requested column is
  missing. If `False`, ignore missing columns.

### Examples

Drop one column:

```python
users.drop('age')
```

Drop multiple columns:

```python
users.drop('name', 'age')
```

Drop columns from a list:

```python
users.drop(['name', 'age'])
```

Ignore missing columns:

```python
users.drop('maybe_missing', strict=False)
```

---

## `sort(by, *more_by, descending=False)`

[Back to contents](#contents)

Sort the frame using a polars-style API.

### Parameters

- `by`: Column name, or a list/tuple of column names.
- `*more_by`: Additional column names.
- `descending`: Boolean or sequence of booleans for sort direction.

### Notes

- Sort keys must be column names.
- String sorting case sensitivity depends on the underlying database collation.

### Examples

Sort ascending:

```python
users.sort('age')
```

Sort descending:

```python
users.sort('age', descending=True)
```

Sort by multiple columns:

```python
users.sort('age', 'name')
```

Use different directions per column:

```python
users.sort(
    ['age', 'name'],
    descending=[True, False],
)
```

Collect sorted rows:

```python
df = (
    users
    .sort('age', descending=True)
    .collect()
)
```

---

## `order_by(*keys)`

[Back to contents](#contents)

Order rows using a compact sql-style column syntax.

### Parameters

- `*keys`: Column names.
    - Prefix with `-` for descending.
    - Prefix with `+` for explicit ascending.

### Examples

Order ascending:

```python
users.order_by('name')
```

Order descending:

```python
users.order_by('-age')
```

Order by multiple columns:

```python
users.order_by('-age', 'name')
```

This is similar to:

```python
users.sort(['age', 'name'], descending=[True, False])
```

---

## `limit(n)`

[Back to contents](#contents)

Limit the number of records returned to `n`.

### Parameters

- `n`: Non-negative integer row limit.

### Examples

Return the first 10 rows:

```python
users.limit(10)
```

Sort before limiting:

```python
users.sort('age', descending=True).limit(5)
```

Collect a limited result:

```python
from lazybear import col

df = (
    users
    .filter(col('age') >= 30)
    .sort('age', descending=True)
    .limit(10)
    .collect()
)
```

[Back to API reference](../api.md)