# Expressions & Namespaces

LazyBear expressions (`Expr`) are used to define transformations, filters, and aggregations. They mimic the Polars expression API but are compiled into SQLAlchemy expressions for execution on the database server.

## Basic Expressions

### `col(name)`
References a column by name.

```python
import lazybear as lb

lf.select(lb.col('age'))
# or simply
lf.select('age')
```

### `lit(value)`
Creates a literal value expression.

```python
import lazybear as lb

lf.with_columns(
    status=lb.lit('active')
)
```

### `coalesce(*columns)`

`coalesce` returns the first non-null value. Literal values can be directly included as 'defaults'.

```python
import lazybear as lb

lf.with_columns(
        lb.coalesce(lb.col('nickname'), lb.col('first_name'), 'unknown').alias('display_name'),
        lb.coalesce(lb.col('score'), 0).alias('score_filled'),
)
```

This is equivalent to SQL like:

```sql
COALESCE(nickname, first_name, 'unknown')
COALESCE(score, 0)
```

Use it to replace nulls or choose the first available value across columns.

```python
lf.filter(
    lb.coalesce(lb.col('score'), lb.col('bonus'), 0) > 75
)
```

`coalesce()` requires at least one argument.

### Comparison Operators
Standard Python comparison operators are supported: `==`, `!=`, `>`, `>=`, `<`, `<=`.

```python
lf.filter(lb.col('age') >= 18)
```

### Arithmetic Operators
Standard arithmetic operators: `+`, `-`, `*`, `/`, `%`.

```python
lf.with_columns(
    age_next_year=lb.col('age') + 1
)
```

### Boolean Operators
Use `&` for AND, `|` for OR.

```python
lf.filter(
    (lb.col('age') > 30) & (lb.col('name') == 'Ahti')
)
```

### `is_null()` / `is_not_null()`
Check for null values.

```python
lf.filter(lb.col('amount').is_not_null())
```

### `is_in(other)`
Checks if a value is present in a list, Polars Series, or another `LazyBearFrame` (subquery).

```python
# With a list
lf.filter(col('id').is_in([1, 2, 3]))

# With a subquery
young_users = lf.filter(lb.col('age') < 30).select('id')
lf.filter(lb.col('id').is_in(young_users))
```

**Warning**: `is_in` will create a where clause, and your server may have limits on how many elements can be included. To get around this, consider using `scan_df` to create a temp table or apply in batches.

### `alias(name)`
Renames an expression.

```python
lf.select(
    (lb.col('age') * 2).alias('double_age')
)
```

## String Namespace (`.str`)

The `.str` namespace provides string-specific operations.

### `contains(pattern, literal=True)`
Checks if the string contains a substring. Case sensitivity depends on the underlying database collation.

- If `literal=True` (default): treats `pattern` as a literal substring.
- If `literal=False`: treats `pattern` as a SQL `LIKE` pattern.

```python
# Literal match
lf.filter(lb.col('name').str.contains('hti'))

# SQL LIKE match
lf.filter(lb.col('name').str.contains('%hti%', literal=False))
```

### `contains_any(patterns, literal=True)`
Checks if the string contains any of the provided patterns.

```python
lf.filter(lb.col('name').str.contains_any(['Ah', 'Ka']))
```

### `starts_with(prefix, ascii_case_insensitive=False)`
Checks if the string starts with the given prefix.

- `ascii_case_insensitive`: If `True`, performs a case-insensitive match by applying `UPPER()` to both sides. Note this only handles ASCII characters correctly in most databases.

```python
lf.filter(lb.col('name').str.starts_with('A'))

# Case-insensitive
lf.filter(lb.col('name').str.starts_with('a', ascii_case_insensitive=True))
```

### `starts_with_any(prefixes)`
Checks if the string starts with any of the provided prefixes.

```python
lf.filter(lb.col('name').str.starts_with_any(['A', 'K']))
```

## Callable Proxy (`.startswith`)

For convenience, `col('x').startswith('a')` is available as a shortcut for `col('x').str.starts_with('a')`.

```python
lf.filter(lb.col('name').startswith('A'))
```

## Examples from Tests

### Complex Filtering
Combining multiple conditions:

```python
# from tests/test_lazybear.py
users.filter((lb.col('age') > 30) & (lb.col('age') < 60))
```

### String Operations
Using `.str` for filtering:

```python
# from tests/test_lazybear.py
users.filter(lb.col('name').str.contains('a', literal=True))
users.filter(lb.col('name').str.starts_with_any(['a', 'k']))
```

### Subqueries
Using `is_in` with another `LazyBearFrame`:

```python
import lazybear as lb

young_ids = lb.scan_table('users', engine).filter(
    lb.col('age') < 30,
).select('id')
users.filter(
    lb.col('id').is_in(young_ids)
)
```
