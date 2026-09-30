# API Reference

## `LazyBearFrame`

The main object representing a lazy SQL query. It is immutable; every transformation returns a new `LazyBearFrame`.

### Properties

#### `columns`

- Returns: `list[str]`
- A list of column names in the current frame.

#### `engine`

- Returns: `sqlalchemy.Engine`
- The SQLAlchemy engine this frame is bound to.

### Transformations

#### `select(*items)`

- Projects columns or expressions.
- Parameters:
    - `items`: Column names, tuples `(alias, expr)`, or `AliasedExpr`.

#### `filter(predicate)`

- Filters rows based on a boolean expression.
- Parameters:
    - `predicate`: An `Expr` or boolean condition.

#### `with_columns(*exprs, **named)`

- Adds or replaces columns.
- Parameters:
    - `exprs`: Positional expressions.
    - `named`: Keyword arguments for aliased expressions.

#### `sort(by, *more_by, descending=False)`

- Sorts the frame. Note that string sorting case sensitivity depends on the underlying database collation.
- Parameters:
    - `by`: Column name or expression to sort by.
    - `descending`: Boolean or sequence of booleans for sort direction.

#### `order_by(*keys)`

- SQLAlchemy-style ordering.
- Parameters:
    - `keys`: Column names (prefix with `-` for descending) or expressions.

#### `limit(n)`

- Limits the number of rocords returned to `n`.


#### `join(other, on=None, *, left_on=None, right_on=None, how='inner', suffix=None, prefix=None, apply_to_all=True, duplicate_columns='rename')`

Join this frame with another `LazyBearFrame` using one or more equality keys.

Both frames must use the same database server.

**Specifying join keys**

Use `on` when the key names are the same:

```python
users.join(accounts, on='user_id')
users.join(accounts, on=['user_id', 'region_id'])
```

Use `left_on` and `right_on` when the key names differ:

```python
users.join(
    accounts,
    left_on='id',
    right_on='user_id',
)
```

A mapping can also associate differently named keys:

```python
users.join(
  accounts,
  on={'id': 'user_id'},
)
```

Do not combine `on` with `left_on` or `right_on`.

**Join strategies**

`how` supports:

- `'inner'`: keep matching rows only.
- `'left'`: keep every left row.
- `'right'`: keep every right row.
- `'full'`: keep rows from both sides.

**Right-column naming**

When right-side column names overlap with left-side names:

- `prefix` adds a prefix to right-side columns.
- `suffix` adds a suffix to right-side columns.
- `prefix` takes precedence if both are supplied.
- An explicit prefix or suffix applies to every right-side column by default.
- Set `apply_to_all=False` to rename only overlapping columns.
- Without an explicit prefix or suffix, overlapping columns receive a generated suffix such as `_right` or `_right2`.

For example:

```python
users.join(orders, on={'id': 'user_id'}, prefix='order_')
```

produces right-side names such as `order_id`, `order_user_id`, and `order_amount`.

To rename only overlapping columns:

```python
users.join(
    orders,
    on={'id': 'user_id'},
    suffix='_order',
    apply_to_all=False,
)
```

Set `duplicate_columns='drop'` to omit right-side columns that would otherwise be renamed:

```python
users.join(
    orders,
    on={'id': 'user_id'},
    duplicate_columns='drop',
)
```

When the same key name is used on both sides, such as `on='id'`, the duplicate right key is omitted automatically.

See the complete [join documentation](join.md).

---

#### `join_where(other, *predicates, how='inner', suffix=None, prefix=None, apply_to_all=True, duplicate_columns='rename')`

Join this frame with another `LazyBearFrame` using arbitrary equality or inequality predicates.

This is useful for non-equi joins, including effective-date and interval matching:

```python
orders.join_where(
    prices,
    col('product_id') == col('product_id_right'),
    col('order_date') >= col('valid_from'),
    col('order_date') <= col('valid_to'),
)
```

This is equivalent to:

```sql
ON orders.product_id = prices.product_id
AND orders.order_date >= prices.valid_from
AND orders.order_date <= prices.valid_to
```

Multiple predicates are combined with `AND`. To express alternatives within one predicate, combine expressions with `|`.

**Join strategies**

`how` supports:

- `'inner'`: keep matching rows only.
- `'left'`: keep every left row.
- `'right'`: keep every right row.

A full join is not supported by `join_where`.

**Referencing right-side columns**

Predicates use the right-column names produced by the prefix and suffix rules.

When a column exists on both sides, the right-side reference receives a generated suffix by default:

```python
col('id') == col('id_right')
```

**Do not write:**

```python
# WRONG!
col('id') == col('id')
```

Both expressions resolve to the left-side `id`, producing `left.id = left.id` rather than a comparison between frames.

If an explicit prefix or suffix is supplied, use the resulting names in the predicates. Because explicit naming applies to all right-side columns by default, this example suffixes every right-side reference:

```python
orders.join_where(
    prices,
    col('product_id') == col('product_id_price'),
    col('order_date') >= col('valid_from_price'),
    col('order_date') <= col('valid_to_price'),
    suffix='_price',  # by default, suffix applies to all columns
)
```

To rename only overlapping right-side columns:

```python
orders.join_where(
    prices,
    col('product_id') == col('product_id_price'),
    col('order_date') >= col('valid_from'),
    col('order_date') <= col('valid_to'),
    suffix='_price',
    apply_to_all=False,
)
```

`prefix` takes precedence over `suffix` when both are supplied.

**Dropping duplicate columns**

Set `duplicate_columns='drop'` to omit selected right-side columns from the result:

```python
orders.join_where(
    prices,
    col('product_id') == col('product_id_right'),
    col('order_date') >= col('valid_from'),
    col('order_date') <= col('valid_to'),
    duplicate_columns='drop',
)
```

Dropped right-side columns remain available under their renamed names while the join predicates are evaluated.

The result row order is not guaranteed. Call `sort` or `order_by` when deterministic ordering is required.

See [equality and range-join examples](join.md#join-with-equality-and-range-predicates).


#### `group_by(*keys)`

- Groups by one or more columns. Returns a `GroupedLazyBearFrame`.

## `GroupedLazyBearFrame`

A frame representing grouped data, returned by `LazyBearFrame.group_by`.

### `agg(**aggregations)`

- Performs aggregations on the grouped data.
- Parameters:
    - `aggregations`: Keyword arguments where the key is the output column name and the value is a tuple `(column, function)`.
- Supported functions: `'count'`, `'sum'`, `'avg'`, `'mean'`, `'min'`, `'max'`.

```python
lf.group_by('department').agg(
    total_salary=('salary', 'sum'),
    avg_age=(col('age'), 'mean'),
    employee_count=('id', 'count')
)
```

### Execution & Materialization

#### `collect(limit=None, infer_schema_length=200)`

- Executes the query and returns a `polars.DataFrame`.
- Some database dialects may apply result cleaning after materialization. For example, Teradata/`teradatasql` string columns have trailing whitespace stripped to account for character datatype padding.

#### `to_arrow(limit=None)`

- Executes the query and returns a `pyarrow.Table`.

#### `collect_batches(chunk_size=10_000)`

- Streams the query in batches, yielding `polars.DataFrame` chunks.
- Some database dialects may apply result cleaning after materialization. For example, Teradata/`teradatasql` string columns have trailing whitespace stripped to account for character datatype padding.

#### `iter_rows(named=False, chunk_size=10_000)`

- Yields rows as tuples (default) or dictionaries (if `named=True`).
- Some database dialects may apply result cleaning after materialization. For example, Teradata/`teradatasql` string columns have trailing whitespace stripped to account for character datatype padding.

#### `explain()`

- Returns the SQL query as a string.

### I/O Helpers

#### `write_parquet(file, chunk_size=None, start_index=0, **kwargs)`

- Writes results to Parquet. If `chunk_size` is set, writes multiple files.

#### `write_csv(file, chunk_size=None, **kwargs)`

- Writes results to a CSV file.

### Advanced

#### `to_select()`

- Returns the underlying SQLAlchemy `Select` object.

## Scanning Functions: Create Temporary Tables on the Server

This has limited testing.

### `scan_table(table_name, engine, schema=None, lowercase=True)`

- Creates a `LazyBearFrame` from a database table.
- **Lowercaseing**: If `True` (default), column names are exposed as lowercase. This is useful for databases that return uppercase column names by default (e.g., Snowflake, Oracle, DB2) to keep code consistent with Polars. Set to `False` to preserve the exact casing from the database.

### `scan_sql_query(query, engine, columns=None)`

- Creates a `LazyBearFrame` from a raw SQL SELECT query.

### `scan_df(df, engine, table_name=None)`

- Creates a `TempLazyBearFrame` from a local `polars.DataFrame`.
- **Temp Tables**: The DataFrame is uploaded to a temporary table on the database only when `collect()` or similar materialization methods are called. The table is automatically dropped after the result is fetched.
- **Dialect Support**: Beta support for SQLite, PostgreSQL, SQL Server, Oracle, and Teradata.
