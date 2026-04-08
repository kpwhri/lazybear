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

- Limits the number of rows.

#### `join(other, on=None, *, left_on=None, right_on=None, how='inner', suffixes=('_x', '_y'))`

- Joins with another `LazyBearFrame`.

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

#### `to_arrow(limit=None)`

- Executes the query and returns a `pyarrow.Table`.

#### `collect_batches(chunk_size=10_000)`

- Streams the query in batches, yielding `polars.DataFrame` chunks.

#### `iter_rows(named=False, chunk_size=10_000)`

- Yields rows as tuples (default) or dictionaries (if `named=True`).

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
