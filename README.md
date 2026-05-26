# LazyBear

LazyBear is a polars-flavored lazy SQL query builder on top of sqlalchemy. It lets you compose familiar
transformations like `select`, `filter`, `with_columns`, `join`, `group_by().agg(...)`, and `sort`/`order_by` against a
SQL database, and then materialize results to polars or arrow (or stream them in batches).

The purpose of this library is to provide lazy, polars-like access to a single sql database server, providing multi-site
(and multi-server) stable code while pushing most memory-intensive operations to the remote server.

- Familiar, chainable API similar to polars
- Backed by sqlalchemy for broad database support
- Zero data is loaded until you call `collect()`/`to_arrow()`/writers
- Convenient I/O helpers for CSV and Parquet (using polars)

Full API documentation:

* [Markdown](docs/index.md)
* [Github Pages](https://kpwhri.github.io/lazybear)

## Installation

* LazyBear targets Python 3.11+.
* Required dependencies:
    - `sqlalchemy`
    - `polars`
* Optional:
    - `pyarrow` for `to_arrow()`

Install from PyPI:

```
pip install lazybear-polars
```

Install from source using pip:

```
# after git clone
pip install -e .
# straight from repo
pip install git+https://github.com/kpwhri/lazybear.git@master
```

## Usage

### Quickstart

Below is an end-to-end walkthrough using an in-memory sqlite database. The same api should work for other databases
supported by sqlalchemy.

#### Create Playgruond

First, let's setup the backend play data:

```python
import sqlalchemy as sa
import polars as pl

# create a sqlalchemy engine
eng = sa.create_engine('sqlite:///:memory:')

# prepare some tables for the demo
meta = sa.MetaData()
users = sa.Table(
    'users', meta,
    sa.Column('id', sa.Integer, primary_key=True),
    sa.Column('name', sa.String),
    sa.Column('age', sa.Integer),
)
orders = sa.Table(
    'orders', meta,
    sa.Column('id', sa.Integer, primary_key=True),
    sa.Column('user_id', sa.Integer),
    sa.Column('amount', sa.Float),
)
meta.create_all(eng)
with eng.begin() as conn:
    conn.execute(users.insert(), [
        {'id': 1, 'name': 'Ahti', 'age': 30},
        {'id': 2, 'name': 'Kalma', 'age': 28},
        {'id': 3, 'name': 'Tellervo', 'age': 41},
        {'id': 4, 'name': 'Ukko', 'age': 41},
    ])
    conn.execute(orders.insert(), [
        {'id': 10, 'user_id': 1, 'amount': 12.5},
        {'id': 11, 'user_id': 1, 'amount': 7.5},
        {'id': 12, 'user_id': 2, 'amount': 99.0},
    ])
```

#### Implemented SQL Operations

Now, let's see what we can do:

```python

from lazybear import scan_table, scan_sql_query, col

# scan tables lazily
lf_users = scan_table('users', eng)  # columns exposed lowercase by default
lf_orders = scan_table('orders', eng)

# basic select / filter / collect → returns a polars DataFrame
out_df = (
    lf_users
    .filter(col('age') > 30)
    .select('id', 'name', 'age')
    .collect()
)
print(out_df)

# with_columns — add or replace columns
with_cols_df = (
    lf_users
    .with_columns(('age2', col('age') * 2), decade=col('age') / 10 * 10)
    .select('id', 'age2', 'decade')
    .collect()
)
print(with_cols_df)

# order and limit (keeps ordering stable even across subqueries)
ordered_df = lf_users.order_by('-age', 'name').limit(2).collect()
print(ordered_df)

# polars-style sort api
sorted_df = lf_users.sort(by='age', descending=True).collect()
print(sorted_df)

# joins
joined_df = (
    lf_users
    .join(lf_orders, left_on='id', right_on='user_id', how='left')
    .select('id', 'name', 'age', 'amount')
    .collect()
)
print(joined_df)

# rename all right-side columns with a prefix
prefixed_join_df = (
    lf_users
    .join(lf_orders, left_on='id', right_on='user_id', how='left', prefix='order_')
    .select('id', 'name', 'order_id', 'order_amount')
    .collect()
)
print(prefixed_join_df)

# joins (left columns keep names; right overlaps get suffixed with `_y` by default)
joined_df = (
    lf_users
    .join(lf_orders, on={'id': 'user_id'}, how='left')
    .select('id', 'name', 'age', ('amount_y', col('amount')))
    .collect()
)
print(joined_df)

# when chaining multiple joins that overlap on the same column names,
# provide a different suffixes=... value for later joins to avoid label collisions.
chained_join_df = (
    lf_users
    .join(lf_orders, on={'id': 'user_id'}, how='left')
    .join(lf_orders, on={'id': 'user_id'}, how='left', suffixes=('_x2', '_y2'))
    .select('id', 'name', 'amount', 'amount_y2')
    .order_by('id', 'amount', 'amount_y2')
    .collect()
)
print(chained_join_df)

# group_by + aggregations
agg_df = (
    lf_users
    .group_by('age')
    .agg(n=('id', 'count'), min_id=('id', 'min'))
    .collect()
)
print(agg_df)

# expressions: membership, null checks, and string helpers
expr_df = (
    lf_users
    .filter(
        (col('name').str.contains('k', literal=True)) |
        (col('name').startswith('A')) |
        (col('age').is_in([28, 41]))
    )
    .collect()
)
print(expr_df)
```

### Uploading Temp Tables (Beta)

You can use a local polars DataFrame in lazy sql operations by creating a temporary table. This is useful for joining
local data with database tables.

Be aware that the table will be inserted when `collect` is called. After the collection is complete, a best effort
attempt to delete the temp table will be completed.

```python
import polars as pl
from lazybear import scan_df, scan_table, col

# local polars dataframe
df_local = pl.DataFrame({
    'user_id': [1, 3],
    'status': ['active', 'inactive']
})

# create a temp table frame
lf_temp = scan_df(df_local, eng, table_name='user_status')

# join with a database table
lf_users = scan_table('users', eng)
result = (
    lf_users
    .join(lf_temp, on={'id': 'user_id'})
    .select('name', 'status')
    .collect()
)
```

**Limitations & Behavior:**

- Temp tables are created and data is inserted only when `collect()` (or another execution method) is called.
- Best-effort cleanup (DROP TABLE) is performed after the data is fetched.
- Currently offers _beta_ support for sqlite, PostgreSQL, SQL Server (MSSQL), Oracle, and Teradata. A warning is issued
  for other dialects.
- For certain dialects, will attempt bulk insert to speed up processing

#### Exporting Data

Finally, we'll

```python
# iterating rows
for row in lf_users.order_by('id').iter_rows():
    # row is a tuple by default
    print(row)
for row in lf_users.order_by('id').iter_rows(named=True):
    # named=True yields dicts
    print(row)

# streaming in batches
for batch_df in lf_users.order_by('id').collect_batches(chunk_size=2):
    print('Batch:', batch_df)

# scan a raw SQL query (must be SELECT)
q_df = (
    scan_sql_query('select name, age from users where age > 30', eng)
    .filter(col('age') < 60)
    .select('name')
    .collect()
)
print(q_df)

# materialize to arrow (requires optional pyarrow)
arrow_tbl = lf_users.to_arrow()

# explain shows the composed SQL (with literal binds where possible)
print(lf_users.filter(col('age') > 30).explain())
```

### I/O helpers

- CSV

```python
# single file
lf_users.order_by('id').write_csv('users.csv', include_header=True)

# append in chunks to one file
lf_users.order_by('id').write_csv('users_chunked.csv', chunk_size=2, include_header=True)
```

- Parquet

```python
# single file
lf_users.collect().write_parquet('users.parquet')

# chunked parts with a common prefix like users-00000.parquet, users-00001.parquet, ...
lf_users.order_by('id').write_parquet('users.parquet', chunk_size=2)
```

Notes:

- `write_csv`/`write_parquet` use polars under the hood. For chunked Parquet, files are created with a numeric suffix.
- `to_arrow()` requires `pyarrow` to be installed.

## API

### Properties

- `columns`: Returns a list of column names in the frame.
- `engine`: Returns the SQLAlchemy `Engine` that the frame is bound to.

### Joins

`join` keeps left-side columns unchanged and then appends columns from the right-side frame.

| Option                | Default    | Behavior                                                                                                                                                  |
|-----------------------|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `how`                 | `'inner'`  | Join type: `'inner'`, `'left'`, `'right'`, or `'full'`.                                                                                                   |
| `on`                  | `None`     | Join key or keys. Use a string/list when key names match, or a mapping like `{'left_id': 'right_id'}` when key names differ.                              |
| `left_on`, `right_on` | `None`     | Explicit left and right join keys. Use these instead of `on`.                                                                                             |
| `prefix`              | `None`     | Prefix for right-side renamed columns. Takes precedence over `suffix` and deprecated `suffixes`.                                                          |
| `suffix`              | `None`     | Suffix for right-side renamed columns. Used only when `prefix` is not supplied.                                                                           |
| `apply_to_all`        | `True`     | When using `prefix` or `suffix`, apply it to every right-side column. Set to `False` to rename only right-side columns that overlap with left-side names. |
| `duplicate_columns`   | `'rename'` | Use `'rename'` to rename overlapping right-side columns, or `'drop'` to omit them.                                                                        |
| `suffixes`            | `None`     | Deprecated. Use `suffix` or `prefix` instead. If supplied, `suffixes[-1]` is used for right-side column renaming.                                         |

Right-side naming precedence is:

1. `prefix`
2. `suffix`
3. `suffixes[-1]`
4. generated suffix such as `_right` or `_right2`

`prefix` and `suffix` are not combined. If both are supplied, `prefix` wins.

Examples:

```python
# Default: duplicate right-side names are renamed with a generated suffix.
joined = lf_users.join(lf_orders, on={'id': 'user_id'}, how='left')
# columns: id, name, age, id_right, user_id, amount

# Prefix every right-side column.
joined = lf_users.join(lf_orders, on={'id': 'user_id'}, prefix='order_')
# columns: id, name, age, order_id, order_user_id, order_amount

# Suffix every right-side column.
joined = lf_users.join(lf_orders, on={'id': 'user_id'}, suffix='_order')
# columns: id, name, age, id_order, user_id_order, amount_order

# Suffix only overlapping right-side columns.
joined = lf_users.join(
    lf_orders,
    on={'id': 'user_id'},
    suffix='_order',
    apply_to_all=False,
)
# columns: id, name, age, id_order, user_id, amount

# Drop overlapping right-side columns.
joined = lf_users.join(
    lf_orders,
    on={'id': 'user_id'},
    duplicate_columns='drop',
)
# columns: id, name, age, user_id, amount
```

### Materialization & Execution

- `collect(limit=None, infer_schema_length=200)`: Materializes the lazy query into a polars `DataFrame`.
- `to_arrow(limit=None)`: Materializes the query into a `pyarrow.Table`.
- `collect_batches(chunk_size=10_000)`: Returns an iterator of polars `DataFrame` chunks.
- `iter_rows(named=False, chunk_size=10_000)`: Returns an iterator of row tuples (or dicts if `named=True`).
- `explain()`: Returns the rendered SQL string for the query.

### I/O Helpers

- `write_parquet(file, chunk_size=None, start_index=0, **kwargs)`: Writes the result to one or more Parquet files.
- `write_csv(file, chunk_size=None, **kwargs)`: Writes the result to a CSV file.

### Advanced

- immutability: every transform (`select`, `filter`, `with_columns`, `join`, `sort`, `limit`, `group_by`) returns a new
  `LazyBearFrame`.
- `to_select()` returns the current SQLAlchemy `Select` if you need to interop with SQLAlchemy APIs directly.
- join column naming: overlapping right-side columns are suffixed with `_y` by default; If you chain multiple joins that
  would reuse the same labels, pass custom `suffixes` on later joins to keep names unique.
- case sensitivity: `scan_table(..., lowercase=True)` exposes columns as lowercase labels by default. Set
  `lowercase=False` to preserve database-reflected casing.
- `explain()` returns the rendered SQL string; if supported, literal binds are inlined.

#### Database-specific result cleaning

`lazybear` applies a small database-specific cleaning step after SQL results are materialized into a polars `DataFrame`.

Currently, this is used for Teradata connections, including connections using the `teradatasql` driver. Teradata
character datatypes may return values padded with trailing whitespace. When `lazybear` detects a Teradata connection,
trailing whitespace is stripped from string columns in collected results.

This applies to result materialization methods such as:

- `collect()`
- `collect_batches()`
- `iter_rows()`, through `collect_batches()`

For example, a Teradata value like:

```
python
"ABC   "
```

is returned as:

```
python
"ABC"
```

This cleaning is intentionally implemented through a dispatch layer so that future database-specific cleanup behavior
can be added in one place. For example, other dialects could eventually normalize driver-specific string padding,
timestamp quirks, binary values, or other result-formatting issues before returning a polars `DataFrame`.

**Notes:**

- The Teradata cleanup only affects string columns.
- It strips trailing whitespace, not leading whitespace.
- Non-Teradata connections currently return results unchanged by this cleaning step.

### Minimal example

```python
import sqlalchemy as sa
from lazybear import scan_table, col

engine = sa.create_engine('sqlite:///:memory:')
# ... create a table `users` with columns id, name, age ...
lf = scan_table('users', engine)
df = lf.filter(col('age') >= 30).select('id', 'name').collect()
print(df)
```

## License

This project is licensed under the MIT License. See `LICENSE.txt` or https://kpwhri.mit-license.org for the full text.
