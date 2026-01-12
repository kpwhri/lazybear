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

## Installation

* LazyBear targets Python 3.10+.
* Required dependencies:
    - `sqlalchemy`
    - `polars`
* Optional:
    - `pyarrow` for `to_arrow()`

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

# import lazybear
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

# joins (left columns keep names; right overlaps get suffixed with `_y` by default)
joined_df = (
    lf_users
    .join(lf_orders, on={'id': 'user_id'}, how='left')
    .select('id', 'name', 'age', ('amount_y', col('amount')))
    .collect()
)
print(joined_df)

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

```
# single file
lf_users.order_by('id').write_csv('users.csv', include_header=True)

# append in chunks to one file
lf_users.order_by('id').write_csv('users_chunked.csv', chunk_size=2, include_header=True)
```

- Parquet

```
# single file
lf_users.collect().write_parquet('users.parquet')

# chunked parts with a common prefix like users-00000.parquet, users-00001.parquet, ...
lf_users.order_by('id').write_parquet('users.parquet', chunk_size=2)
```

Notes:

- `write_csv`/`write_parquet` use polars under the hood. For chunked Parquet, files are created with a numeric suffix.
- `to_arrow()` requires `pyarrow` to be installed.

### Advanced

- case sensitivity: `scan_table(..., lowercase=True)` exposes columns as lowercase labels by default. Set
  `lowercase=False` to preserve database-reflected casing.
- `to_select()` returns the current SQLAlchemy `Select` if you need to interop with SQLAlchemy APIs directly.
- `explain()` returns the rendered SQL string; if supported, literal binds are inlined.

### Minimal example

```
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
