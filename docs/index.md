# LazyBear

LazyBear is a polars-flavored lazy SQL query builder on top of sqlalchemy. It lets you compose familiar
transformations like `select`, `filter`, `with_columns`, `join`, `group_by().agg(...)`, and `sort`/`order_by` against a
SQL database, and then materialize results to polars or arrow (or stream them in batches).

The purpose of this library is to provide lazy, polars-like access to a single sql database server, providing multi-site
(and multi-server) stable code while pushing most memory-intensive operations to the remote server.

- [Quickstart](index.md#quickstart)
- [API Reference](api.md)
- [Expressions & Namespaces](expressions.md)

## Installation

```bash
pip install lazybear-polars
```

## Quickstart

```python
import sqlalchemy as sa
from lazybear import scan_table, col

# Create a sqlalchemy engine
engine = sa.create_engine('sqlite:///:memory:')

# Scan a table lazily
lf = scan_table('users', engine)

# Chain transformations and materialize
df = (
    lf
    .filter(col('age') >= 30)
    .select('id', 'name')
    .collect()
)
print(df)
```
