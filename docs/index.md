# LazyBear

LazyBear is a polars-flavored lazy SQL query builder on top of sqlalchemy. It lets you compose familiar
transformations like `select`, `filter`, `with_columns`, `join`, `group_by().agg(...)`, and `sort`/`order_by` against a
SQL database, and then materialize results to polars or arrow (or stream them in batches).

The purpose of this library is to provide lazy, polars-like access to a single sql database server, providing multi-site
(and multi-server) stable code while pushing most memory-intensive operations to the remote server.

- [Quickstart](index.md#quickstart)
- [API Reference](api.md)
  - [LazyBearFrame API](api/lazybearframe.md)
  - [Joins](api/joins.md)
  - [Groups and Aggregation](api/grouping.md)
  - [Execution and Output](api/execution.md)
  - [(Experimental) Query Optimization](api/optimize_query.md)
- [Expressions & Namespaces](expressions.md)
- Functions
  - [Temporal Functions](temporal.md): Cross-db functions that handle dates/time
    - [Calculate Age](temporal.md#calc_age)
    - [Date Add](temporal.md#date_add)
    - [Date Difference](temporal.md#date_diff)
    - [Truncate Date](temporal.md#date_trunc)
    - [Get Date Parts](temporal.md#date_part)
    - [Between Dates](temporal.md#between)
    - [Date is in Interval](temporal.md#in_interval)
    - [Date is in Window](temporal.md#in_window)
    - [Check if Intervals Overlap](temporal.md#intervals_overlap)
- [DataTypes/Type Aliases](dtypes.md)

## Installation

```bash
pip install lazybear-polars
```

## Quickstart

```python
import sqlalchemy as sa
import lazybear as lb

# Create a sqlalchemy engine
engine = sa.create_engine('sqlite:///:memory:')

# Scan a table lazily
lf = lb.scan_table('users', engine)

# Chain transformations and materialize
df = (
    lf
    .filter(
      lb.col('age') >= 30
    )
    .select('id', 'name')
    .collect()
)
print(df)
```
