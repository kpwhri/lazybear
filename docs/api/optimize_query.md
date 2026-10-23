---
layout: default
title: Experimental SQL Optimization
description: Optionally optimize lazybear queries with sqlglot when collecting polars DataFrames.
---

# Experimental SQL Optimization

> **Experimental feature:** SQL query optimization is opt-in. SQL rewriting can potentially change query behavior or
> fail for some expressions, dialects, and parameter types. Compare results with the default execution path before
> relying
> on it.

Queries can optionally be passed through [sqlglot](https://github.com/tobymao/sqlglot) before execution. This lets you
explore whether sqlglot's transformations simplify a query or improve its performance, while continuing to use
sqlalchemy for database access and polars for the resulting DataFrames.

The feature is available through the `optimize` argument to `collect()` and `collect_batches()`:

- `optimize=None` (the default): execute the original sqlalchemy `Select`, without sqlglot rewriting.
- `optimize='sqlite'` (or another [sqlglot dialect](#supported-dialects)): compile, optimize, and execute SQL in that
  dialect.

sqlglot rewrites SQL **before it is sent to the database**. It does not replace the database's own query planner, and a
rewritten query is not necessarily faster. Also, be sure to test the query to ensure similar results. For example, limit
the query to only a few rows (e.g., `limit(10)`) under `optimize=None` and `optimize='tsql'`.

You can also visualize the different queries using `temp_df.explain()` vs `temp_df.explain(optimize='tsql')`.

## Using `collect()`

The default is unchanged:

```python
import lazybear as lb

temp_df = lb.scan_df(df, eng)
temp_df = temp_df.filter(lb.col('age') > 30).select('id', 'age')

# normal execution: allow server's sql optimizer to work on it
df = temp_df.collect()

# experimental: optimize before submission
optimized_df = temp_df.collect(optimize='sqlite')
```

For initial testing, compare both results explicitly. Sorting helps avoid differences caused only by unspecified row
ordering:

```python
assert df.sort('id').equals(optimized_df.sort('id'))
```

For a PostgreSQL connection, use `optimize='postgres'`. For Microsoft SQL Server, use `optimize='tsql'` (not `mssql`).
The `optimize` value names the **sqlglot dialect**, not the sqlalchemy dialect.

## Using `collect_batches()`

The same option is supported when fetching results in chunks:

```python
temp_df = lb.scan_df(df, engine)
temp_df = temp_df.filter(lb.col('age') > 30).select('id', 'age')

# Default: yield DataFrames in batches, without optimization.
for batch in temp_df.collect_batches(chunk_size=10_000):
    do_something(batch)

# Experimental: optimize the SQL before executing it.
for batch in temp_df.collect_batches(chunk_size=10_000, optimize='sqlite'):
    do_something(batch)
```

Optimization happens **once per query**, before execution. It does not optimize each DataFrame batch. `chunk_size`
continues to control the size of batches requested from the database; it does not change the sqlglot optimization rules.

Use `collect()` when you want the full result as one polars DataFrame, or `collect_batches()` when you want to process
the result incrementally.

## Using `iter_rows()`

The default is unchanged:

```python
import lazybear as lb

temp_df = lb.scan_df(df, eng)
temp_df = temp_df.filter(lb.col('age') > 30).select('id', 'age')

# normal execution: allow server's sql optimizer to work on it
for row in temp_df.iter_rows():
    print(row)

# experimental: optimize before submission
for row in temp_df.iter_rows(optimize='sqlite'):
    print(row)
```

For initial testing, compare both results explicitly.

## Supported dialects

Pass a **sqlglot dialect name** directly to `optimize`. Rather than maintaining application-specific aliases, this API
follows sqlglot's own names. These are the built-in names documented in recent sqlglot releases (availability may vary
with your installed version):

|                               | Dialect names                                                                                                        |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------|
| Common databases              | `sqlite`, `postgres`, `mysql`, `tsql`, `oracle`, `duckdb`                                                            |
| Cloud and analytical engines  | `athena`, `bigquery`, `clickhouse`, `databricks`, `redshift`, `snowflake`, `trino`                                   |
| Spark and related SQL engines | `hive`, `presto`, `spark`, `spark2`, `doris`, `dremio`, `drill`, `druid`, `starrocks`                                |
| Additional dialects           | `dax`, `dune`, `exasol`, `fabric`, `materialize`, `prql`, `risingwave`, `singlestore`, `solr`, `tableau`, `teradata` |

In particular, some names differ from sqlalchemy's:

| Database             | sqlalchemy dialect name | Use with sqlglot      |
|----------------------|-------------------------|-----------------------|
| Microsoft SQL Server | `mssql`                 | `optimize='tsql'`     |
| PostgreSQL           | `postgresql`            | `optimize='postgres'` |
| SQLite               | `sqlite`                | `optimize='sqlite'`   |
| MySQL                | `mysql`                 | `optimize='mysql'`    |
| DuckDB               | `duckdb`                | `optimize='duckdb'`   |
| Oracle               | `oracle`                | `optimize='oracle'`   |

sqlglot also supports separately installed **dialect plugins**. The supported set, including community dialects, evolves
over time; consult the [official supported dialect list](https://github.com/tobymao/sqlglot#supported-dialects)
and [dialect API documentation](https://sqlglot.com/sqlglot/dialects) for your installed version.

**Important:** A dialect being supported by sqlglot does not guarantee that every query written in it can be executed
through lazybear. Some dialects are specialized or are not general-purpose database sql dialects. Use
a dialect that matches the sql generated for your database.

## What happens internally?

When optimization is requested, the collection path conceptually:

1. Builds the query as a sqlalchemy `Select`.
2. Compiles the `Select` into SQL text using the engine's SQL dialect, while retaining bound parameter values.
3. Parses and rewrites that text with sqlglot using the requested dialect.
4. Converts the optimized SQL back into an executable sqlalchemy statement, preserving compatible parameter bindings.
5. Executes it through the existing sqlalchemy connection and returns polars DataFrames as usual.

When `optimize=None`, these rewriting steps are skipped and the original `Select` is executed directly.

Neither parsing nor rewriting requires a running SQL Server, PostgreSQL, or other database instance. **Executing** the
optimized query still requires a compatible database connection.

## Limitations and recommendations

- **Experimental correctness:** Check both data values and column names. Rewrites may change projections, aliases,
  ordering, or database-specific behavior.
- **Bound parameters and types:** SQL text does not preserve all of sqlalchemy's bind-processor and result-type
  metadata. Expanding `IN` parameters, custom types, UUIDs, dates, and other complex bindings may require additional
  support. Do not inline untrusted values into SQL as a workaround.
- **Schema-dependent rules:** sqlglot can perform more accurate or extensive transformations when supplied with table
  and column schema information. A query may remain unchanged or fail optimization if required metadata is unavailable.
- **Dialect compatibility:** Select the sqlglot dialect that corresponds to the SQL being compiled and executed. A
  successful parse is not proof that the database will accept the generated SQL.
- **Performance:** A different SQL string is not evidence of a faster query. Compare actual execution plans and timings
  on representative data.
- **Query ordering:** Compare DataFrames after sorting by a stable key unless the query explicitly specifies an
  ordering.

The recommended starting point is `optimize=None` for normal use and `optimize='<dialect>'` for opt-in experiments with
known queries. Test equivalence on your own database and sqlalchemy driver before enabling optimization more broadly.
For example, limit the query to only a few rows (e.g., `limit(10)`) under `optimize=None` and `optimize='tsql'`.

