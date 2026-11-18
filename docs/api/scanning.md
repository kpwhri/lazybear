# Creating Frames

[Back to API reference](../api.md)

LazyBear provides three ways to construct a `LazyBearFrame`: read an existing database table,
start from a SQL `SELECT` query, or use a local polars `DataFrame` in a database query.

All three accept a SQLAlchemy `Engine`. LazyBear builds SQL expressions and delegates execution to
the connected database. Results are returned as polars `DataFrame` objects when you call
[`collect`](execution.md#collect) or another execution method.

## Contents

- [`scan_table`](#scan_table) — start from a database table
- [`sql`](#sql) — start from a sql query
- [`scan_df`](#scan_df) — use an in-memory polars DataFrame in a database query (via temp table)

## Quick example

```python
import lazybear as lb
import sqlalchemy as sa

engine = sa.create_engine('sqlite:///example.db')

users = lb.scan_table('users', engine)

result_df = (
    users
    .filter(lb.col('age') >= 30)
    .select('id', 'name', 'age')
    .collect()
)
```

`scan_table` creates a SQL-backed lazy frame. Filtering and projection are translated to SQL;
only the results of `collect()` are loaded into a local polars DataFrame.

---

<a id="scan_table"></a>

## `scan_table(table_name, engine, schema=None, *, lowercase=True)`

[Back to contents](#contents)

Create a `LazyBearFrame` from an existing database table. LazyBear uses SQLAlchemy reflection
to discover the table's columns and build a queryable source.

### Parameters

| Parameter    | Type                | Description                                                                                                   |
|--------------|---------------------|---------------------------------------------------------------------------------------------------------------|
| `table_name` | `str`               | Name of the database table.                                                                                   |
| `engine`     | `sqlalchemy.Engine` | SQLAlchemy engine connected to the database.                                                                  |
| `schema`     | `str \| None`       | Optional database schema/namespace. Defaults to `None`.                                                       |
| `lowercase`  | `bool`              | Expose column names in lowercase when `True` (default); preserve reflected casing when `False`. Keyword-only. |

**Returns:** `LazyBearFrame`.

### Examples

Scan a table and apply lazy transformations:

```python
import lazybear as lb

users = lb.scan_table('users', engine)

result_df = (
    users
    .filter(lb.col('age') >= 30)
    .select('id', 'name')
    .collect()
)
```

Specify a database schema:

```python
encounters = lb.scan_table('encounters', engine, schema='clinical')
```

Preserve the database's original column capitalization:

```python
patients = lb.scan_table('Patients', engine, lowercase=False)
print(patients.columns)
```

### Notes

- SQLAlchemy reflects the table when `scan_table()` is called; it does not fetch table rows at that point.
- The default `lowercase=True` changes the *exposed column labels* in LazyBear, not the underlying database schema.
- When using `lowercase=False`, subsequent operations must reference the reflected column names, including their casing.
- Whether a `schema` value is required, and how table names are resolved, depends on the database dialect.

---

<a id="sql"></a>

## `sql(query, engine, columns=None)`

[Back to contents](#contents)

Create a `LazyBearFrame` from a SQL `SELECT` statement. Prefer **`lb.sql(...)`**, consistent with the naming of
`pl.sql(...)` in polars. It provides the same
SQL-query scanning behavior as [`scan_sql_query`](#scan_sql_query), rather than implementing
polars' complete SQL context interface.

> **Version note:** The current `sql.py` defines `sql = scan_sql_query`, but the package
> initializer does not expose `sql` as a callable `lb.sql(...)` attribute.
> Apply the accompanying `sql_export.patch` before using `lb.sql(...)`.
> Until then, use `lb.scan_sql_query(...)` with the same arguments.

### Parameters

| Parameter | Type                    | Description                                                                                                                  |
|-----------|-------------------------|------------------------------------------------------------------------------------------------------------------------------|
| `query`   | `str`                   | Raw SQL `SELECT` statement.                                                                                                  |
| `engine`  | `sqlalchemy.Engine`     | SQLAlchemy engine on which the query will run.                                                                               |
| `columns` | `Sequence[str] \| None` | Optional result-column names. If omitted, LazyBear executes the query once to inspect cursor metadata without fetching rows. |

**Returns:** `LazyBearFrame`.

### Examples

Start from a SQL query and continue with LazyBear transformations:

```python
import lazybear as lb

users = lb.sql(
    '''
    SELECT id, name, age
    FROM users
    WHERE active = 1
    ''',
    engine,
)

result_df = (
    users
    .filter(lb.col('age') >= 30)
    .select('id', 'name')
    .collect()
)
```

Provide column names explicitly to avoid the initial metadata query:

```python
adults = lb.sql(
    'SELECT id, age FROM users WHERE age >= 18',
    engine,
    columns=['id', 'age'],
)
```

Inspect the composed SQL without collecting rows:

```python
print(adults.filter(lb.col('age') >= 65).explain())
```

### Notes

- The current SQL-query scanner accepts only SQL strings whose first non-whitespace keyword is `SELECT`. Statements
  beginning with `WITH`, and DML/DDL statements, are not supported by this scanner.
- If `columns` is omitted, the query is **executed for metadata** immediately, even though no result rows are fetched.
  Supplying `columns` avoids that preliminary execution.
- The declared `columns` must match the columns returned by the SQL statement (including intended aliases).
- The query becomes a subquery for later operations. Prefer a composable `SELECT` without a trailing semicolon.
- SQL syntax and behavior are determined by the connected database. Do not interpolate untrusted values into raw SQL
  strings.

---

<a id="scan_sql_query"></a>

## `scan_sql_query(query, engine, columns=None)`

[Back to contents](#contents)

Create a `LazyBearFrame` from a raw SQL `SELECT` statement. This is the **currently implemented
name** for SQL-query scanning. Prefer [`lb.sql(...)`](#sql) once the alias is exported; keep
`scan_sql_query(...)` for compatibility with existing code.

### Parameters

The parameters and return type are the same as [`sql`](#sql): `query`, `engine`, and optional
`columns`.

### Example

```python
import lazybear as lb

users = lb.scan_sql_query(
    'SELECT id, name, age FROM users',
    engine,
    columns=['id', 'name', 'age'],
)

result_df = users.filter(lb.col('age') >= 30).collect()
```

### Notes

- `scan_sql_query()` is available in the inspected source.
- Like `sql()`, it accepts only `SELECT`-prefixed SQL and may execute the query to infer column names.
- Using `lb.sql()` is a naming preference, not a change to query execution behavior.

---

<a id="scan_df"></a>

## `scan_df(df, engine, table_name=None)`

[Back to contents](#contents)

Create a `TempLazyBearFrame` from a local polars `DataFrame` so that it can participate in
SQL-backed LazyBear operations. The DataFrame is uploaded to a temporary database table **when the query is executed**,
not when `scan_df()` is called.

### Parameters

| Parameter    | Type                | Description                                                                                                                                           |
|--------------|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
| `df`         | `polars.DataFrame`  | Local polars DataFrame to upload during execution.                                                                                                    |
| `engine`     | `sqlalchemy.Engine` | Database engine in which the temporary table will be created.                                                                                         |
| `table_name` | `str \| None`       | Optional name component for the temporary table. If omitted, LazyBear generates a name. An automatically generated suffix is added to supplied names. |

**Returns:** `TempLazyBearFrame` (a subclass of `LazyBearFrame`).

### Examples

Use a small local cohort to filter a larger database table:

```python
import polars as pl
import lazybear as lb

cohort = pl.DataFrame({'patient_id': [101, 205, 319]})

patients = lb.scan_table('patients', engine)
cohort_table = lb.scan_df(cohort, engine)

result_df = (
    patients
    .join(cohort_table, on='patient_id', how='inner')
    .select('patient_id', 'age')
    .collect()
)
```

Supply a descriptive temporary-table name:

```python
cohort_table = lb.scan_df(cohort, engine, table_name='cohort')
```

### Notes

- The local data is transferred **to the database**; the database performs the SQL join or transformation before
  returning results.
- `scan_df()` itself does not immediately create or populate a temporary table. Materialization occurs when an execution
  method such as `collect()` or `collect_batches()` runs.
- lazybear creates temporary tables on the query connection and attempts to clean them up afterward. Temporary-table
  behavior and permissions vary across database dialects.
- Use the same sqlalchemy `Engine` for frames you intend to join; joins across different servers are not supported.
- Database column types are inferred from the polars schema. Some polars types have no dedicated database-type mapping
  and fall back to string columns, so check type compatibility for less common dtypes.
- Uploading large local DataFrames can be expensive. For large inputs already stored in the database, consider
  `scan_table()` instead.

---

## Choosing a scan function

| Starting point                             | Recommended API                                               |
|--------------------------------------------|---------------------------------------------------------------|
| Existing database table                    | `lb.scan_table('users', engine)`                              |
| SQL `SELECT` statement                     | `lb.sql('SELECT ...', engine)` (after exporting the function) |
| Existing code that uses SQL scanning       | `lb.scan_sql_query('SELECT ...', engine)`                     |
| Local polars DataFrame for a database join | `lb.scan_df(df, engine)`                                      |

LazyBear expressions use `lb.col(...)`, not `pl.col(...)`, even when the query originates from
a polars DataFrame. Transformations return new lazy frames; execution occurs only when an
output method is called, except for table reflection and optional SQL-result metadata discovery.

[Back to API reference](../api.md)

