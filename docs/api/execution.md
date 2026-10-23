

# Execution and Output

[Back to API reference](../api.md)

lazybear queries are lazy. They are executed only when materialized with methods such as `collect`, `collect_batches`,
`iter_rows`, or writer methods.

## Contents

- [`collect`](#collect)
- [`collect_batches`](#collect_batches)
- [`iter_rows`](#iter_rows)
- [`explain`](#explain)
- [`to_select`](#to_select)
- [`write_parquet`](#write_parquet)
- [`write_parquet_dir`](#write_parquet_dir)
- [`write_csv`](#write_csv)

---

<a name="collect"></a>
## `collect(limit=None, *, infer_schema_length=200)`

[Back to contents](#contents)

Execute the query and return a polars `DataFrame`.

### Parameters

- `limit`: Optional row limit.
- `infer_schema_length`: Passed to polars when constructing the result dataframe.

### Notes

Some database dialects may apply result cleaning after materialization. For example, teradata string columns may have
trailing whitespace stripped to account for character datatype padding.

### Examples

Collect all rows:
```python
df = users.collect()
```
Collect with a limit:
```python
df = users.collect(limit=100)
```
Build a query and collect:
```python
from lazybear import col

df = (
    users
    .filter(col('age') >= 30)
    .select('id', 'name', 'age')
    .sort('age', descending=True)
    .collect()
)
```
---

<a name="collect_batches"></a>
## `collect_batches(chunk_size=10_000)`

[Back to contents](#contents)

Stream query results in batches, yielding polars `DataFrame` chunks.

### Parameters

- `chunk_size`: Number of rows to fetch per batch. Must be positive.

### Notes

Some database dialects may apply result cleaning after materialization. For example, teradata string columns may have
trailing whitespace stripped to account for character datatype padding.

### Examples

Iterate over batches:
```python
for batch in users.collect_batches(chunk_size=5_000):
    print(batch.shape)
```
Process batches incrementally:
```python
total_rows = 0

for batch in users.collect_batches(chunk_size=10_000):
    total_rows += batch.height
```
---

<a name="iter_rows"></a>
## `iter_rows(named=False, chunk_size=10_000)`

[Back to contents](#contents)

Iterate over result rows.

### Parameters

- `named`: If `False`, yield tuples. If `True`, yield dictionaries.
- `chunk_size`: Number of rows to fetch per batch. Must be positive.

### Notes

Some database dialects may apply result cleaning after materialization. For example, teradata string columns may have
trailing whitespace stripped to account for character datatype padding.

### Examples

Iterate as tuples:
```python
for row in users.iter_rows():
    print(row)
```
Iterate as dictionaries:
```python
for row in users.iter_rows(named=True):
    print(row['name'])
```
Use a custom chunk size:
```python
for row in users.iter_rows(named=True, chunk_size=5_000):
    process(row)
```
---

## `explain()`

[Back to contents](#contents)

Return the generated sql query as a string.

### Examples

Inspect generated sql:
```python
sql = users.filter(age=41).explain()

print(sql)
```
Debug a larger pipeline:
```python
query = (
    users
    .filter(age=41)
    .select('id', 'name')
)

print(query.explain())
```
---

## `to_select()`

[Back to contents](#contents)

Return the underlying sqlalchemy `Select` object.

This is an advanced escape hatch for integrating with sqlalchemy directly.

### Examples

Get the sqlalchemy select:
```python
stmt = users.filter(age=41).to_select()
```
Execute manually:
```python
stmt = users.select('id', 'name').to_select()

with engine.connect() as conn:
    rows = conn.execute(stmt).fetchall()
```
---

<a name="write_parquet"></a>
## `write_parquet(file, chunk_size=None, start_index=0, **kwargs)`

[Back to contents](#contents)

Write query results to parquet.

### Parameters

- `file`: Output path.
- `chunk_size`: Optional chunk size. If provided, output may be split across multiple files.
- `start_index`: Starting index for chunked output filenames.
- `**kwargs`: Additional writer options.

### Examples

Write one parquet file:
```python
users.write_parquet('users.parquet')
```
Write filtered results:
```python
users.filter(age=41).write_parquet('users_age_41.parquet')
```
Write in chunks (this will write to `users-000000.parquet`, `users-000001.parquet`, etc.:
```python
users.write_parquet(
    'users.parquet',
    chunk_size=10_000,
)
```
---

<a name="write_parquet_dir"></a>
## `write_parquet_dir(outdir, chunk_size=10_000, start_index=0, prefix='part', **kwargs)`

[Back to contents](#contents)

Write query results to directory of parquet files.

### Parameters

- `outdir`: Output directory/folder.
- `chunk_size`: Optional chunk size. If provided, output may be split across multiple files.
- `start_index`: Starting index for chunked output filenames.
- `prefix`: For parquet filename, use this prefix (defaults to `part` so `part-000001.parquet`)
- `**kwargs`: Additional writer options.

### Examples

Write to directory (default chunk size of 10_000):
```python
users.write_parquet('users')
```
Write filtered results:
```python
users.filter(age=41).write_parquet('users_age_41.parquet')
```
Write in chunks:
```python
users.write_parquet(
    'users.parquet',
    chunk_size=10_000,
)
```
---

<a name="write_csv"></a>
## `write_csv(file, chunk_size=None, **kwargs)`

[Back to contents](#contents)

Write query results to csv.

### Parameters

- `file`: Output path.
- `chunk_size`: Optional chunk size.
- `**kwargs`: Additional writer options.

### Examples

Write one csv file:
```python
users.write_csv('users.csv')
```
Write selected results:
```python
users.select('id', 'name').write_csv('user_names.csv')
```
Write in chunks:
```python
users.write_csv(
    'users.csv',
    chunk_size=10_000,
)
```
[Back to API reference](../api.md)
