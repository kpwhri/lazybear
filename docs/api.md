# API reference

`lazybear` provides a polars-inspired lazy API for building sql queries with sqlalchemy-backed execution.

The main object is `LazyBearFrame`. It represents a lazy sql query and is usually created with `scan_table`,
`scan_sql_query`, or `scan_df`.

lazybear is intended to feel familiar to polars users, but it is not intended to be feature complete. The goal is to
push filtering, projection, joins, grouping, and limiting into the database so less data needs to be collected locally
for analysis.

## Contents

- [Creating frames](api/scanning.md)
    - [`scan_table`](api/scanning.md#scan_tabletable_name-engine-schemanone-lowercasetrue)
    - [`scan_sql_query`](api/scanning.md#scan_sql_queryquery-engine-columnsnone)
    - [`scan_df`](api/scanning.md#scan_dfdf-engine-table_namenone)
- [`LazyBearFrame`](api/lazybearframe.md)
    - [Properties](api/lazybearframe.md#properties)
    - [`select`](api/lazybearframe.md#selectitems)
    - [`filter`](api/lazybearframe.md#filterpredicates-constraints)
    - [`with_columns`](api/lazybearframe.md#with_columnsexprs-named)
    - [`drop`](api/lazybearframe.md#dropcolumns-stricttrue)
    - [`sort`](api/lazybearframe.md#sortby-more_by-descendingfalse)
    - [`order_by`](api/lazybearframe.md#order_bykeys)
    - [`limit`](api/lazybearframe.md#limitn)
- [Joins](api/joins.md)
    - [
      `join`](api/joins.md#joinother-onnone--left_onnone-right_onnone-howinner-suffixnone-prefixnone-apply_to_alltrue-duplicate_columnsrename)
    - [
      `join_where`](api/joins.md#join_whereother-predicates-howinner-suffixnone-prefixnone-apply_to_alltrue-duplicate_columnsrename)
- [Grouping and aggregation](api/grouping.md)
    - [`group_by`](api/grouping.md#group_bykeys)
    - [`GroupedLazyBearFrame.agg`](api/grouping.md#aggaggregations)
- [Execution and output](api/execution.md)
    - [`collect`](api/execution.md#collectlimitnone-infer_schema_length200)
    - [`collect_batches`](api/execution.md#collect_batcheschunk_size10_000)
    - [`iter_rows`](api/execution.md#iter_rowsnamedfalse-chunk_size10_000)
    - [`explain`](api/execution.md#explain)
    - [`to_select`](api/execution.md#to_select)
    - [`write_parquet`](api/execution.md#write_parquetfile-chunk_sizenone-start_index0-kwargs)
    - [`write_csv`](api/execution.md#write_csvfile-chunk_sizenone-kwargs)

## Quick example

```python
from lazybear import col, scan_table

users = scan_table('users', eng)

result = (
    users
    .filter(col('age') >= 30)
    .select('id', 'name', 'age')
    .sort('age', descending=True)
    .collect()
)
```

## Notes

- Use `lazybear.col`, not `polars.col` or `pl.col`, when building lazybear expressions.
- Transformations are immutable. Methods return new `LazyBearFrame` objects.
- Query execution happens only when calling methods such as `collect`, `collect_batches`, `iter_rows`, or writer
  methods.
- sql behavior can vary by database dialect, especially around string collation, date handling, and temporary tables.
