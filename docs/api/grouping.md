# Grouping and aggregation

[Back to API reference](../api.md)

Use `group_by` to group rows and `agg` to compute grouped aggregations.

## Contents

- [`group_by`](#group_bykeys)
- [`GroupedLazyBearFrame.agg`](#aggaggregations)

---

## `group_by(*keys)`

[Back to contents](#contents)

Group by one or more columns.

Returns a `GroupedLazyBearFrame`.

### Parameters

- `*keys`: Column names to group by.

### Examples

Group by one column:

```python
orders.group_by('user_id')
```

Group by multiple columns:

```python
orders.group_by('user_id', 'product_id')
```

Aggregate after grouping:

```python
orders.group_by('user_id').agg(
    total_amount=('amount', 'sum'),
)
```

Collect grouped results:

```python
df = (
    orders
    .group_by('user_id')
    .agg(total_amount=('amount', 'sum'))
    .sort('user_id')
    .collect()
)
```

---

## `agg(**aggregations)`

[Back to contents](#contents)

Perform aggregations on grouped data.

### Parameters

- `**aggregations`: Keyword arguments where each key is the output column name.
- Each value is a tuple of `(column_or_expr, function)`.

Supported functions:

- `'count'`
- `'sum'`
- `'avg'`
- `'mean'`
- `'min'`
- `'max'`

### Examples

Count rows per group:

```python
orders.group_by('user_id').agg(
    order_count=('id', 'count'),
)
```

Sum a column:

```python
orders.group_by('user_id').agg(
    total_amount=('amount', 'sum'),
)
```

Compute multiple aggregations:

```python
orders.group_by('user_id').agg(
    order_count=('id', 'count'),
    total_amount=('amount', 'sum'),
    avg_amount=('amount', 'mean'),
)
```

Use an expression:

```python
from lazybear import col

orders.group_by('user_id').agg(
    double_total=(col('amount') * 2, 'sum'),
)
```

Collect the grouped result:

```python
df = (
    orders
    .group_by('user_id')
    .agg(total_amount=('amount', 'sum'))
    .sort('user_id')
    .collect()
)
```

[Back to API reference](../api.md)