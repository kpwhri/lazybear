# Joins

[Back to API reference](../api.md)

lazybear supports two join APIs:

- [
  `join`](#join)
  for equality-key joins.
- [`join_where`](#join_where)
  for predicate joins, including non-equi joins, range joins, and effective-dated joins.

Both APIs return a new `LazyBearFrame`.

## Contents

- [Overview](#overview)
- [
  `join`](#join)
    - [Parameters](#join-parameters)
    - [Specifying join keys](#specifying-join-keys)
    - [Join strategies](#join-strategies)
    - [Right-column naming](#right-column-naming)
    - [Duplicate right-side columns](#duplicate-right-side-columns)
    - [Examples](#join-examples)
- [`join_where`](#join_where)
    - [Parameters](#join_where-parameters)
    - [Predicate rules](#predicate-rules)
    - [Referencing right-side columns](#referencing-right-side-columns)
    - [Range and effective-date joins](#range-and-effective-date-joins)
    - [Naming rules in `join_where`](#naming-rules-in-join_where)
    - [Dropping duplicate columns in `join_where`](#dropping-duplicate-columns-in-join_where)
    - [Examples](#join_where-examples)
- [Choosing between `join` and `join_where`](#choosing-between-join-and-join_where)

---

## Overview

[Back to contents](#contents)

Use `join` when rows should match by equality between one or more key columns.

Use `join_where` when the join condition needs arbitrary predicates, such as:

- equality plus date ranges
- interval overlap
- inequality joins
- multiple non-key predicates

Both frames must use the same database server.

```python
from lazybear import col, scan_table

users = scan_table('users', engine)
orders = scan_table('orders', engine)

joined = users.join(
    orders,
    on={'id': 'user_id'},
)
```

For non-equi joins:

```python
from lazybear import col

joined = orders.join_where(
    prices,
    col('product_id') == col('product_id_right'),
    col('order_date') >= col('valid_from'),
    col('order_date') <= col('valid_to'),
)
```

---

<a id="join"></a>

##

`join(other, on=None, *, left_on=None, right_on=None, how='inner', suffix=None, prefix=None, apply_to_all=True, duplicate_columns='rename')`

[Back to contents](#contents)

Join this frame with another `LazyBearFrame` using one or more equality keys.

This is the standard join API for key-based joins. It is similar in spirit to polars joins, but backed by sql generated
through sqlalchemy.

Both frames must use the same database server.

### `join` parameters

[Back to contents](#contents)

- `other`: The right-side `LazyBearFrame`.
- `on`: Join key or keys to use when both frames expose the same key names. Also accepts a mapping of
  `{left_column: right_column}` when key names differ.
- `left_on`: Left-side join key or keys. Must be used with `right_on`.
- `right_on`: Right-side join key or keys. Must be used with `left_on`.
- `how`: Join type. One of `'inner'`, `'left'`, `'right'`, or `'full'`.
- `suffix`: Suffix used to rename right-side columns. Ignored when `prefix` is provided.
- `prefix`: Prefix used to rename right-side columns. Takes precedence over `suffix`.
- `apply_to_all`: When `True` and a `prefix` or `suffix` value is used, apply it to every right-side column. When
  `False`, apply it only to right-side columns whose names overlap with left-side columns.
- `duplicate_columns`: Controls right-side columns that would duplicate left-side labels.
    - `'rename'`: Rename duplicate right-side columns. If no explicit `prefix` or `suffix` is supplied, lazybear
      generates
      a useful non-conflicting suffix such as `_right` or `_right2`.
    - `'drop'`: Omit right-side columns selected for renaming. With explicit `prefix` or `suffix` and
      `apply_to_all=True`, this drops all right-side columns. With `apply_to_all=False`, this drops only overlapping
      right-side columns.

---

### Specifying join keys

[Back to contents](#contents)

Use `on` when the key names are the same:

```python
users.join(accounts, on='user_id')
```

Use multiple keys:

```python
users.join(accounts, on=['user_id', 'region_id'])
```

Use `left_on` and `right_on` when the key names differ:

```python
users.join(
    accounts,
    left_on='id',
    right_on='user_id',
)
```

A mapping can also associate differently named keys:

```python
users.join(
    accounts,
    on={'id': 'user_id'},
)
```

Do not combine `on` with `left_on` or `right_on`.

---

### Join strategies

[Back to contents](#contents)

`how` supports:

- `'inner'`: keep matching rows only.
- `'left'`: keep every left row.
- `'right'`: keep every right row.
- `'full'`: keep rows from both sides.

Inner join:

```python
users.join(
    orders,
    on={'id': 'user_id'},
    how='inner',
)
```

Left join:

```python
users.join(
    orders,
    on={'id': 'user_id'},
    how='left',
)
```

Right join:

```python
users.join(
    orders,
    on={'id': 'user_id'},
    how='right',
)
```

Full join:

```python
users.join(
    orders,
    on={'id': 'user_id'},
    how='full',
)
```

---

### Right-column naming

[Back to contents](#contents)

When right-side column names overlap with left-side names, right-side columns may need to be renamed.

Right-side column naming precedence is:

1. `prefix`
2. `suffix`
3. generated non-conflicting suffix when `duplicate_columns='rename'`

`prefix` and `suffix` are not combined. If both are supplied, `prefix` wins.

When an explicit `prefix` or `suffix` is supplied:

- It applies to every right-side column by default.
- Set `apply_to_all=False` to rename only overlapping columns.

Without an explicit `prefix` or `suffix`, overlapping columns receive a generated suffix such as `_right` or `_right2`.

If the same key name is used on both sides, such as `on='id'`, the duplicate right key is omitted automatically.

Use a prefix for right-side columns:

```python
joined = users.join(
    orders,
    on={'id': 'user_id'},
    prefix='order_',
)
```

This produces right-side names such as `order_id`, `order_user_id`, and `order_amount`.

Use a suffix for right-side columns:

```python
joined = users.join(
    orders,
    on={'id': 'user_id'},
    suffix='_order',
)
```

Rename only overlapping columns:

```python
joined = users.join(
    orders,
    on={'id': 'user_id'},
    suffix='_order',
    apply_to_all=False,
)
```

---

### Duplicate right-side columns

[Back to contents](#contents)

Set `duplicate_columns='drop'` to omit right-side columns that would otherwise be renamed:

```python
joined = users.join(
    orders,
    on={'id': 'user_id'},
    duplicate_columns='drop',
)
```

With explicit `prefix` or `suffix` and `apply_to_all=True`, `duplicate_columns='drop'` drops all right-side columns
selected for renaming. Because explicit naming applies to all right-side columns by default, this can drop every
right-side column.

To drop only overlapping right-side columns while using an explicit suffix:

```python
joined = users.join(
    orders,
    on={'id': 'user_id'},
    suffix='_order',
    apply_to_all=False,
    duplicate_columns='drop',
)
```

---

### `join` examples

[Back to contents](#contents)

Join users to orders with differently named keys:

```python
users_with_orders = users.join(
    orders,
    on={'id': 'user_id'},
)
```

Select columns after the join:

```python
df = (
    users
    .join(orders, on={'id': 'user_id'}, prefix='order_')
    .select('id', 'name', 'order_amount')
    .collect()
)
```

Join on multiple keys:

```python
joined = left.join(
    right,
    on=['user_id', 'region_id'],
)
```

Join with differently named multiple keys:

```python
joined = left.join(
    right,
    on={
        'id': 'user_id',
        'region': 'region_id',
    },
)
```

Use `left_on` and `right_on`:

```python
joined = users.join(
    orders,
    left_on='id',
    right_on='user_id',
)
```

Drop duplicate right-side columns:

```python
joined = users.join(
    orders,
    on={'id': 'user_id'},
    duplicate_columns='drop',
)
```

---

<a id="join_where"></a>

##

`join_where(other, *predicates, how='inner', suffix=None, prefix=None, apply_to_all=True, duplicate_columns='rename')`

[Back to contents](#contents)

Join this frame with another `LazyBearFrame` using arbitrary equality or inequality predicates.

This mirrors polars `DataFrame.join_where` while compiling predicates into the sql `ON` clause. Multiple predicates are
combined with `AND`.

This is useful for non-equi joins, including effective-date and interval matching.

Both frames must use the same database server.

### `join_where` parameters

[Back to contents](#contents)

- `other`: The right-side `LazyBearFrame`.
- `*predicates`: One or more equality or inequality expressions. An iterable of expressions may also be supplied.
- `how`: Join type. One of `'inner'`, `'left'`, or `'right'`.
- `suffix`: Suffix used to rename right-side columns. Ignored when `prefix` is provided.
- `prefix`: Prefix used to rename right-side columns. Takes precedence over `suffix`.
- `apply_to_all`: When an explicit prefix or suffix is supplied, apply it to every right-side column when `True` and
  only
  overlapping columns when `False`.
- `duplicate_columns`: Rename or drop selected right-side columns. Dropped columns remain available while resolving the
  join predicates.

`join_where` supports these join strategies:

- `'inner'`: keep matching rows only.
- `'left'`: keep every left row.
- `'right'`: keep every right row.

A full join is not supported by `join_where`.

---

### Predicate rules

[Back to contents](#contents)

Predicates are lazybear expressions built with `lazybear.col`.

Multiple predicates are combined with `AND`:

```python
from lazybear import col

orders.join_where(
    prices,
    col('product_id') == col('product_id_right'),
    col('order_date') >= col('valid_from'),
    col('order_date') <= col('valid_to'),
)
```

This is equivalent to a sql `ON` clause like:

```
sql
ON orders.product_id = prices.product_id
AND orders.order_date >= prices.valid_from
AND orders.order_date <= prices.valid_to
```

To express alternatives inside a predicate, combine expressions with `|`:

```python
from lazybear import col

left.join_where(
    right,
    (col('status') == col('status_right')) | (col('fallback') == col('fallback_right')),
)
```

---

### Referencing right-side columns

[Back to contents](#contents)

Predicates use the right-column names produced by the same prefix and suffix rules used by `join`.

When a column exists on both sides, the right-side reference receives a generated suffix by default:

```python
from lazybear import col

col('id') == col('id_right')
```

Do not write:

```python
from lazybear import col

# wrong
col('id') == col('id')
```

Both expressions resolve to the left-side `id`, producing a condition equivalent to:

```
sql
left.id = left.id
```

rather than a comparison between frames.

---

### Range and effective-date joins

[Back to contents](#contents)

A common `join_where` use case is joining events or orders to effective-dated dimension rows.

For example, join orders to product prices where:

- product ids match
- `order_date` is on or after `valid_from`
- `order_date` is on or before `valid_to`

```python
from lazybear import col

priced_orders = orders.join_where(
    prices,
    col('product_id') == col('product_id_right'),
    col('order_date') >= col('valid_from'),
    col('order_date') <= col('valid_to'),
)
```

This corresponds to:

```
sql
ON orders.product_id = prices.product_id
AND orders.order_date >= prices.valid_from
AND orders.order_date <= prices.valid_to
```

Collect deterministic results by sorting explicitly:

```python
from lazybear import col

df = (
    orders
    .join_where(
        prices,
        col('product_id') == col('product_id_right'),
        col('order_date') >= col('valid_from'),
        col('order_date') <= col('valid_to'),
    )
    .sort('order_id')
    .collect()
)
```

Result row order is not guaranteed unless you call `sort` or `order_by`.

---

### Naming rules in `join_where`

[Back to contents](#contents)

`join_where` applies the same right-side naming rules as `join`.

If an explicit prefix or suffix is supplied, use the resulting names in the predicates.

Because explicit naming applies to all right-side columns by default, this example suffixes every right-side reference:

```python
from lazybear import col

orders.join_where(
    prices,
    col('product_id') == col('product_id_price'),
    col('order_date') >= col('valid_from_price'),
    col('order_date') <= col('valid_to_price'),
    suffix='_price',
)
```

To rename only overlapping right-side columns, set `apply_to_all=False`:

```python
from lazybear import col

orders.join_where(
    prices,
    col('product_id') == col('product_id_price'),
    col('order_date') >= col('valid_from'),
    col('order_date') <= col('valid_to'),
    suffix='_price',
    apply_to_all=False,
)
```

`prefix` takes precedence over `suffix` when both are supplied.

Use a prefix:

```python
from lazybear import col

orders.join_where(
    prices,
    col('product_id') == col('price_product_id'),
    col('order_date') >= col('price_valid_from'),
    col('order_date') <= col('price_valid_to'),
    prefix='price_',
)
```

---

### Dropping duplicate columns in `join_where`

[Back to contents](#contents)

Set `duplicate_columns='drop'` to omit selected right-side columns from the result:

```python
from lazybear import col

orders.join_where(
    prices,
    col('product_id') == col('product_id_right'),
    col('order_date') >= col('valid_from'),
    col('order_date') <= col('valid_to'),
    duplicate_columns='drop',
)
```

Dropped right-side columns remain available under their renamed names while the join predicates are evaluated.

This means you can still reference `col('product_id_right')` in the predicate even if the right-side duplicate
`product_id` column is dropped from the result.

---

### `join_where` examples

[Back to contents](#contents)

Inner non-equi join:

```python
from lazybear import col

joined = orders.join_where(
    prices,
    col('amount') >= col('minimum_amount'),
)
```

Left non-equi join:

```python
from lazybear import col

joined = orders.join_where(
    prices,
    col('amount') >= col('minimum_amount'),
    how='left',
)
```

Right non-equi join:

```python
from lazybear import col

joined = orders.join_where(
    prices,
    col('amount') >= col('minimum_amount'),
    how='right',
)
```

Effective-date join with duplicate right-side columns dropped:

```python
from lazybear import col

priced_orders = orders.join_where(
    prices,
    col('product_id') == col('product_id_right'),
    col('order_date') >= col('valid_from'),
    col('order_date') <= col('valid_to'),
    duplicate_columns='drop',
)
```

Select output columns after a range join:

```python
from lazybear import col

df = (
    orders
    .join_where(
        prices,
        col('product_id') == col('product_id_right'),
        col('order_date') >= col('valid_from'),
        col('order_date') <= col('valid_to'),
        duplicate_columns='drop',
    )
    .select('order_id', 'product_id', 'order_date', 'amount', 'price', 'price_label')
    .sort('order_id')
    .collect()
)
```

Use a suffix only for overlapping right-side columns:

```python
from lazybear import col

joined = orders.join_where(
    prices,
    col('product_id') == col('product_id_price'),
    col('order_date') >= col('valid_from'),
    col('order_date') <= col('valid_to'),
    suffix='_price',
    apply_to_all=False,
)
```

---

## Choosing between `join` and `join_where`

[Back to contents](#contents)

Use `join` when every join condition is equality between key columns:

```python
users.join(
    orders,
    on={'id': 'user_id'},
)
```

Use `join_where` when the join condition includes non-equality predicates:

```python
from lazybear import col

orders.join_where(
    prices,
    col('product_id') == col('product_id_right'),
    col('order_date') >= col('valid_from'),
    col('order_date') <= col('valid_to'),
)
```

In general:

- Prefer `join` for normal key joins.
- Use `join_where` for range joins, interval joins, and effective-dated lookups.
- Use `sort` or `order_by` after joining when deterministic row order is required.

[Back to API reference](../api.md)