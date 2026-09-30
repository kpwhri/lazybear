# Joins

LazyBear supports SQL-style joins between two lazy frames. Joins are useful when you have related data in separate
tables and want to combine them into one result. The simple form is `join`. To join with inequalities/date ranges, use [`join_where`](#join-with-equality-and-range-predicates).

```python
from lazybear import scan_table, col

users = scan_table('users', engine)
orders = scan_table('orders', engine)
```

Example tables:

`users`

| id | name  | age |
|----|-------|-----|
| 1  | Ahti  | 30  |
| 2  | Kalma | 28  |
| 3  | Ukko  | 41  |

`orders`

| id | user_id | amount |
|----|---------|--------|
| 10 | 1       | 12.5   |
| 11 | 1       | 7.5    |
| 12 | 2       | 99.0   |

## Basic join with different key names

Use `left_on` and `right_on` when the join columns have different names.

```python
out = (
    users
    .join(
        orders,
        left_on='id',
        right_on='user_id',
        how='left',
    )
    .select('id', 'name', 'amount')
    .collect()
)
```

This joins:

```text
users.id == orders.user_id
```

## Join types

Use `how` to choose the join type.

```python
users.join(orders, left_on='id', right_on='user_id', how='inner')
users.join(orders, left_on='id', right_on='user_id', how='left')
users.join(orders, left_on='id', right_on='user_id', how='right')
users.join(orders, left_on='id', right_on='user_id', how='full')
```

Supported values are:

| `how`     | Meaning                                    |
|-----------|--------------------------------------------|
| `'inner'` | Keep only matching rows                    |
| `'left'`  | Keep all left rows and matching right rows |
| `'right'` | Keep all right rows and matching left rows |
| `'full'`  | Keep rows from both sides                  |

## Join when key names are the same

If both frames have the same join column name, use `on`.

```python
left = scan_table('left_table', engine)
right = scan_table('right_table', engine)

out_df = (
    left
    .join(right, on='id', how='inner')
    .collect()
)
```

This joins:

```text
left.id == right.id
```

## Join on multiple keys with the same names

Use a list with `on`.

```python
out_df = (
    left
    .join(
        right,
        on=['user_id', 'product_id'],
        how='inner',
    )
    .collect()
)
```

This joins:

```text
left.user_id == right.user_id
left.product_id == right.product_id
```

## Join on multiple keys with different names

Prefer `left_on` and `right_on` when the key names differ.

```python
out_df = (
    orders
    .join(
        order_lookup,
        left_on=['user_id', 'product_id'],
        right_on=['lookup_user_id', 'lookup_product_id'],
        how='inner',
    )
    .select('id', 'user_id', 'product_id', 'amount')
    .collect()
)
```

This joins keys by position:

```text
orders.user_id    == order_lookup.lookup_user_id
orders.product_id == order_lookup.lookup_product_id
```

The first item in `left_on` matches the first item in `right_on`, the second matches the second, and so on.

## Join with equality and range predicates

Use `join_where` when matching requires inequalities or a mix of equality and
inequality predicates. For example, this is equivalent to SQL
`ON events.id = ranges.id AND events.date BETWEEN ranges.start AND ranges.end`:

```python
out_df = (
    events
    .join_where(
        ranges,
        col('id') == col('id_right'),  # n.b., dupe column names will have `_right` appended
        col('date') >= col('start'),
        col('date') <= col('end'),
        how='inner',
    )
    .collect()
)
```

Predicates are combined with `AND`. The date boundaries above are inclusive.
`join_where` supports `inner`, `left`, and `right` joins.

By default, overlapping right-side columns receive a generated suffix such as
`_right`. Use that renamed column inside the predicate. As with `join`, an
explicit prefix or suffix applies to every right-side column by default:

```python
events.join_where(
    ranges,
    col('id') == col('id_range'),  # takes suffix `_range` rather that `_right` due to suffix arg being specified
    col('date') >= col('start'),
    col('date') <= col('end'),
    suffix='_range',
    apply_to_all=False,
)
```

Set `duplicate_columns='drop'` to omit selected right-side columns from the
result. They remain available under their renamed predicate names while the
join condition is compiled. `join_where` intentionally does not support the
deprecated `suffixes` argument.

## Dict form for different key names

You can also use a dictionary with `on`.

```python
out_df = (
    users
    .join(
        orders,
        on={'id': 'user_id'},
        how='left',
    )
    .select('id', 'name', 'amount')
    .collect()
)
```

This means:

```text
users.id == orders.user_id
```

For multiple keys:

```python
out_df = (
    orders
    .join(
        order_lookup,
        on={
            'user_id': 'lookup_user_id',
            'product_id': 'lookup_product_id',
        },
        how='inner',
    )
    .collect()
)
```

This means:

```text
orders.user_id    == order_lookup.lookup_user_id
orders.product_id == order_lookup.lookup_product_id
```

## Selecting columns after a join

After joining, use `select` to choose the columns you want.

```python
out_df = (
    users
    .join(orders, left_on='id', right_on='user_id', how='left')
    .select('id', 'name', 'amount')
    .collect()
)
```

## Handling duplicate column names

If both frames have columns with the same name, LazyBear keeps the left column name and renames overlapping right
columns.

You can control right-side names with `prefix`:

```python
out_df = (
    users
    .join(
        orders,
        left_on='id',
        right_on='user_id',
        how='left',
        prefix='order_',
    )
    .select('id', 'name', 'order_id', 'order_amount')
    .collect()
)
```

Or with `suffix`:

```python
out_df = (
    users
    .join(
        orders,
        left_on='id',
        right_on='user_id',
        how='left',
        suffix='_order',
    )
    .select('id', 'name', 'id_order', 'amount_order')
    .collect()
)
```

## Dropping duplicate right-side columns

Use `duplicate_columns='drop'` to omit overlapping right-side columns.

```python
out_df = (
    users
    .join(
        orders,
        left_on='id',
        right_on='user_id',
        how='left',
        duplicate_columns='drop',
    )
    .collect()
)
```

## Common patterns

### Users with orders

```python
out_df = (
    users
    .join(orders, left_on='id', right_on='user_id', how='left')
    .select('id', 'name', 'amount')
    .collect()
)
```

### Orders with products

```python
orders = scan_table('orders', engine)
products = scan_table('products', engine)

out_df = (
    orders
    .join(products, left_on='product_id', right_on='id', how='left', prefix='product_')
    .select('id', 'user_id', 'amount', 'product_name')
    .collect()
)
```

### Joining three tables

```python
users = scan_table('users', engine)
orders = scan_table('orders', engine)
products = scan_table('products', engine)

out_df = (
    users
    .join(orders, left_on='id', right_on='user_id', how='left', prefix='order_')
    .join(products, left_on='order_product_id', right_on='id', how='left', prefix='product_')
    .select(
        'id',
        'name',
        'order_id',
        'order_amount',
        'product_name',
    )
    .collect()
)
```

## Summary

Use:

```python
join(other, on='id')
```

when both sides use the same key name.

Use:

```python
join(other, on=['key1', 'key2'])
```

when both sides use the same multiple key names.

Use:

```python
join(other, left_on='id', right_on='user_id')
```

when the key names differ.

Use:

```python
join(
    other,
    left_on=['user_id', 'product_id'],
    right_on=['lookup_user_id', 'lookup_product_id'],
)
```

when multiple key names differ.

You can also use dictionary syntax:

```python
join(other, on={'left_key': 'right_key'})
```

Or for multiple keys:
```python
join(other, on={'left_key_1': 'right_key_1', 'left_key_2': 'right_key_2'})
```
