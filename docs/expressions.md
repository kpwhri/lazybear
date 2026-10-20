# Expressions & Namespaces

[Back to documentation index](index.md)

`lazybear` expressions (`Expr`) define transformations, filters, conditional logic, and computed columns. They mimic parts
of the polars expression API, but compile to SQLAlchemy expressions for execution on the database server.

## Table of Contents

- [Basic Expressions](#basic-expressions)
    - [`col(name)`](#colname)
    - [`lit(value)`](#litvalue)
    - [`coalesce(*values)`](#coalescevalues)
    - [
      `when(*predicates, **constraints).then(value).otherwise(value)`](#when)
- [Operators](#operators)
    - [Comparison Operators](#comparison-operators)
    - [Arithmetic Operators](#arithmetic-operators)
    - [Boolean Operators](#boolean-operators)
- [Null Checks](#null-checks)
    - [`is_null()`](#is_null)
    - [`is_not_null()`](#is_not_null)
- [Membership Checks](#membership-checks)
    - [`is_in(other)`](#is_inother)
- [Aliases](#aliases)
    - [`alias(name)`](#aliasname)
- [String Namespace (`.str`)](#string-namespace-str)
    - [`contains(pattern, literal=True)`](#containspattern-literaltrue)
    - [`contains_any(patterns, literal=True)`](#contains_anypatterns-literaltrue)
    - [`starts_with(prefix, ascii_case_insensitive=False)`](#starts_with)
    - [`starts_with_any(prefixes)`](#starts_with_any)
- [Callable Proxy (`.startswith`)](#callable-proxy-startswith)
- [Examples](#examples)
    - [Complex Filtering](#complex-filtering)
    - [String Filtering](#string-filtering)
    - [Subqueries](#subqueries)

## Basic Expressions

### `col(name)`

References a column by name.

```python
import lazybear as lb

lf.select(lb.col('age'))
```

In many places where an expression is accepted, a string column name can also be used directly.

```python
lf.select('age')
```

### `lit(value)`

Creates a literal value expression.

```python
import lazybear as lb

lf.with_columns(
    status=lb.lit('active')
)
```

Use `lit()` when you want to make it explicit that a value is a literal expression.

### `coalesce(*values)`

Returns the first non-null value from the provided expressions or literals.

```python
import lazybear as lb

lf.with_columns(
    lb.coalesce(
        lb.col('nickname'),
        lb.col('first_name'),
        'unknown',
    ).alias('display_name'),
    lb.coalesce(lb.col('score'), 0).alias('score_filled'),
)
```

This is equivalent to SQL like:

```
sql
COALESCE(nickname, first_name, 'unknown')
COALESCE(score, 0)
```

Use `coalesce()` to replace nulls or choose the first available value across several columns.

```python
lf.filter(
    lb.coalesce(lb.col('score'), lb.col('bonus'), 0) > 75
)
```

`coalesce()` requires at least one argument.

<a id="when"></a>
### `when(*predicates, **constraints).then(value).otherwise(value)`

Builds a conditional expression, similar to Polars `when().then().otherwise()` or SQL `CASE WHEN`.

```python
import lazybear as lb

lf.with_columns(
    lb.when(lb.col('age') < 30)
    .then('young')
    .otherwise('older')
    .alias('age_group')
)
```

This is equivalent to SQL like:

```
sql
CASE
  WHEN age < 30 THEN 'young'
  ELSE 'older'
END
```

`when()` starts a conditional expression. `then()` provides the value to use when the condition matches. `otherwise()`
provides the fallback value.

If `otherwise()` is omitted, the fallback value is `NULL`.

```python
lf.select(
    lb.when(lb.col('age') < 30)
    .then('young')
    .alias('age_group')
)
```

Multiple conditions can be chained. Conditions are evaluated in order, and the first matching condition wins.

```python
lf.with_columns(
    lb.when(lb.col('score') >= 90)
    .then('excellent')
    .when(lb.col('score') >= 70)
    .then('passing')
    .otherwise('failing')
    .alias('score_group')
)
```

Values passed to `then()` and `otherwise()` may be literals, LazyBear expressions, or SQLAlchemy expressions.

```python
lf.with_columns(
    lb.when(lb.col('age') < 30)
    .then(lb.col('age') * 2)
    .otherwise(lb.col('age'))
    .alias('adjusted_age')
)
```

String values in `then()` and `otherwise()` are treated as literal strings, not column names.

```python
lf.select(
    lb.when(lb.col('age') < 30)
    .then('name')
    .otherwise('age')
    .alias('result')
)
```

The above returns the literal strings `'name'` and `'age'`.

Multiple predicates passed to `when()` are combined with `AND`.

```python
lf.with_columns(
    lb.when(
        lb.col('age') >= 40,
        lb.col('name') == 'Tellervo',
    )
    .then(True)
    .otherwise(False)
    .alias('matches')
)
```

Keyword constraints can be used as shorthand for equality checks.

```python
lf.with_columns(
    lb.when(age=41, name='Ukko')
    .then('match')
    .otherwise('other')
    .alias('result')
)
```

This is equivalent to:

```python
lf.with_columns(
    lb.when(
        lb.col('age') == 41,
        lb.col('name') == 'Ukko',
    )
    .then('match')
    .otherwise('other')
    .alias('result')
)
```

A keyword constraint with `None` checks for SQL `NULL`.

```python
lf.with_columns(
    lb.when(score=None)
    .then('missing')
    .otherwise('present')
    .alias('score_status')
)
```

This is equivalent to checking:

```python
lb.col('score').is_null()
```

`when()` requires at least one predicate or keyword constraint.

```python
lb.when()
# raises ValueError
```

## Operators

### Comparison Operators

Standard Python comparison operators are supported:

- `==`
- `!=`
- `>`
- `>=`
- `<`
- `<=`

```python
lf.filter(lb.col('age') >= 18)
```

### Arithmetic Operators

Standard arithmetic operators are supported:

- `+`
- `-`
- `*`
- `/`
- `%`

```python
lf.with_columns(
    age_next_year=lb.col('age') + 1
)
```

### Boolean Operators

Use `&` for `AND` and `|` for `OR`.

```python
lf.filter(
    (lb.col('age') > 30) & (lb.col('name') == 'Ahti')
)
```

Use parentheses around each comparison when combining expressions with `&` or `|`.

## Null Checks

### `is_null()`

Checks whether an expression is `NULL`.

```python
lf.filter(lb.col('amount').is_null())
```

### `is_not_null()`

Checks whether an expression is not `NULL`.

```python
lf.filter(lb.col('amount').is_not_null())
```

## Membership Checks

### `is_in(other)`

Checks whether a value is present in another collection or subquery.

Supported inputs include:

- Python iterables, such as lists, tuples, or sets
- Polars Series
- SQLAlchemy expressions or selectables
- LazyBearFrame subqueries

```python
import lazybear as lb

lf.filter(lb.col('id').is_in([1, 2, 3]))
```

With a subquery:

```python
young_users = (
    lf
    .filter(lb.col('age') < 30)
    .select('id')
)

lf.filter(lb.col('id').is_in(young_users))
```

An empty iterable produces a false condition.

```python
lf.filter(lb.col('id').is_in([]))
```

**Warning:** `is_in()` may generate a SQL `IN (...)` clause. Some databases limit how many values can be included in
such a clause. For very large inputs, consider loading values into a temporary table or processing them in batches.

## Aliases

### `alias(name)`

Renames an expression.

```python
lf.select(
    (lb.col('age') * 2).alias('double_age')
)
```

Aliases are commonly used in `select()` and `with_columns()`.

```python
lf.with_columns(
    (lb.col('score') + lb.col('bonus')).alias('total_score')
)
```

## String Namespace (`.str`)

The `.str` namespace provides string-specific expression methods.

### `contains(pattern, literal=True)`

Checks whether a string contains a pattern.

Case sensitivity depends on the underlying database collation.

If `literal=True`, which is the default, `pattern` is treated as a literal substring.

```python
lf.filter(lb.col('name').str.contains('hti'))
```

If `literal=False`, `pattern` is treated as a SQL `LIKE` pattern.

```python
lf.filter(lb.col('name').str.contains('%hti%', literal=False))
```

### `contains_any(patterns, literal=True)`

Checks whether a string contains any of the provided patterns.

```python
lf.filter(
    lb.col('name').str.contains_any(['Ah', 'Ka'])
)
```

<a id="starts_with"></a>
### `starts_with(prefix, ascii_case_insensitive=False)`

Checks whether a string starts with the given prefix.

```python
lf.filter(lb.col('name').str.starts_with('A'))
```

If `ascii_case_insensitive=True`, LazyBear performs a case-insensitive comparison by applying `UPPER()` to both sides.

```python
lf.filter(
    lb.col('name').str.starts_with('a', ascii_case_insensitive=True)
)
```

Note that this only handles ASCII characters reliably on most databases.

<a id="starts_with_any"></a>
### `starts_with_any(prefixes)`

Checks whether a string starts with any of the provided prefixes.

```python
lf.filter(
    lb.col('name').str.starts_with_any(['A', 'K'])
)
```

## Callable Proxy (`.startswith`)

For convenience, `col('x').startswith('a')` is available as a shortcut for `col('x').str.starts_with('a')`.

```python
lf.filter(lb.col('name').startswith('A'))
```

## Examples

### Complex Filtering

Combine multiple conditions with boolean operators.

```python
import lazybear as lb

users.filter(
    (lb.col('age') > 30) & (lb.col('age') < 60)
)
```

### String Filtering

Use the string namespace for string-specific filters.

```python
import lazybear as lb

users.filter(lb.col('name').str.contains('a', literal=True))
users.filter(lb.col('name').str.starts_with_any(['a', 'k']))
```

### Subqueries

Use `is_in()` with another `LazyBearFrame`.

```python
import lazybear as lb

young_ids = (
  lb.scan_table('users', engine)
  .filter(lb.col('age') < 30)
  .select('id')
)

users.filter(
  lb.col('id').is_in(young_ids)
)
```
