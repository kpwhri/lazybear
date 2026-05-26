import pytest

import polars as pl

from lazybear import scan_table


def test_drop_single_column(sqlite_engine):
    users = scan_table('users', sqlite_engine)

    out = users.drop('age').collect()

    assert out.columns == ['id', 'name']
    assert out.height == 4


def test_drop_multiple_columns_varargs(sqlite_engine):
    users = scan_table('users', sqlite_engine)

    out = users.drop('name', 'age').collect()

    assert out.columns == ['id']
    assert out['id'].to_list() == [1, 2, 3, 4]


def test_drop_multiple_columns_list(sqlite_engine):
    users = scan_table('users', sqlite_engine)

    out = users.drop(['name', 'age']).collect()

    assert out.columns == ['id']
    assert out['id'].to_list() == [1, 2, 3, 4]


def test_drop_missing_column_raises(sqlite_engine):
    users = scan_table('users', sqlite_engine)

    with pytest.raises(KeyError, match='Column\\(s\\) not found'):
        users.drop('missing').collect()


def test_drop_non_string_column_raises(sqlite_engine):
    users = scan_table('users', sqlite_engine)

    with pytest.raises(TypeError, match='drop\\(\\) columns must be strings'):
        users.drop(123).collect()


def test_drop_all_columns_raises(sqlite_engine):
    users = scan_table('users', sqlite_engine)

    with pytest.raises(ValueError, match='cannot remove all columns'):
        users.drop('id', 'name', 'age').collect()


def test_drop_after_join(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)

    out = (
        users
        .join(orders, on={'id': 'user_id'}, how='left', prefix='order_')
        .drop('order_user_id')
        .select('id', 'name', 'order_id', 'order_amount')
        .order_by('id', 'order_id')
        .collect()
    )

    assert out.columns == ['id', 'name', 'order_id', 'order_amount']
    assert out.filter(pl.col('id') == 1).height == 2


def test_drop_missing_column_raises_by_default(sqlite_engine):
    users = scan_table('users', sqlite_engine)

    with pytest.raises(KeyError, match='Column\\(s\\) not found'):
        users.drop('missing').collect()


def test_drop_missing_column_ignored_when_strict_false(sqlite_engine):
    users = scan_table('users', sqlite_engine)

    out = users.drop('missing', strict=False).collect()

    assert out.columns == ['id', 'name', 'age']
    assert out.height == 4


def test_drop_existing_and_missing_column_when_strict_false(sqlite_engine):
    users = scan_table('users', sqlite_engine)

    out = users.drop('age', 'missing', strict=False).collect()

    assert out.columns == ['id', 'name']


def test_drop_all_existing_columns_raises_even_when_strict_false(sqlite_engine):
    users = scan_table('users', sqlite_engine)

    with pytest.raises(ValueError, match='cannot remove all columns'):
        users.drop('id', 'name', 'age', 'missing', strict=False).collect()
