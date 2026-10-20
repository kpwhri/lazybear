import polars as pl
import pytest

from lazybear import col, scan_table


def test_filter_single_predicate(sqlite_engine):
    lf = scan_table('users', sqlite_engine)

    out = lf.filter(col('age') > 30).collect().sort('id')

    assert out.to_dicts() == [
        {'id': 3, 'name': 'Tellervo', 'age': 41},
        {'id': 4, 'name': 'Ukko', 'age': 41},
    ]


def test_filter_multiple_predicates_are_anded(sqlite_engine):
    lf = scan_table('users', sqlite_engine)

    out = lf.filter(
        col('age') == 41, col('name') == 'Ukko',
    ).collect()

    assert out.to_dicts() == [{'id': 4, 'name': 'Ukko', 'age': 41}, ]


def test_filter_constraints_are_equality_predicates(sqlite_engine):
    lf = scan_table('users', sqlite_engine)

    out = lf.filter(age=41).collect().sort('id')

    assert out.to_dicts() == [
        {'id': 3, 'name': 'Tellervo', 'age': 41},
        {'id': 4, 'name': 'Ukko', 'age': 41},
    ]


def test_filter_multiple_constraints_are_anded(sqlite_engine):
    lf = scan_table('users', sqlite_engine)

    out = lf.filter(age=41, name='Tellervo').collect()

    assert out.to_dicts() == [
        {'id': 3, 'name': 'Tellervo', 'age': 41},
    ]


def test_filter_mixes_predicates_and_constraints(sqlite_engine):
    lf = scan_table('users', sqlite_engine)

    out = lf.filter(col('id') > 2, age=41).collect().sort('id')

    assert out.to_dicts() == [
        {'id': 3, 'name': 'Tellervo', 'age': 41},
        {'id': 4, 'name': 'Ukko', 'age': 41},
    ]


def test_filter_accepts_iterable_predicates(sqlite_engine):
    lf = scan_table('users', sqlite_engine)

    out = lf.filter(
        [col('age') == 41, col('id') == 4]
    ).collect()

    assert out.to_dicts() == [{'id': 4, 'name': 'Ukko', 'age': 41}]


def test_filter_accepts_iterable_predicates2(sqlite_engine):
    lf = scan_table('users', sqlite_engine)
    preds = [col('age') == 41, col('id') == 4]
    out = lf.filter(*preds).collect()

    assert out.to_dicts() == [{'id': 4, 'name': 'Ukko', 'age': 41}]


def test_filter_without_predicates_or_constraints_is_noop(sqlite_engine):
    lf = scan_table('users', sqlite_engine)

    out = lf.filter().collect().sort('id')

    assert out.to_dicts() == [
        {'id': 1, 'name': 'Ahti', 'age': 30},
        {'id': 2, 'name': 'Kalma', 'age': 28},
        {'id': 3, 'name': 'Tellervo', 'age': 41},
        {'id': 4, 'name': 'Ukko', 'age': 41},
    ]


def test_filter_constraint_unknown_column_raises_key_error(sqlite_engine):
    lf = scan_table('users', sqlite_engine)

    with pytest.raises(KeyError, match='Column'):
        lf.filter(missing=1)


def test_filter_rejects_polars_expression(sqlite_engine):
    lf = scan_table('users', sqlite_engine)

    with pytest.raises(TypeError, match='Polars expression'):
        lf.filter(pl.col('age') > 30)


def test_filter_null_predicate(sqlite_engine):
    lf = scan_table('heroes', sqlite_engine)

    out = lf.filter(
        col('first_name').is_null()
    ).select('id').collect().sort('id')

    assert out.to_dicts() == [
        {'id': 2},
        {'id': 3},
        {'id': 6},
        {'id': 8},
    ]


def test_filter_string_constraint(sqlite_engine):
    lf = scan_table('products', sqlite_engine)

    out = lf.filter(category='myth').collect().sort('id')

    assert out.to_dicts() == [
        {'id': 102, 'name': 'hiisi', 'category': 'myth'},
        {'id': 103, 'name': 'louhi', 'category': 'myth'},
    ]


def test_filter_preserves_column_order_and_names(sqlite_engine):
    lf = scan_table('users', sqlite_engine)

    out = lf.filter(col('age') >= 30).collect()

    assert out.columns == ['id', 'name', 'age']
