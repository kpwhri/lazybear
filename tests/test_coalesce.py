import pytest

from lazybear import scan_table, col, coalesce, lit


def test_coalesce_with_non_null_column_preserves_original_values(sqlite_engine):
    users = scan_table('users', sqlite_engine)

    out = (
        users
        .with_columns(
            coalesce(col('age'), 0).alias('age_or_default')
        )
        .select('id', 'age_or_default')
        .order_by('id')
        .collect()
    )

    assert out['age_or_default'].to_list() == [30, 28, 41, 41]


def test_coalesce_uses_literal_when_column_is_null(sqlite_engine):
    heroes = scan_table('heroes', sqlite_engine)

    out = (
        heroes
        .with_columns(
            coalesce(col('score'), 0).alias('score_filled')
        )
        .select('id', 'score_filled')
        .order_by('id')
        .collect()
    )

    assert out['score_filled'].to_list() == [100, 0, 0, 90, 70, 0, 0, 60]


def test_coalesce_uses_second_column_when_first_column_is_null(sqlite_engine):
    heroes = scan_table('heroes', sqlite_engine)

    out = (
        heroes
        .with_columns(
            coalesce(col('score'), col('bonus')).alias('score_or_bonus')
        )
        .select('id', 'score_or_bonus')
        .order_by('id')
        .collect()
    )

    assert out['score_or_bonus'].to_list() == [100, 50, None, 90, 70, 95, None, 60]


def test_coalesce_uses_first_non_null_value_across_multiple_columns(sqlite_engine):
    heroes = scan_table('heroes', sqlite_engine)

    out = (
        heroes
        .with_columns(
            coalesce(col('nickname'), col('first_name'), col('backup_name'), 'unknown').alias('display_name')
        )
        .select('id', 'display_name')
        .order_by('id')
        .collect()
    )

    assert out['display_name'].to_list() == [
        'Väinämöinen',
        'Jouk',
        'Ilmarinen',
        'Lempi',
        'Kullervo',
        'Louhi',
        'Aino',
        'Marjatta',
    ]


def test_coalesce_uses_final_literal_when_all_previous_values_are_null(sqlite_engine):
    heroes = scan_table('heroes', sqlite_engine)

    out = (
        heroes
        .filter(col('id') == 3)
        .with_columns(
            coalesce(col('nickname'), col('first_name'), 'unknown').alias('display_name')
        )
        .select('display_name')
        .collect()
    )

    assert out['display_name'].to_list() == ['unknown']


def test_coalesce_accepts_lit_expression_as_fallback(sqlite_engine):
    heroes = scan_table('heroes', sqlite_engine)

    out = (
        heroes
        .with_columns(
            coalesce(col('score'), lit(99)).alias('score_filled')
        )
        .select('id', 'score_filled')
        .order_by('id')
        .collect()
    )

    assert out['score_filled'].to_list() == [100, 99, 99, 90, 70, 99, 99, 60]


def test_coalesce_can_be_used_in_filter(sqlite_engine):
    heroes = scan_table('heroes', sqlite_engine)

    out = (
        heroes
        .filter(coalesce(col('score'), col('bonus'), 0) > 75)
        .select('id')
        .order_by('id')
        .collect()
    )

    assert out['id'].to_list() == [1, 4, 6]


def test_coalesce_can_be_used_in_arithmetic_expression(sqlite_engine):
    heroes = scan_table('heroes', sqlite_engine)

    out = (
        heroes
        .with_columns(
            (coalesce(col('score'), 0) + coalesce(col('bonus'), 0)).alias('total_power')
        )
        .select('id', 'total_power')
        .order_by('id')
        .collect()
    )

    assert out['total_power'].to_list() == [100, 50, 0, 100, 70, 95, 0, 65]


def test_coalesce_can_be_aliased_and_selected(sqlite_engine):
    heroes = scan_table('heroes', sqlite_engine)

    out = (
        heroes
        .select(
            'id',
            coalesce(col('nickname'), col('first_name'), col('backup_name'), 'unknown').alias('name')
        )
        .order_by('id')
        .collect()
    )

    assert out.columns == ['id', 'name']
    assert out['name'].to_list() == [
        'Väinämöinen',
        'Jouk',
        'Ilmarinen',
        'Lempi',
        'Kullervo',
        'Louhi',
        'Aino',
        'Marjatta',
    ]


def test_coalesce_can_return_null_when_all_values_are_null(sqlite_engine):
    heroes = scan_table('heroes', sqlite_engine)

    out = (
        heroes
        .with_columns(
            coalesce(col('nickname'), col('first_name')).alias('maybe_name')
        )
        .select('id', 'maybe_name')
        .order_by('id')
        .collect()
    )

    assert out['maybe_name'].to_list() == ['Väinämöinen', 'Jouk', None, 'Lempi', 'Kullervo', 'Louhi', 'Aino', None]


def test_coalesce_supports_null_literal_before_fallback(sqlite_engine):
    heroes = scan_table('heroes', sqlite_engine)

    out = (
        heroes
        .with_columns(
            coalesce(None, col('nickname'), 'fallback').alias('name'),
        )
        .select('id', 'name')
        .order_by('id')
        .collect()
    )

    assert out['name'].to_list() == [
        'fallback', 'Jouk', 'fallback', 'Lempi', 'fallback', 'Louhi', 'fallback', 'fallback',
    ]


def test_coalesce_can_be_used_with_null_checks(sqlite_engine):
    heroes = scan_table('heroes', sqlite_engine)

    out = (
        heroes
        .filter(
            coalesce(col('nickname'), col('first_name')).is_null(),
        )
        .select('id')
        .order_by('id')
        .collect()
    )

    assert out['id'].to_list() == [3, 8]


def test_coalesce_can_be_used_with_is_not_null(sqlite_engine):
    heroes = scan_table('heroes', sqlite_engine)

    out = (
        heroes
        .filter(
            coalesce(col('nickname'), col('first_name')).is_not_null()
        )
        .select('id')
        .order_by('id')
        .collect()
    )

    assert out['id'].to_list() == [1, 2, 4, 5, 6, 7]


def test_coalesce_requires_at_least_one_argument():
    with pytest.raises(ValueError, match='Coalesce requires at least one argument'):
        coalesce()
