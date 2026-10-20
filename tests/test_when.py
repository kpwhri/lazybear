import pytest
import sqlalchemy as sa

import lazybear as lb


def test_when_then_otherwise_select(sqlite_engine):
    users = lb.scan_table('users', sqlite_engine)

    out = (
        users
        .select(
            'id',
            lb.when(lb.col('age') < 30)
            .then('young')
            .otherwise('older')
            .alias('age_group'),
        )
        .order_by('id')
        .collect()
    )

    assert out['id'].to_list() == [1, 2, 3, 4]
    assert out['age_group'].to_list() == ['older', 'young', 'older', 'older']


def test_when_then_without_otherwise_returns_null(sqlite_engine):
    users = lb.scan_table('users', sqlite_engine)

    out = (
        users
        .select('id', lb.when(lb.col('age') < 30).then('young').alias('age_group'))
        .order_by('id')
        .collect()
    )

    assert out['age_group'].to_list() == [None, 'young', None, None]


def test_when_then_chained_conditions(sqlite_engine):
    users = lb.scan_table('users', sqlite_engine)

    out = (
        users
        .select(
            'id',
            lb.when(lb.col('age') < 30)
            .then('under_30')
            .when(lb.col('age') < 40)
            .then('thirties')
            .otherwise('40_plus')
            .alias('age_group'),
        )
        .order_by('id')
        .collect()
    )

    assert out['age_group'].to_list() == ['thirties', 'under_30', '40_plus', '40_plus']


def test_when_then_first_matching_condition_wins(sqlite_engine):
    users = lb.scan_table('users', sqlite_engine)

    out = (
        users
        .select(
            'id',
            lb.when(lb.col('age') >= 30)
            .then('first')
            .when(lb.col('age') >= 40)
            .then('second')
            .otherwise('neither')
            .alias('result'),
        )
        .order_by('id')
        .collect()
    )

    assert out['result'].to_list() == ['first', 'neither', 'first', 'first']


def test_when_multiple_predicates_are_combined_with_and(sqlite_engine):
    users = lb.scan_table('users', sqlite_engine)

    out = (
        users
        .select(
            'id',
            lb.when(
                lb.col('age') >= 40,
                lb.col('name') == 'Tellervo',
            )
            .then(1)
            .otherwise(0)
            .alias('matches'),
        )
        .order_by('id')
        .collect()
    )

    assert out['matches'].to_list() == [0, 0, 1, 0]


def test_when_keyword_constraints(sqlite_engine):
    users = lb.scan_table('users', sqlite_engine)

    out = (
        users
        .select(
            'id',
            lb.when(age=41, name='Ukko')
            .then('match')
            .otherwise('other')
            .alias('result'),
        )
        .order_by('id')
        .collect()
    )

    assert out['result'].to_list() == ['other', 'other', 'other', 'match']


def test_when_mixed_predicates_and_keyword_constraints(sqlite_engine):
    users = lb.scan_table('users', sqlite_engine)

    out = (
        users
        .select(
            'id',
            lb.when(lb.col('age') >= 40, name='Tellervo')
            .then(True)
            .otherwise(False)
            .alias('matches'),
        )
        .order_by('id')
        .collect()
    )

    assert out['matches'].to_list() == [False, False, True, False]


def test_when_keyword_constraint_matches_null(sqlite_engine):
    heroes = lb.scan_table('heroes', sqlite_engine)

    out = (
        heroes
        .select(
            'id',
            lb.when(score=None)
            .then('missing')
            .otherwise('present')
            .alias('score_status'),
        )
        .order_by('id')
        .collect()
    )

    assert out['score_status'].to_list() == [
        'present', 'missing', 'missing', 'present',
        'present', 'missing', 'missing', 'present',
    ]


def test_when_null_predicate_uses_otherwise(sqlite_engine):
    heroes = lb.scan_table('heroes', sqlite_engine)

    out = (
        heroes
        .select(
            'id',
            lb.when(lb.col('score') >= 90)
            .then('high')
            .otherwise('not_high')
            .alias('score_group'),
        )
        .order_by('id')
        .collect()
    )

    assert out['score_group'].to_list() == [
        'high', 'not_high', 'not_high', 'high',
        'not_high', 'not_high', 'not_high', 'not_high',
    ]


def test_when_then_column_expressions(sqlite_engine):
    users = lb.scan_table('users', sqlite_engine)

    out = (
        users
        .select(
            'id',
            lb.when(lb.col('age') < 30)
            .then(lb.col('age') * 2)
            .otherwise(lb.col('age'))
            .alias('adjusted_age'),
        )
        .order_by('id')
        .collect()
    )

    assert out['adjusted_age'].to_list() == [30, 56, 41, 41]


def test_when_then_literal_and_sqlalchemy_expressions(sqlite_engine):
    users = lb.scan_table('users', sqlite_engine)

    out = (
        users
        .select(
            'id',
            lb.when(sa.true())
            .then(lb.lit('literal'))
            .otherwise(sa.literal('fallback'))
            .alias('matched'),
            lb.when(sa.false())
            .then(lb.lit('unused'))
            .otherwise(sa.literal('fallback'))
            .alias('unmatched'),
        )
        .order_by('id')
        .collect()
    )

    assert out['matched'].to_list() == ['literal'] * 4
    assert out['unmatched'].to_list() == ['fallback'] * 4


def test_when_strings_in_then_and_otherwise_are_literals(sqlite_engine):
    users = lb.scan_table('users', sqlite_engine)

    out = (
        users
        .select(
            'id',
            lb.when(lb.col('age') < 30)
            .then('name')
            .otherwise('age')
            .alias('result'),
        )
        .order_by('id')
        .collect()
    )

    assert out['result'].to_list() == ['age', 'name', 'age', 'age']


def test_when_then_with_columns_alias(sqlite_engine):
    users = lb.scan_table('users', sqlite_engine)

    updated = users.with_columns(
        lb.when(lb.col('age') >= 40)
        .then('older')
        .otherwise('younger')
        .alias('age_group')
    )
    out = updated.select('id', 'age_group').order_by('id').collect()

    assert 'age_group' not in users.columns
    assert 'age_group' in updated.columns
    assert out['age_group'].to_list() == ['younger', 'younger', 'older', 'older']


def test_when_then_with_columns_named_expression(sqlite_engine):
    users = lb.scan_table('users', sqlite_engine)

    out = (
        users
        .with_columns(
            is_older=lb.when(lb.col('age') >= 40).then(1).otherwise(0),
        )
        .select('id', 'is_older')
        .order_by('id')
        .collect()
    )

    assert out['is_older'].to_list() == [0, 0, 1, 1]


def test_when_then_nested_expression(sqlite_engine):
    heroes = lb.scan_table('heroes', sqlite_engine)

    out = (
        heroes
        .select(
            'id',
            lb.when(lb.col('score').is_null())
            .then(
                lb.when(lb.col('bonus').is_null())
                .then(-1)
                .otherwise(lb.col('bonus'))
            )
            .otherwise(lb.col('score'))
            .alias('effective_score'),
        )
        .order_by('id')
        .collect()
    )

    assert out['effective_score'].to_list() == [100, 50, -1, 90, 70, 95, -1, 60]


def test_when_then_chaining_does_not_change_previous_expression(sqlite_engine):
    users = lb.scan_table('users', sqlite_engine)
    base = lb.when(lb.col('age') < 30).then('young')
    extended = base.when(lb.col('age') < 40).then('adult')

    out = (
        users
        .select(
            'id',
            base.alias('base'),
            extended.alias('extended'),
        )
        .order_by('id')
        .collect()
    )

    assert out['base'].to_list() == [None, 'young', None, None]
    assert out['extended'].to_list() == ['adult', 'young', None, None]


def test_when_compiles_to_sql_case(sqlite_engine):
    users = lb.scan_table('users', sqlite_engine)

    query = users.select(
        lb.when(lb.col('age') < 30)
        .then('young')
        .otherwise('older')
        .alias('age_group')
    )

    assert 'CASE' in query.explain().upper()
    assert 'WHEN' in query.explain().upper()


def test_when_requires_at_least_one_predicate():
    with pytest.raises(ValueError, match='predicate'):
        lb.when()

    base = lb.when(lb.col('age') < 30).then('young')
    with pytest.raises(ValueError, match='predicate'):
        base.when()


def test_when_otherwise_none_returns_null(sqlite_engine):
    users = lb.scan_table('users', sqlite_engine)

    out = (
        users
        .select(
            'id',
            lb.when(lb.col('age') < 30)
            .then('young')
            .otherwise(None)
            .alias('age_group'),
        )
        .order_by('id')
        .collect()
    )

    assert out['age_group'].to_list() == [None, 'young', None, None]


def test_when_chained_mixed_predicates_and_keyword_constraints(sqlite_engine):
    users = lb.scan_table('users', sqlite_engine)

    out = (
        users
        .select(
            'id',
            lb.when(lb.col('age') < 30)
            .then('young')
            .when(lb.col('age') >= 40, name='Tellervo')
            .then('tellervo_40_plus')
            .otherwise('other')
            .alias('result'),
        )
        .order_by('id')
        .collect()
    )

    assert out['result'].to_list() == ['other', 'young', 'tellervo_40_plus', 'other']


def test_when_then_none_returns_null(sqlite_engine):
    users = lb.scan_table('users', sqlite_engine)

    out = (
        users
        .select(
            'id',
            lb.when(
                lb.col('age') < 30
            )
            .then(None)
            .otherwise('not_null')
            .alias('result'),
        )
        .order_by('id')
        .collect()
    )

    assert out['result'].to_list() == ['not_null', None, 'not_null', 'not_null']


def test_when_then_boolean_expression_values(sqlite_engine):
    users = lb.scan_table('users', sqlite_engine)

    out = (
        users
        .select(
            'id',
            lb.when(lb.col('age') < 30)
            .then(lb.col('age') > 20)
            .otherwise(lb.col('age') > 40)
            .alias('result'),
        )
        .order_by('id')
        .collect()
    )

    assert out['result'].to_list() == [False, True, True, True]


def test_when_computed_predicate(sqlite_engine):
    users = lb.scan_table('users', sqlite_engine)

    out = (
        users
        .select(
            'id',
            lb.when((lb.col('age') + 1) >= 42)
            .then('match')
            .otherwise('other')
            .alias('result'),
        )
        .order_by('id')
        .collect()
    )

    assert out['result'].to_list() == ['other', 'other', 'match', 'match']


def test_when_is_not_null_predicate(sqlite_engine):
    heroes = lb.scan_table('heroes', sqlite_engine)

    out = (
        heroes
        .select(
            'id',
            lb.when(lb.col('score').is_not_null())
            .then('present')
            .otherwise('missing')
            .alias('score_status'),
        )
        .order_by('id')
        .collect()
    )

    assert out['score_status'].to_list() == [
        'present', 'missing', 'missing', 'present',
        'present', 'missing', 'missing', 'present',
    ]


def test_when_is_in_predicate(sqlite_engine):
    users = lb.scan_table('users', sqlite_engine)

    out = (
        users
        .select(
            'id',
            lb.when(lb.col('name').is_in(['Tellervo', 'Ukko']))
            .then('match')
            .otherwise('other')
            .alias('result'),
        )
        .order_by('id')
        .collect()
    )

    assert out['result'].to_list() == ['other', 'other', 'match', 'match']


def test_when_chained_keyword_constraints(sqlite_engine):
    users = lb.scan_table('users', sqlite_engine)

    out = (
        users
        .select(
            'id',
            lb.when(age=28)
            .then('young')
            .when(age=41)
            .then('forty_one')
            .otherwise('other')
            .alias('result'),
        )
        .order_by('id')
        .collect()
    )

    assert out['result'].to_list() == ['other', 'young', 'forty_one', 'forty_one']
