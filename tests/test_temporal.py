from datetime import date

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import mssql, mysql, oracle, postgresql, sqlite

from lazybear import (
    between, calc_age, col, date_add, date_diff, date_part, date_trunc,
    in_interval, in_window, intervals_overlap, lit, scan_table,
)


@pytest.fixture()
def age_df():
    eng = sa.create_engine('sqlite:///:memory:')
    meta = sa.MetaData()
    people = sa.Table(
        'people', meta,
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('birth_date', sa.Date),
        sa.Column('index_date', sa.Date),
    )
    meta.create_all(eng)
    with eng.begin() as conn:
        conn.execute(people.insert(), [
            {'id': 1, 'birth_date': date(2000, 10, 20), 'index_date': date(2026, 10, 21)},
            {'id': 2, 'birth_date': date(2000, 10, 21), 'index_date': date(2026, 10, 21)},
            {'id': 3, 'birth_date': date(2000, 10, 22), 'index_date': date(2026, 10, 21)},
            {'id': 4, 'birth_date': date(2000, 2, 29), 'index_date': date(2025, 2, 28)},
            {'id': 5, 'birth_date': date(2000, 2, 29), 'index_date': date(2025, 3, 1)},
            {'id': 6, 'birth_date': None, 'index_date': date(2026, 10, 21)},
            {'id': 7, 'birth_date': date(2000, 1, 1), 'index_date': None},
            {'id': 8, 'birth_date': date(2026, 10, 21), 'index_date': date(2026, 10, 21)},
            {'id': 9, 'birth_date': date(1980, 12, 31), 'index_date': date(2026, 10, 21)},
            {'id': 10, 'birth_date': date(2000, 2, 29), 'index_date': date(2024, 2, 28)},
            {'id': 11, 'birth_date': date(2000, 2, 29), 'index_date': date(2024, 2, 29)},
            {'id': 12, 'birth_date': date(2026, 11, 1), 'index_date': date(2026, 10, 21)},
        ])
    yield scan_table('people', eng)
    eng.dispose()


@pytest.fixture()
def temporal_df():
    eng = sa.create_engine('sqlite:///:memory:')
    meta = sa.MetaData()
    events = sa.Table(
        'events', meta,
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('event_date', sa.Date),
        sa.Column('index_date', sa.Date),
    )
    meta.create_all(eng)
    with eng.begin() as conn:
        conn.execute(events.insert(), [
            {'id': 1, 'event_date': date(2025, 10, 3), 'index_date': date(2026, 1, 1)},
            {'id': 2, 'event_date': date(2025, 10, 2), 'index_date': date(2026, 1, 1)},
            {'id': 3, 'event_date': date(2026, 1, 1), 'index_date': date(2026, 1, 1)},
            {'id': 4, 'event_date': date(2026, 4, 1), 'index_date': date(2026, 1, 1)},
            {'id': 5, 'event_date': date(2026, 4, 2), 'index_date': date(2026, 1, 1)},
            {'id': 6, 'event_date': None, 'index_date': date(2026, 1, 1)},
            {'id': 7, 'event_date': date(2026, 7, 12), 'index_date': date(2026, 1, 1)},
            {'id': 8, 'event_date': date(2026, 4, 1), 'index_date': None},
        ])
    yield scan_table('events', eng)
    eng.dispose()


def test_calculate_age_with_reference_column(age_df):
    result = (
        age_df
        .with_columns(
            age=calc_age(col('birth_date'), col('index_date'))
        )
        .select('id', 'age')
        .order_by('id')
        .collect()
    )
    assert result['age'].to_list() == [26, 26, 25, 24, 25, None, None, 0, 45, 23, 24, -1]


def test_calculate_age_with_reference_column_alias(age_df):
    result = (
        age_df
        .with_columns(
            calc_age(col('birth_date'), col('index_date')).alias('age')
        )
        .select('id', 'age')
        .order_by('id')
        .collect()
    )
    assert result['age'].to_list() == [26, 26, 25, 24, 25, None, None, 0, 45, 23, 24, -1]


@pytest.mark.parametrize('reference', [
    date(2026, 10, 21),
    lit(date(2026, 10, 21)),
])
def test_calculate_age_with_static_reference(age_df, reference):
    result = (
        age_df
        .select(
            'id',
            calc_age(col('birth_date'), reference).alias('age')
        )
        .order_by('id')
        .collect()
    )
    assert result['age'].to_list() == [26, 26, 25, 26, 26, None, 26, 0, 45, 26, 26, -1]


def test_calculate_age_in_filter(age_df):
    result = (
        age_df
        .filter(
            calc_age(col('birth_date'), col('index_date')) >= 26,
        )
        .select('id')
        .order_by('id')
        .collect()
    )
    assert result['id'].to_list() == [1, 2, 9]


@pytest.mark.parametrize(('dialect', 'date_parts'), [
    (sqlite.dialect(), ("STRFTIME('%Y'", "STRFTIME('%m'", "STRFTIME('%d'")),
    (postgresql.dialect(), ('EXTRACT(year FROM', 'EXTRACT(month FROM', 'EXTRACT(day FROM')),
    (mssql.dialect(), ('DATEPART(year,', 'DATEPART(month,', 'DATEPART(day,')),
    (oracle.dialect(), ('EXTRACT(year FROM', 'EXTRACT(month FROM', 'EXTRACT(day FROM')),
    (mysql.dialect(), ('EXTRACT(year FROM', 'EXTRACT(month FROM', 'EXTRACT(day FROM')),
], ids=['sqlite', 'postgresql', 'mssql', 'oracle', 'mysql'])
@pytest.mark.parametrize('reference_kind', ['literal', 'column'])
def test_calculate_age_compiles_for_dialects(dialect, date_parts, reference_kind):
    """Check emitted SQL and bind behavior without connecting to other DBs."""
    people = sa.table(
        'people',
        sa.column('birth_date', sa.Date),
        sa.column('index_date', sa.Date),
    )
    reference = (
        date(2026, 10, 21)
        if reference_kind == 'literal'
        else people.c.index_date
    )
    # sqlalchemy columns do not need a LazyBearFrame to resolve.
    age_expr = calc_age(people.c.birth_date, reference).to_sa(None)
    compiled = sa.select(age_expr.label('age')).compile(dialect=dialect)
    sql = str(compiled)

    assert isinstance(age_expr.type, sa.Integer)
    assert 'people.birth_date' in sql
    assert ' - CASE WHEN' in sql
    assert ' ELSE ' in sql
    assert all(part in sql for part in date_parts)
    assert 1 in compiled.params.values()
    assert 0 in compiled.params.values()

    bound_dates = [value for value in compiled.params.values() if isinstance(value, date)]
    if reference_kind == 'literal':
        assert bound_dates == [date(2026, 10, 21)]
        assert '2026-10-21' not in sql  # bound, not interpolated
    else:
        assert 'people.index_date' in sql
        assert bound_dates == []


def test_date_add_and_date_diff_sqlite(temporal_df):
    result = (
        temporal_df
        .with_columns(
            plus=date_add(col('index_date'), years=1, months=2, days=3),
            minus=date_add(col('index_date'), days=-90),
            delta=date_diff(col('index_date'), col('event_date')),
        )
        .select('id', 'plus', 'minus', 'delta')
        .order_by('id')
        .collect()
    )
    assert result['plus'].to_list() == [date(2027, 3, 4)] * 7 + [None]
    assert result['minus'].to_list() == [date(2025, 10, 3)] * 7 + [None]
    assert result['delta'].to_list() == [-90, -91, 0, 90, 91, None, 192, None]


def test_date_trunc_and_part_sqlite(temporal_df):
    result = (
        temporal_df
        .select(
            'id',
            date_trunc(col('event_date'), 'quarter').alias('quarter_start'),
            date_trunc(col('event_date'), 'month').alias('month_start'),
            date_part(col('event_date'), 'quarter').alias('quarter'),
            date_part(col('event_date'), 'weekday').alias('weekday'),
        )
        .order_by('id')
        .collect()
    )
    assert result['quarter_start'].to_list() == [
        date(2025, 10, 1), date(2025, 10, 1),
        date(2026, 1, 1), date(2026, 4, 1), date(2026, 4, 1),
        None, date(2026, 7, 1), date(2026, 4, 1),
    ]
    assert result['month_start'].to_list() == [
        date(2025, 10, 1), date(2025, 10, 1),
        date(2026, 1, 1), date(2026, 4, 1), date(2026, 4, 1),
        None, date(2026, 7, 1), date(2026, 4, 1),
    ]
    assert result['quarter'].to_list() == [4, 4, 1, 2, 2, None, 3, 2]
    assert result['weekday'].to_list() == [5, 4, 4, 3, 4, None, 7, 3]


@pytest.mark.parametrize(('days', 'expected'), [
    (-90, [1, 3]),
    (0, [3]),
    (90, [3, 4]),
])
def test_in_interval_signed_and_inclusive(temporal_df, days, expected):
    result = (
        temporal_df
        .filter(in_interval(col('event_date'), col('index_date'), ndays=days))
        .select('id')
        .order_by('id')
        .collect()
    )
    assert result['id'].to_list() == expected


def test_in_window_asymmetric_and_day_zero_exclusion(temporal_df):
    result = (
        temporal_df
        .filter(in_window(col('event_date'), col('index_date'), start_days=-90, end_days=90))
        .select('id')
        .order_by('id')
        .collect()
    )
    assert result['id'].to_list() == [1, 3, 4]
    result = (
        temporal_df
        .filter(in_window(col('event_date'), col('index_date'), start_days=1, end_days=90))
        .select('id')
        .order_by('id')
        .collect()
    )
    assert result['id'].to_list() == [4]


@pytest.mark.parametrize('predicate', [
    between(col('event_date'), date(2025, 10, 3), date(2026, 1, 1)),
    col('event_date').between(date(2025, 10, 3), date(2026, 1, 1)),
])
def test_between_function_and_expression_method(temporal_df, predicate):
    result = temporal_df.filter(predicate).select('id').order_by('id').collect()
    assert result['id'].to_list() == [1, 3]


def test_intervals_overlap_sqlite_including_empty_and_null():
    eng = sa.create_engine('sqlite:///:memory:')
    meta = sa.MetaData()
    spans = sa.Table(
        'spans', meta,
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('a_start', sa.Date), sa.Column('a_end', sa.Date),
        sa.Column('b_start', sa.Date), sa.Column('b_end', sa.Date),
    )
    meta.create_all(eng)
    d = lambda day: date(2026, 1, day)
    with eng.begin() as conn:
        conn.execute(spans.insert(), [
            {'id': 1, 'a_start': d(1), 'a_end': d(4), 'b_start': d(3), 'b_end': d(7)},
            {'id': 2, 'a_start': d(1), 'a_end': d(4), 'b_start': d(4), 'b_end': d(7)},
            {'id': 3, 'a_start': d(1), 'a_end': d(1), 'b_start': d(1), 'b_end': d(7)},
            {'id': 4, 'a_start': d(1), 'a_end': d(4), 'b_start': d(1), 'b_end': d(4)},
            {'id': 5, 'a_start': d(1), 'a_end': None, 'b_start': d(1), 'b_end': d(4)},
        ])
    result = (
        scan_table('spans', eng)
        .select(
            'id',
            intervals_overlap(
                col('a_start'), col('a_end'), col('b_start'), col('b_end'),
            ).alias('overlap'),
        )
        .order_by('id')
        .collect()
    )
    assert result['overlap'].to_list() == [True, False, False, True, None]
    eng.dispose()


@pytest.mark.parametrize(('dialect', 'expected'), [
    (sqlite.dialect(), ('DATE(', 'JULIANDAY(', 'STRFTIME(', 'PRINTF(')),
    (postgresql.dialect(), ('INTERVAL', 'DATE_TRUNC(', 'EXTRACT(ISODOW', 'CAST(')),
    (mssql.dialect(), ('DATEADD(', 'DATEDIFF(', 'DATEFROMPARTS(', 'DATEPART(')),
    (oracle.dialect(), ('ADD_MONTHS(', 'TRUNC(', "'IW'", 'EXTRACT(')),
    (mysql.dialect(), ('DATE_ADD(', 'DATEDIFF(', 'MAKEDATE(', 'WEEKDAY(')),
], ids=['sqlite', 'postgresql', 'mssql', 'oracle', 'mysql'])
def test_temporal_functions_compile_for_dialects(dialect, expected):
    people = sa.table(
        'people',
        sa.column('event_date', sa.Date),
        sa.column('index_date', sa.Date),
    )
    expressions = [
        date_add(people.c.index_date, days=-90, months=1),
        date_diff(people.c.index_date, people.c.event_date),
        date_trunc(people.c.event_date, 'quarter'),
        date_part(people.c.event_date, 'weekday'),
        date_part(people.c.event_date, 'quarter'),
        between(people.c.event_date, date(2025, 1, 1), date(2026, 1, 1)),
        in_interval(people.c.event_date, people.c.index_date, ndays=-90),
        in_window(people.c.event_date, people.c.index_date, start_days=-90, end_days=30),
        intervals_overlap(people.c.event_date, people.c.index_date,
                          people.c.event_date, people.c.index_date),
    ]
    query = sa.select(*[expr.to_sa(None) for expr in expressions])
    compiled = query.compile(dialect=dialect)
    sql = str(compiled)
    assert all(fragment in sql for fragment in expected)
    assert ' BETWEEN ' in sql
    assert ' AND ' in sql
    assert 'people.event_date' in sql
    assert -90 in compiled.params.values()
    assert date(2025, 1, 1) in compiled.params.values()
    assert '2025-01-01' not in sql  # Python dates remain bound parameters


@pytest.mark.parametrize(('func', 'args', 'kwargs', 'error'), [
    (date_trunc, (date(2026, 1, 1), 'week'), {}, ValueError),
    (date_part, (date(2026, 1, 1), 'dow'), {}, ValueError),
    (date_add, (date(2026, 1, 1),), {'days': 1.5}, TypeError),
    (date_add, (date(2026, 1, 1),), {'days': True}, TypeError),
    (in_interval, (date(2026, 1, 1), date(2026, 1, 1)), {'ndays': 1.5}, TypeError),
    (in_window, (date(2026, 1, 1), date(2026, 1, 1)),
     {'start_days': 90, 'end_days': 1}, ValueError),
])
def test_temporal_invalid_arguments(func, args, kwargs, error):
    with pytest.raises(error):
        func(*args, **kwargs)
