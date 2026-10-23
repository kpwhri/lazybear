from datetime import date

import pytest
import sqlalchemy as sa
from sqlalchemy.dialects import mssql, mysql, oracle, postgresql, sqlite

from lazybear import calc_age, col, lit, scan_table


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
