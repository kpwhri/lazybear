import pytest
import sqlalchemy as sa

import polars as pl

from lazybear import scan_table, col, scan_sql_query


@pytest.fixture()
def sqlite_engine():
    eng = sa.create_engine('sqlite:///:memory:')
    meta = sa.MetaData()

    t_users = sa.Table(
        'users', meta,
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('name', sa.String),
        sa.Column('age', sa.Integer),
    )

    t_orders = sa.Table(
        'orders', meta,
        sa.Column('id', sa.Integer, primary_key=True),
        sa.Column('user_id', sa.Integer),
        sa.Column('amount', sa.Float),
    )

    meta.create_all(eng)

    with eng.begin() as conn:
        conn.execute(t_users.insert(), [
            {'id': 1, 'name': 'Ahti', 'age': 30},
            {'id': 2, 'name': 'Kalma', 'age': 28},
            {'id': 3, 'name': 'Tellervo', 'age': 41},
            {'id': 4, 'name': 'Ukko', 'age': 41},
        ])
        conn.execute(t_orders.insert(), [
            {'id': 10, 'user_id': 1, 'amount': 12.5},
            {'id': 11, 'user_id': 1, 'amount': 7.5},
            {'id': 12, 'user_id': 2, 'amount': 99.0},
        ])

    yield eng


@pytest.fixture()
def sqlite_engine_mixcase():
    eng = sa.create_engine('sqlite:///:memory:')
    meta = sa.MetaData()
    t = sa.Table(
        'MixedCase', meta,
        sa.Column('ID', sa.Integer, primary_key=True),
        sa.Column('UserName', sa.String),
        sa.Column('AGE', sa.Integer),
    )
    meta.create_all(eng)
    with eng.begin() as conn:
        conn.execute(t.insert(), [{'ID': 1, 'UserName': 'Ahti', 'AGE': 30}])
    return eng


def test_select_filter_collect(sqlite_engine):
    lf = scan_table('users', sqlite_engine)
    out = lf.filter(col('age') > 30).select('id', 'age').collect()
    assert isinstance(out, pl.DataFrame)
    assert set(out.columns) == {'id', 'age'}
    assert out.shape[0] == 2  # Tellervo, Ukko


def test_filter_join_select_collect(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)
    out = (
        users.filter((col('age') > 30) & (col('age') < 60))
        .join(orders, on={'id': 'user_id'}, how='left')
        .select('id', 'age')
        .collect()
    )

    assert isinstance(out, pl.DataFrame)
    assert set(out.columns) == {'id', 'age'}
    assert out.shape[0] == 2  # Tellervo, Ukko


def test_with_columns(sqlite_engine):
    lf = scan_table('users', sqlite_engine)
    out = lf.with_columns(('age2', col('age') * 2), decade=col('age') / 10 * 10).select('id', 'age2',
                                                                                        'decade').collect()
    assert set(out.columns) == {'id', 'age2', 'decade'}
    assert out.filter(pl.col('id') == 1)['age2'][0] == 60


def test_order_by_and_limit(sqlite_engine):
    lf = scan_table('users', sqlite_engine)
    # order desc by age then asc by name, and limit 2
    out = lf.order_by('-age', 'name').limit(2).collect()
    assert out.shape[0] == 2
    # top ages are 41 for Tellervo, Ukko -> sorted by name yields Tellervo first
    assert out['name'][0] in {'Tellervo', 'Ukko'}


def test_groupby_agg(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    out = users.group_by('age').agg(n=('id', 'count'), min_id=('id', 'min')).collect()
    assert 'n' in out.columns and 'min_id' in out.columns and 'age' in out.columns
    # ages present: 30, 28, 41
    assert set(out['age'].to_list()) == {28, 30, 41}


def test_join(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)
    j = users.join(orders, on={'id': 'user_id'}, how='left')
    out = j.select('id', 'name', 'amount').order_by('id', 'amount').collect()
    # user 3 and 4 have no orders, ensure left join kept them (amount null)
    ids = out['id'].to_list()
    assert 3 in ids and 4 in ids


def test_join_left_on_right_on_single(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)
    j = users.join(orders, left_on='id', right_on='user_id', how='left')
    out = j.select('id', 'name', 'amount').order_by('id', 'amount').collect()
    ids = out['id'].to_list()
    assert 3 in ids and 4 in ids


def test_scan_table_lowercase_default(sqlite_engine_mixcase):
    lf = scan_table('MixedCase', sqlite_engine_mixcase)
    # default should lowercase the exposed column keys
    assert set(lf.columns) == {'id', 'username', 'age'}
    # Ensure collect works and returns lowercased columns
    df = lf.collect()
    assert set(df.columns) == {'id', 'username', 'age'}
    # Ensure lf.filter and lf.with_columns work with lowercased keys
    df2 = lf.filter(
        col('id') == 1
    ).with_columns(
        ('age_plus', col('age') + 1)
    ).select('id', 'username', 'age_plus').collect()
    assert df2.shape[0] == 1
    assert set(df2.columns) == {'id', 'username', 'age_plus'}
    assert df2['age_plus'][0] == 31


def test_scan_table_no_lowercase(sqlite_engine_mixcase):
    lf = scan_table('MixedCase', sqlite_engine_mixcase, lowercase=False)
    # when disabled, preserve original casing from reflection
    assert set(lf.columns) == {'ID', 'UserName', 'AGE'}
    df = lf.collect()
    assert set(df.columns) == {'ID', 'UserName', 'AGE'}
    # ensure lf.filter and lf.with_columns work with original-case keys
    df2 = lf.filter(
        col('ID') == 1
    ).with_columns(
        ('AGE_PLUS', col('AGE') + 1)
    ).select('ID', 'UserName', 'AGE_PLUS').collect()
    assert df2.shape[0] == 1
    assert set(df2.columns) == {'ID', 'UserName', 'AGE_PLUS'}
    assert df2['AGE_PLUS'][0] == 31


def test_str_contains_literal(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    # contains literal substring 'a' -> Ahti, Kalma
    out = users.filter(
        col('name').str.contains('a', literal=True)
    ).select('name').order_by('name').collect()
    assert out['name'].to_list() == ['Ahti', 'Kalma']


def test_str_contains_any_literal(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    # any of ['Ah','Ka'] -> Ahti, Kalma
    out = users.filter(
        col('name').str.contains_any(['Ah', 'Ka'], literal=True)
    ).select('name').order_by('name').collect()
    assert out['name'].to_list() == ['Ahti', 'Kalma']


def test_str_starts_with(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    out = users.filter(
        col('name').str.starts_with('a')
    ).select('name').collect()
    assert set(out['name'].to_list()) == {'Ahti'}


def test_str_startswith_any(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    out = users.filter(
        col('name').str.starts_with_any(['a', 'k'])
    ).select('name').order_by('name').collect()
    assert out['name'].to_list() == ['Ahti', 'Kalma']


def test_join_left_on_right_on_list_mismatch_raises(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)
    with pytest.raises(ValueError):
        users.join(orders, left_on=['id', 'age'], right_on=['user_id'], how='inner').collect()


def test_join_on_and_left_right_on_mutually_exclusive(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    orders = scan_table('orders', sqlite_engine)
    with pytest.raises(ValueError):
        users.join(orders, on='id', left_on='id', right_on='user_id').collect()


def test_collect_iter(sqlite_engine):
    users = scan_table('users', sqlite_engine).order_by('id')
    chunks = list(users.collect_batches(chunk_size=2))
    assert len(chunks) == 2
    assert sum(ch.shape[0] for ch in chunks) == 4


def test_to_arrow_optional(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    try:
        tab = users.to_arrow(limit=2)
    except RuntimeError as e:
        pytest.skip('pyarrow not installed: install with `pip install pyarrow`')
    assert tab.num_rows == 2


def test_sort_polars_style(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    # Single key ascending
    out1 = users.sort(by='age').collect()
    assert out1['age'].to_list() == sorted(out1['age'].to_list())

    # Single key descending
    out2 = users.sort(by='age', descending=True).collect()
    assert out2['age'].to_list() == sorted(out2['age'].to_list(), reverse=True)

    # Multiple keys using by + more_by, descending broadcast
    out3 = users.sort(by='age', more_by='name', descending=[True, False]).collect()
    ages = out3['age'].to_list()
    assert ages == sorted(ages, reverse=True)


def test_sort_desc_flags_validation(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    with pytest.raises(ValueError):
        users.sort(by=['age', 'name'], descending=[True]).collect()


def test_sql_query(sqlite_engine):
    df = scan_sql_query('select name, age from users where age > 30', sqlite_engine).filter(col('age') < 60).select(
        'name').collect()
    assert len(df['name'].to_list()) == 2


def test_write_csv_chunked_single_file(sqlite_engine, tmp_path):
    users = scan_table('users', sqlite_engine).order_by('id')
    target = tmp_path / 'users.csv'
    users.write_csv(target, chunk_size=2, include_header=True)
    df = pl.read_csv(target)
    # Expect same number of rows and ids 1..4
    assert df.shape[0] == 4
    assert set(df['id'].to_list()) == {1, 2, 3, 4}


def test_write_parquet_chunked_prefix(sqlite_engine, tmp_path):
    users = scan_table('users', sqlite_engine).order_by('id')
    base = tmp_path / 'users.parquet'
    users.write_parquet(base, chunk_size=2)
    # Expect part files like users-00000.parquet and users-00001.parquet
    parts = sorted(tmp_path.glob('users-*.parquet'))
    assert len(parts) >= 2
    df = pl.read_parquet(str(tmp_path / 'users-*.parquet'))
    assert df.shape[0] == 4
    assert set(df['id'].to_list()) == {1, 2, 3, 4}


def test_iter_rows_default_and_named(sqlite_engine):
    users = scan_table('users', sqlite_engine).order_by('id')
    # Default (tuples)
    rows = list(users.iter_rows())
    assert rows[0] == (1, 'Ahti', 30)
    assert rows[-1] == (4, 'Ukko', 41)
    # Named dicts
    rows_named = list(users.iter_rows(named=True))
    assert rows_named[0] == {'id': 1, 'name': 'Ahti', 'age': 30}
    assert rows_named[-1] == {'id': 4, 'name': 'Ukko', 'age': 41}


def test_iter_rows_chunked(sqlite_engine):
    users = scan_table('users', sqlite_engine).order_by('id')
    it = users.iter_rows(chunk_size=2)
    first_two = [next(it), next(it)]
    assert first_two == [(1, 'Ahti', 30), (2, 'Kalma', 28)]
    remaining = list(it)
    assert remaining[-1] == (4, 'Ukko', 41)


def test_is_in_with_literal_list(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    out = (
        users.filter(col('age').is_in([28, 41]))
        .select('id', 'age')
        .order_by('id')
        .collect()
    )
    assert set(out['age'].to_list()) == {28, 41}
    assert out['id'].to_list() == [2, 3, 4]


def test_is_in_with_empty_iterable(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    out = users.filter(col('age').is_in([])).collect()
    assert out.shape[0] == 0


def test_is_in_with_polars_series(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    vals = pl.Series([28, 30])
    out = (
        users.filter(col('age').is_in(vals))
        .select('id', 'age')
        .order_by('id')
        .collect()
    )
    # ages present should be 28 and 30, ids 1 (30) and 2 (28)
    assert set(out['age'].to_list()) == {28, 30}
    assert out['id'].to_list() == [1, 2]


def test_is_in_treats_string_as_scalar(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    out = (
        users.filter(col('name').is_in('Ahti'))
        .select('id', 'name')
        .order_by('id')
        .collect()
    )
    assert out.shape[0] == 1
    assert out['id'].to_list() == [1]
    assert out['name'].to_list() == ['Ahti']


def test_is_in_with_lazyframe_subquery(sqlite_engine):
    users = scan_table('users', sqlite_engine)
    young_ids = scan_table('users', sqlite_engine).filter(col('age') < 30).select('id')
    out = (
        users.filter(col('id').is_in(young_ids))
        .select('id', 'name')
        .order_by('id')
        .collect()
    )
    assert out['id'].to_list() == [2]
    assert out['name'].to_list() == ['Kalma']


def test_limit_applies_in_order(sqlite_engine):
    age41 = scan_table('users', sqlite_engine).order_by('age').limit(1).filter(col('age') == 41)
    result_df = age41.collect()
    assert result_df.height == 0
