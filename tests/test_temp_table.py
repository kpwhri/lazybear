import polars as pl
import pytest
import sqlalchemy as sa
from lazybear import col, scan_df, scan_table


@pytest.fixture
def hero_df():
    return pl.DataFrame({
        'name': ['Väinämöinen', 'Joukahainen', 'Ilmarinen'],
        'power': [100, 50, 80]
    })


def test_temp_table_sqlite(sqlite_engine, hero_df):
    lf = scan_df(hero_df, sqlite_engine, table_name='heroes')

    # check if we can select and filter
    out_df = lf.filter(col('power') > 60).select('name').collect()
    exp_df = hero_df.filter(pl.col('power') > 60).select('name')

    assert out_df.shape == exp_df.shape
    assert sorted(out_df['name'].to_list()) == sorted(exp_df['name'].to_list())


def test_temp_table_join_with_regular_table(sqlite_engine, hero_df):
    with sqlite_engine.connect() as conn:
        conn.execute(sa.text('CREATE TABLE myth_info (name TEXT, origin TEXT)'))
        conn.execute(sa.text("INSERT INTO myth_info VALUES ('Väinämöinen', 'Kalevala')"))
        conn.execute(sa.text("INSERT INTO myth_info VALUES ('Joukahainen', 'Kalevala')"))
        conn.execute(sa.text("INSERT INTO myth_info VALUES ('Ilmarinen', 'Kalevala')"))
        conn.commit()

    lf_temp = scan_df(hero_df, sqlite_engine, table_name='temp_heroes')
    lf_table = scan_table('myth_info', sqlite_engine)

    joined_df = lf_temp.join(lf_table, on='name').collect()

    assert joined_df.shape == (3, 4)
    assert 'origin' in joined_df.columns
    assert 'power' in joined_df.columns
    assert all(joined_df['origin'] == 'Kalevala')


def test_temp_table_cleanup(sqlite_engine, hero_df):
    lf = scan_df(hero_df, sqlite_engine, table_name='cleanup_test')
    lf.collect()

    # after collect, the table should be dropped
    with sqlite_engine.connect() as conn:
        res = conn.execute(sa.text("SELECT name FROM sqlite_master WHERE type='table' AND name='cleanup_test'"))
        assert res.fetchone() is None


def test_temp_table_multiple_collects(sqlite_engine, hero_df):
    # each collect should recreate and then cleanup
    lf = scan_df(hero_df, sqlite_engine, table_name='multi_test')

    df1 = lf.collect()
    assert df1.shape == (3, 2)

    df2 = lf.filter(col('power') < 90).collect()
    assert df2.shape == (2, 2)


def test_temp_table_with_types(sqlite_engine):
    df = pl.DataFrame({
        'a': [1, 2, 3],
        'b': [1.5, 2.5, 3.5],
        'c': ['x', 'y', 'z'],
        'd': [True, False, True]
    })

    lf = scan_df(df, sqlite_engine)
    out_df = lf.collect()

    assert out_df.shape == df.shape
    for c in df.columns:
        assert out_df[c].to_list() == df[c].to_list()


@pytest.mark.skip(reason='Template for manual testing real servers.')
def test_manual_server_upload():
    """Template for manual testing on real servers (Oracle, SQL Server, Teradata).
    
    To use: 
    1. Uncomment and configure the engine for your server.
    2. Run with pytest -s
    """
    # engine = sa.create_engine('oracle+thin://user:pass@host:port/service')
    # engine = sa.create_engine('mssql+pyodbc://user:pass@dsn')
    # engine = sa.create_engine('teradatasql://host:user:pass')

    # if 'engine' not in locals():
    #     pytest.skip("Manual engine not configured")

    # df = pl.DataFrame({'a': [1, 2, 3], 'b': ['high', 'mid', 'low']})
    # lf = scan_df(df, engine, table_name='manual_temp_test')
    # out_df = lf.filter(col('a') > 1).collect()
    # print(out_df)
    # assert out_df.shape == (2, 2)
