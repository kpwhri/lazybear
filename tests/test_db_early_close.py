from contextlib import closing

import polars as pl
from sqlalchemy import event

import lazybear as lb


def test_collect_iter(sqlite_engine):
    users = lb.scan_table('users', sqlite_engine).order_by('id')

    with closing(users.collect_batches(chunk_size=2)) as batches:
        chunks = list(batches)

    assert len(chunks) == 2
    assert sum(ch.shape[0] for ch in chunks) == 4


def test_collect_iter_early_close(sqlite_engine):
    users = lb.scan_table('users', sqlite_engine).order_by('id')

    checkins = []

    def on_checkin(dbapi_conn, connection_record):
        checkins.append(connection_record)

    event.listen(sqlite_engine.pool, 'checkin', on_checkin)

    try:
        with closing(users.collect_batches(chunk_size=2)) as batches:
            chunk = next(batches)
            assert chunk.shape[0] == 2

            # Connection should still be held by the generator.
            before_close = len(checkins)

        # Exiting closing() should return the connection to the pool.
        assert len(checkins) == before_close + 1
    finally:
        event.remove(sqlite_engine.pool, 'checkin', on_checkin)


def test_temp_table_sqlite_early_close(sqlite_engine, users_df):
    temp_df = lb.scan_df(users_df, sqlite_engine, table_name='heroes')
    temp_df = temp_df.filter(lb.col('age') >= 30).select('name')

    table_exists_on_checkin = []

    def on_checkin(dbapi_conn, connection_record):
        exists = dbapi_conn.execute(
            f"""
            SELECT 1
            FROM sqlite_temp_master
            WHERE type = 'table' AND name = '{temp_df._table_name}'
            """
        ).fetchone() is not None

        table_exists_on_checkin.append(exists)

    event.listen(sqlite_engine, 'checkin', on_checkin)

    try:
        with closing(temp_df.collect_batches(chunk_size=1)) as batches:
            chunk = next(batches)
            assert chunk.height == 1

            checkins_before_close = len(table_exists_on_checkin)

        # closing the generator returns its connection
        assert len(table_exists_on_checkin) > checkins_before_close

        # ensure temp table was dropped before the connection returned
        assert table_exists_on_checkin[-1] is False

        # query can be executed again successfully
        out_df = temp_df.collect()
        exp_df = users_df.filter(pl.col('age') >= 30).select('name')

        assert sorted(out_df['name'].to_list()) == sorted(exp_df['name'].to_list())

    finally:
        event.remove(sqlite_engine, 'checkin', on_checkin)
