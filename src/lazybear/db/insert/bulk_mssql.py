import polars as pl
import sqlalchemy as sa

def bulk_insert_pyodbc_fast_executemany(
    conn: sa.Connection,
    df: pl.DataFrame,
    *,
    full_table_name: str,     # e.g. "#vdw_temp_..." or "[dbo].[t]"
    columns: list[str],
    chunk_size: int = 10_000,
) -> None:
    """
    Fast path for SQL Server + pyodbc: enable cursor.fast_executemany and send rows as tuples.
    """
    if df.height == 0:
        return

    raw = conn.connection  # DBAPI connection (pyodbc.Connection)
    cur = raw.cursor()
    try:
        cur.fast_executemany = True

        col_list = ', '.join(f'[{c}]' for c in columns)  # sketch quoting
        placeholders = ', '.join('?' for _ in columns)
        sql = f'INSERT INTO {full_table_name} ({col_list}) VALUES ({placeholders})'

        # stream in chunks to avoid huge memory use
        batch: list[tuple] = []
        for row in df.select(columns).iter_rows():
            batch.append(row)
            if len(batch) >= chunk_size:
                cur.executemany(sql, batch)
                batch.clear()

        if batch:
            cur.executemany(sql, batch)

        # commit is handled by the outer sqlalchemy transaction so don’t call raw.commit() here.
    finally:
        cur.close()