import polars as pl
import sqlalchemy as sa


def bulk_insert_teradatasql_executemany(
        conn: sa.Connection,
        df: pl.DataFrame,
        *,
        full_table_name: str,
        columns: list[str],
        chunk_size: int = 5_000,
) -> None:
    """
    Fast cross-platform path for Teradata using teradatasql executemany in batches.

    Args:
        full_table_name: e.g. '"dbc"."volatile_table_name"' or just the volatile table name
    """
    if df.height == 0:
        return

    raw = conn.connection  # DBAPI connection (teradatasql.Connection)
    cur = raw.cursor()
    try:
        col_list = ', '.join(f'"{c}"' for c in columns)  # sketch quoting
        placeholders = ', '.join('?' for _ in columns)  # teradatasql supports qmark in many setups
        sql = f'INSERT INTO {full_table_name} ({col_list}) VALUES ({placeholders})'

        batch: list[tuple] = []
        for row in df.select(columns).iter_rows():
            batch.append(row)
            if len(batch) >= chunk_size:
                cur.executemany(sql, batch)
                batch.clear()

        if batch:
            cur.executemany(sql, batch)
    finally:
        cur.close()
