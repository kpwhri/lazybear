import datetime as dt
import polars as pl
import sqlalchemy as sa


def bulk_insert_oracledb_executemany(
        conn: sa.Connection,
        df: pl.DataFrame,
        *,
        full_table_name: str,
        columns: list[str],
        chunk_size: int = 10_000,
) -> None:
    """
    Fast path for Oracle + oracledb: array binding via cursor.executemany.

    Args:
        full_table_name: e.g. '"SCHEMA"."GTT_NAME"' (GTT is a schema object)
    """
    if df.height == 0:
        return

    raw = conn.connection  # DBAPI connection (oracledb.Connection)
    cur = raw.cursor()
    try:
        # Optional but often beneficial: set input sizes to avoid repeated type inference
        # Sketch only: you’ll likely map each column to an oracledb type/size based on dtype + max length.
        # Example:
        # import oracledb
        # cur.setinputsizes(None, oracledb.DB_TYPE_VARCHAR, ...)

        col_list = ', '.join(f'"{c}"' for c in columns)  # sketch quoting
        bind_list = ', '.join(f':{i + 1}' for i in range(len(columns)))  # positional binds :1, :2, ...
        sql = f'INSERT INTO {full_table_name} ({col_list}) VALUES ({bind_list})'

        # Build batches as list-of-tuples (or list-of-lists)
        batch: list[tuple] = []
        for row in df.select(columns).iter_rows():
            batch.append(row)
            if len(batch) >= chunk_size:
                cur.executemany(sql, batch)
                batch.clear()

        if batch:
            cur.executemany(sql, batch)

        # commit/rollback is controlled by sqlalchemy transaction
    finally:
        cur.close()
