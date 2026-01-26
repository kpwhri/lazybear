import io
import csv
import polars as pl
import sqlalchemy as sa

def bulk_insert_psycopg_copy(
    conn: sa.Connection,
    df: pl.DataFrame,
    *,
    full_table_name: str,
    columns: list[str],
) -> None:
    """
    Fast path for Postgres + psycopg3 using COPY FROM STDIN.
    Requires a psycopg3 connection underneath sqlalchemy.

    Args:
         full_table_name:  e.g. '"pg_temp"."vdw_temp_..."' or '"public"."t"'
    """
    if df.height == 0:
        return

    raw = conn.connection  # sqlalchemy -> dbapi connection (psycopg.Connection)
    # sqlalchemy may wrap; psycopg3 exposes .cursor() directly.
    with raw.cursor() as cur:
        # csv is the easiest universal encoding for COPY (avoiding binary)
        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")

        # stream rows out of polars without materializing everything
        for row in df.select(columns).iter_rows():
            writer.writerow(row)

        buf.seek(0)

        col_list = ', '.join(f'"{c}"' for c in columns)  # sketch quoting
        sql = f'COPY {full_table_name} ({col_list}) FROM STDIN WITH (FORMAT csv)'

        # psycopg3: copy protocol
        with cur.copy(sql) as copy:
            copy.write(buf.getvalue())