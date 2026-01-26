import polars as pl
import sqlalchemy as sa

from lazybear.db.insert.bulk_mssql import bulk_insert_pyodbc_fast_executemany
from lazybear.db.insert.bulk_oracledb import bulk_insert_oracledb_executemany
from lazybear.db.insert.bulk_psycopg3 import bulk_insert_psycopg_copy
from lazybear.db.insert.bulk_teradata import bulk_insert_teradatasql_executemany


def bulk_insert_fast(conn: sa.Connection, df: pl.DataFrame, table_name: str, columns: list[str]) -> None:
    dialect = conn.dialect.name
    driver = getattr(conn.dialect, 'driver', '')  # e.g. 'psycopg', 'pyodbc', 'oracledb'

    if dialect == 'postgresql' and 'psycopg' in driver:
        bulk_insert_psycopg_copy(conn, df, full_table_name=table_name, columns=columns)
        return

    if dialect == 'mssql' and 'pyodbc' in driver:
        bulk_insert_pyodbc_fast_executemany(conn, df, full_table_name=table_name, columns=columns)
        return

    if dialect == 'oracle' and 'oracledb' in driver:
        bulk_insert_oracledb_executemany(conn, df, full_table_name=table_name, columns=columns)
        return

    if dialect == 'teradata' or 'teradatasql' in driver:
        bulk_insert_teradatasql_executemany(conn, df, full_table_name=table_name, columns=columns)
        return

    raise NotImplementedError(f'No fast bulk insert path for {dialect}+{driver}')
