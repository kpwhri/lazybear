import polars as pl
import sqlalchemy as sa

from lazybear.db.clean.clean_teradata import clean_teradata
from lazybear.db.db_utils import get_dialect_and_driver, is_teradata


def clean_dataframe(conn: sa.Engine, df: pl.DataFrame):
    dialect, driver = get_dialect_and_driver(conn)

    if is_teradata(dialect, driver):
        return clean_teradata(df)
    return df
