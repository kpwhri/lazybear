from __future__ import annotations

from typing import Sequence

import polars as pl
import sqlalchemy as sa
from sqlalchemy import Engine

from lazybear.core import LazyBearFrame, TempLazyBearFrame


def scan_df(df: pl.DataFrame, engine: Engine, table_name: str | None = None) -> TempLazyBearFrame:
    """Create a TempLazyBearFrame from a polars DataFrame.

    The DataFrame will be inserted into a temporary table on the database when
    the frame is collected or otherwise executed.

    Parameters:
        df : polars.DataFrame
            The DataFrame to insert.
        engine : sqlalchemy.Engine
            The sqlalchemy engine connected to the database.
        table_name : str | None
            Optional name for the temporary table. If not provided, a random name
            will be generated.
    """
    return TempLazyBearFrame(engine, df, table_name)


def scan_table(
        table_name: str,
        engine: Engine,
        schema: str | None = None,
        *,
        lowercase: bool = True,
) -> LazyBearFrame:
    """Create a LazyBearFrame from a database table.

    This reflects the table via sqlalchemy and builds a namespaced selectable.

    Parameters:
        table_name : str
            The table name to reflect.
        engine : sqlalchemy.Engine
            The sqlalchemy engine connected to the database.
        schema : str | None
            Optional schema/namespace of the table.
        lowercase : bool, default True
            If True, expose column names in the returned ``LazyFrame`` as lowercase keys. When False,
            preserve the database/reflected casing.
    """
    meta = sa.MetaData()
    tbl = sa.Table(table_name, meta, schema=schema, autoload_with=engine)

    if lowercase:
        # build a select that labels each column with lowercased name so subquery namespace (sq.c[...]) has lower
        labelled_cols = [col.label(col.key.lower()) for col in tbl.c]
        base_sel = sa.select(*labelled_cols).select_from(tbl)
    else:
        base_sel = sa.select(tbl)

    sq = base_sel.subquery()
    # namespace columns using the keys exposed by the subquery
    cols = {c.key: sq.c[c.key] for c in sq.c}
    return LazyBearFrame(engine, sq, cols)


def scan_sql_query(query: str, engine: Engine, columns: Sequence[str] | None = None) -> LazyBearFrame:
    """Create a LazyFrame from a raw SQL SELECT statement.

    Parameters:
        engine : sqlalchemy.Engine
            The SQLAlchemy engine to execute against.
        query : str
            A raw SQL SELECT statement. Must be a SELECT; DML/DDL is not supported.
        columns : Sequence[str] | None
            Optional explicit list of column names. If omitted, the names are inferred by executing
            the query once to obtain cursor metadata (no rows need to be fetched).

    Notes:
        - This function does not execute the query immediately for data; it only inspects metadata when
          ``columns`` is not provided in order to build a selectable with named columns.
        - The returned ``LazyBearFrame`` can be composed with other operations (select/filter/join/etc.).
    """
    if not isinstance(query, str):
        raise TypeError('query must be a string')
    q_strip = query.lstrip().lower()
    if not q_strip.startswith('select'):
        raise ValueError('sca_sql_query() expects a SELECT statement')

    names: list[str]
    if columns is None:
        # execute once to obtain column names from result metadata
        with engine.connect() as conn:
            res = conn.execute(sa.text(query))
            names = list(res.keys())
            # no need to fetch rows; close the cursor
            res.close()
    else:
        names = list(columns)

    # build a textual selectable with declared columns so it can be used in FROM/SELECT contexts
    txt = sa.text(query).columns(*[sa.column(n) for n in names])
    sq = txt.subquery()
    cols = {c.key: sq.c[c.key] for c in sq.c}
    return LazyBearFrame(engine, sq, cols)
