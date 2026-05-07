from types import SimpleNamespace

import polars as pl

from lazybear.db.clean.clean_teradata import clean_teradata
from lazybear.db.clean.dispatch import clean_dataframe


def _fake_conn(dialect_name: str, driver: str = ''):
    return SimpleNamespace(
        dialect=SimpleNamespace(
            name=dialect_name,
            driver=driver,
        )
    )


def test_clean_teradata_returns_original_frame_when_no_string_columns():
    df = pl.DataFrame({
        'id': [1, 2],
        'amount': [10.5, 20.0],
    })

    out = clean_teradata(df)

    assert out.equals(df)


def test_clean_teradata_strips_trailing_whitespace_from_strings():
    df = pl.DataFrame({
        'name': ['Ahti\u00a0\u00a0', 'Kalma', None],
        'code': ['ABC\u00a0', '\u00a0ABC\u00a0', 'XYZ  '],
        'mixed': ['ABC \t\n', 'XYZ\u00a0 ', 'QRS'],
        'age': [30, 28, 41],
    })

    out = clean_teradata(df)

    assert out['name'].to_list() == ['Ahti', 'Kalma', None]
    assert out['code'].to_list() == ['ABC', '\u00a0ABC', 'XYZ']
    assert out['mixed'].to_list() == ['ABC', 'XYZ', 'QRS']
    assert out['age'].to_list() == [30, 28, 41]


def test_clean_teradata_strips_regular_trailing_spaces():
    df = pl.DataFrame({
        'value': ['ABC  ', 'XYZ\u00a0  ', 'QRS \u00a0'],
    })

    out = clean_teradata(df)

    assert out['value'].to_list() == ['ABC', 'XYZ', 'QRS']


def test_clean_dataframe_dispatch_cleans_for_teradata_dialect():
    conn = _fake_conn('teradata')
    df = pl.DataFrame({'name': ['Ahti\u00a0']})

    out = clean_dataframe(conn, df)

    assert out['name'].to_list() == ['Ahti']


def test_clean_dataframe_dispatch_cleans_for_teradatasql_driver():
    conn = _fake_conn('default', 'teradatasql')
    df = pl.DataFrame({'name': ['Ahti\u00a0']})

    out = clean_dataframe(conn, df)

    assert out['name'].to_list() == ['Ahti']


def test_clean_dataframe_dispatch_does_not_clean_for_non_teradata():
    conn = _fake_conn('sqlite')
    df = pl.DataFrame({'name': ['Ahti\u00a0']})

    out = clean_dataframe(conn, df)

    assert out['name'].to_list() == ['Ahti\u00a0']
