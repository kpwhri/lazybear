import polars as pl


def clean_teradata(df: pl.DataFrame) -> pl.DataFrame:
    """Strip trailing non-breaking spaces returned by Teradata character datatypes.
    In my version, this is realized as a ' ' (32/0x20), but polars suggests it is NBSP. Safest
        seems to be `strip_chars_end()`

    This is a workaround for: https://github.com/Teradata/python-driver/blob/master/samples/CharPadding.py
    """
    string_cols = [
        name for name, dtype in df.schema.items()
        if dtype in {pl.String, pl.Utf8}
    ]
    if not string_cols:
        return df

    return df.with_columns(
        pl.col(string_cols).str.strip_chars_end()
    )
