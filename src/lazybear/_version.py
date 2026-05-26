from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('lazybear-polars')
except PackageNotFoundError:
    __version__ = '0+unknown'
