import sqlalchemy as sa


def get_dialect_and_driver(conn: sa.Connection | sa.Engine):
    dialect = conn.dialect.name.lower()
    driver = getattr(conn.dialect, 'driver', '').lower()  # e.g. 'psycopg', 'pyodbc', 'oracledb'
    return dialect, driver


def is_teradata(dialect, driver):
    return 'teradata' in dialect or 'teradatasql' in driver
