from sqlalchemy import create_engine
import pandas as pd
import urllib
import json

SERVER = "innotrack-sql-server.database.windows.net"
DATABASE = "InnoTrackDB"
USERNAME = "innotrackadmin"
PASSWORD = "Innotrack@admin233"

params = urllib.parse.quote_plus(
    f"DRIVER={{ODBC Driver 18 for SQL Server}};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    f"UID={USERNAME};"
    f"PWD={PASSWORD};"
    "Encrypt=yes;"
    "TrustServerCertificate=no;"
    "Connection Timeout=30;"
)

connection_string = (
    f"mssql+pyodbc:///?odbc_connect={params}"
)

engine = create_engine(connection_string)

try:

    with engine.connect() as conn:
        print("SQL Connected Successfully")

except Exception as e:
    print("Connection Failed")
    print(e)





def load_preprocessed_projects():

    query = """
    SELECT *
    FROM PreProcessed_Projects
    """

    df = pd.read_sql(
        query,
        engine
    )

    if "features" in df.columns:

        df["features"] = (
            df["features"]
            .apply(json.loads)
        )

    return df