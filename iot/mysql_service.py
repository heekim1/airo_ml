import os
import MySQLdb
import csv
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DATABASES = {
    "mysql": {
        "ENGINE": os.environ.get("DB_ENGINE"),
        "NAME": os.environ.get("SQL_NAME"),
        "USER": os.environ.get("SQL_USER"),
        "PASSWORD": os.environ.get("SQL_PASSWORD"),
        "HOST": "3.39.89.97", #os.environ.get("SQL_HOST"),
        "PORT": os.environ.get("SQL_PORT"),
    },
}
print(f"DATABASES: {DATABASES}")

def get_mysql_connection():
    db_settings = DATABASES["mysql"]
    connection = MySQLdb.connect(
        host=db_settings["HOST"],
        user=db_settings["USER"],
        password=db_settings["PASSWORD"],
        database=db_settings["NAME"],
        port=int(db_settings["PORT"]),
    )
    return connection
