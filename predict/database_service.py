import os
import MySQLdb
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Load environment variables from .env file
load_dotenv()

DATABASES = {
    "mysql": {
        "ENGINE": os.environ.get("DB_ENGINE"),
        "NAME": os.environ.get("SQL_NAME"),
        "USER": os.environ.get("SQL_USER"),
        "PASSWORD": os.environ.get("SQL_PASSWORD"),
        "HOST": os.environ.get("SQL_HOST"),
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


def get_latest_20_data(imei):
    connection = get_mysql_connection()
    cursor = connection.cursor()
    query = """
        SELECT sound_db, noise_db, breath_rate, heart_rate, temperature, humedity, dt_cr
        FROM airo_message
        WHERE imei = %s
        ORDER BY dt_cr DESC
        LIMIT 20
    """
    cursor.execute(query, (imei,))
    rows = cursor.fetchall()
    connection.close()
    return rows

def get_latest_60_data(imei):
    connection = get_mysql_connection()
    cursor = connection.cursor()
    query = """
        SELECT sound_db, noise_db, breath_rate, heart_rate, temperature, humedity, dt_cr
        FROM airo_message
        WHERE imei = %s
        ORDER BY dt_cr DESC
        LIMIT 60
    """
    cursor.execute(query, (imei,))
    rows = cursor.fetchall()
    connection.close()
    return rows


def get_latest_120_data(imei):
    connection = get_mysql_connection()
    cursor = connection.cursor()
    query = """
        SELECT sound_db, noise_db, breath_rate, heart_rate, temperature, humedity, dt_cr
        FROM airo_message
        WHERE imei = %s
        ORDER BY dt_cr DESC
        LIMIT 120
    """
    cursor.execute(query, (imei,))
    rows = cursor.fetchall()
    connection.close()
    return rows

def get_latest_180_data(imei):
    connection = get_mysql_connection()
    cursor = connection.cursor()
    query = """
        SELECT sound_db, noise_db, breath_rate, heart_rate, temperature, humedity, dt_cr
        FROM airo_message
        WHERE imei = %s
        ORDER BY dt_cr DESC
        LIMIT 180
    """
    cursor.execute(query, (imei,))
    rows = cursor.fetchall()
    connection.close()
    return rows

def get_latest_240_data(imei):
    connection = get_mysql_connection()
    cursor = connection.cursor()
    query = """
        SELECT sound_db, noise_db, breath_rate, heart_rate, temperature, humedity, dt_cr
        FROM airo_message
        WHERE imei = %s
        ORDER BY dt_cr DESC
        LIMIT 240
    """
    cursor.execute(query, (imei,))
    rows = cursor.fetchall()
    connection.close()
    return rows