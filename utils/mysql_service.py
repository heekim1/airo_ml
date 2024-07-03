import os
import MySQLdb
import csv
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

DATABASES = {
    "mysql": {
        "ENGINE": os.environ.get("ENGINE"),
        "NAME": os.environ.get("NAME"),
        "USER": os.environ.get("USER"),
        "PASSWORD": os.environ.get("PASSWORD"),
        "HOST": os.environ.get("HOST"),
        "PORT": os.environ.get("PORT"),
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

class airo_device:
    def __init__(self, imei, sim, type, status, dt_cr, dt_up):
        self.imei = imei
        self.sim = sim
        self.type = type
        self.status = status
        self.dt_cr = dt_cr
        self.dt_up = dt_up

    @staticmethod
    def get_all():
        connection = get_mysql_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT imei, sim, type, status, dt_cr, dt_up FROM airo_device")
        devices = [
            airo_device(imei, sim, type, status, dt_cr, dt_up)
            for imei, sim, type, status, dt_cr, dt_up in cursor.fetchall()
        ]
        connection.close()
        return devices
    


def retrieve_data_and_write_csv():
    devices = airo_device.get_all()

    connection = get_mysql_connection()
    cursor = connection.cursor()

    with open('imei_data_last_month.csv', 'w', newline='') as csvfile:
        fieldnames = ['imei', 'sound_db', 'noise_db', 'breath_rate', 'heart_rate', 'temperature', 'humedity', 'dt_cr']
        writer = csv.writer(csvfile)
        
        writer.writerow(fieldnames)

        for device in devices:
            # Get the latest date for the current IMEI
            query_latest_date = """
                SELECT MAX(dt_cr) 
                FROM airo_message 
                WHERE imei = %s
            """
            cursor.execute(query_latest_date, (device.imei,))
            latest_date_result = cursor.fetchone()
            if latest_date_result and latest_date_result[0]:
                latest_date = latest_date_result[0]
                one_month_ago = latest_date - timedelta(days=30)
                
                # Retrieve data for the last month for the current IMEI
                query_data = """
                    SELECT 
                        sound_db, 
                        noise_db, 
                        breath_rate, 
                        heart_rate, 
                        temperature, 
                        humedity, 
                        dt_cr
                    FROM 
                        airo_message
                    WHERE 
                        imei = %s AND dt_cr >= %s
                """
                cursor.execute(query_data, (device.imei, one_month_ago))
                rows = cursor.fetchall()

                # Write rows to CSV directly
                for row in rows:
                    writer.writerow([device.imei] + list(row))

    connection.close()

# Call the function to retrieve data and write to CSV
retrieve_data_and_write_csv()