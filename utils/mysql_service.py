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

    # Create a CSV for all data
    with open('imei_data_comprehensive.csv', 'w', newline='') as csvfile_all:
        fieldnames = ['imei', 'sound_db', 'noise_db', 'breath_rate', 'heart_rate', 'temperature', 'humedity', 'dt_cr', 'time_in_minutes']
        writer_all = csv.writer(csvfile_all)
        writer_all.writerow(fieldnames)

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
                start_date = datetime(latest_date.year, 7, 1)  # Start date is July 1st

                # Retrieve data for the date range from July 1st to the latest date
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
                        imei = %s AND dt_cr BETWEEN %s AND %s
                """
                cursor.execute(query_data, (device.imei, start_date, latest_date))
                rows = cursor.fetchall()

                # Create a CSV for the individual device
                with open(f'{device.imei}_data_july_to_latest.csv', 'w', newline='') as csvfile_individual:
                    writer_individual = csv.writer(csvfile_individual)
                    writer_individual.writerow(fieldnames)

                    for row in rows:
                        dt_cr = row[6]
                        time_in_minutes = dt_cr.hour * 60 + dt_cr.minute
                        row_with_time = [device.imei] + list(row) + [time_in_minutes]
                        
                        # Write to individual CSV
                        writer_individual.writerow(row_with_time)
                        
                        # Write to the all-inclusive CSV
                        writer_all.writerow(row_with_time)

    cursor.close()
    connection.close()

# Call the function to retrieve data and write to CSV
retrieve_data_and_write_csv()