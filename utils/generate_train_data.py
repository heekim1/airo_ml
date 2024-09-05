import os
import argparse
import MySQLdb
import csv
from datetime import datetime
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

class AiroDevice:
    def __init__(self, imei, sim, device_type, status, dt_cr, dt_up):
        self.imei = imei
        self.sim = sim
        self.device_type = device_type
        self.status = status
        self.dt_cr = dt_cr
        self.dt_up = dt_up

    @staticmethod
    def get_all():
        connection = get_mysql_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT imei, sim, type, status, dt_cr, dt_up FROM airo_device")
        devices = [
            AiroDevice(imei, sim, type, status, dt_cr, dt_up)
            for imei, sim, type, status, dt_cr, dt_up in cursor.fetchall()
        ]
        connection.close()
        return devices
    
    @staticmethod
    def get(imei):
        connection = get_mysql_connection()
        cursor = connection.cursor()
        
        # Use a parameterized query to safely pass the imei value
        query = "SELECT imei, sim, type, status, dt_cr, dt_up FROM airo_device WHERE imei = %s"
        cursor.execute(query, (imei,))
        
        # Fetch one record
        row = cursor.fetchone()
        connection.close()
        
        # If a record is found, create an AiroDevice instance and return it
        if row:
            imei, sim, device_type, status, dt_cr, dt_up = row
            return AiroDevice(imei, sim, device_type, status, dt_cr, dt_up)
        
        # If no record is found, return None or raise an exception
        return None


def generate_imei_report(device, start_date, stop_date, output_path):
    """
    Generate a report for a single AiroDevice within a specific date range.

    Parameters:
        device (AiroDevice): The device for which to generate the report.
        start_date (datetime or None): The start date of the data range. If None, use the earliest available date.
        stop_date (datetime): The stop date of the data range.
        output_path (str): The directory path where the CSV file will be saved.
    """
    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    connection = get_mysql_connection()
    cursor = connection.cursor()

    if stop_date is None:
        # If stop_date is None, retrieve the latest date for the current IMEI
        query_latest_date = """
            SELECT MAX(dt_cr)
            FROM airo_message
            WHERE imei = %s
        """
        cursor.execute(query_latest_date, (device.imei,))
        latest_date_result = cursor.fetchone()
        if latest_date_result and latest_date_result[0]:
            stop_date = latest_date_result[0]

    # Format start and stop dates for file naming
    start_date_str = start_date.strftime('%Y%m%d')
    stop_date_str = stop_date.strftime('%Y%m%d')

    # Retrieve data for the specified date range
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
        ORDER BY 
            dt_cr ASC
    """
    cursor.execute(query_data, (device.imei, start_date, stop_date))
    rows = cursor.fetchall()

    # Prepare CSV for the individual device with formatted date range
    individual_csv_path = os.path.join(output_path, f'{device.imei}.{start_date_str}_{stop_date_str}.csv')
    with open(individual_csv_path, 'w', newline='') as csvfile_individual:
        fieldnames = ['imei', 'sound_db', 'noise_db', 'breath_rate', 'heart_rate', 'temperature', 'humedity', 'dt_cr', 'time_in_minutes']
        writer_individual = csv.writer(csvfile_individual)
        writer_individual.writerow(fieldnames)

        for row in rows:
            dt_cr = row[6]
            time_in_minutes = dt_cr.hour * 60 + dt_cr.minute
            row_with_time = [device.imei] + list(row) + [time_in_minutes]
            
            # Write to individual CSV
            writer_individual.writerow(row_with_time)

    # Close the database connection
    cursor.close()
    connection.close()

if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Generate IMEI report")
    parser.add_argument("--imei", required=True, help="IMEI number of the device")
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--stop-date", help="Stop date in YYYY-MM-DD format")
    parser.add_argument("--output-dir", default=".", help="Output directory for the CSV file (default is current directory)")

    args = parser.parse_args()

    # Parse the dates
    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    stop_date = datetime.strptime(args.stop_date, "%Y-%m-%d") if args.stop_date else None

    # Retrieve the device
    device = AiroDevice.get(args.imei)
    if device:
        generate_imei_report(device, start_date, stop_date, args.output_dir)
    else:
        print(f"No device found with IMEI {args.imei}")
