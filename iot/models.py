# iot/models.py

from .mysql_service import get_mysql_connection


class Device:
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
            Device(imei, sim, type, status, dt_cr, dt_up)
            for imei, sim, type, status, dt_cr, dt_up in cursor.fetchall()
        ]
        connection.close()
        return devices


class Message:
    def __init__(
        self,
        imei,
        status,
        light_lx_h,
        light_lx_l,
        grade_sound,
        sound_db,
        noise_db,
        breath_rate,
        heart_rate,
        reserv_1,
        reserv_2,
        reserv_3,
        reserv_4,
        reserv_5,
        net_rssi_h,
        net_rssi_l,
        coefficient_freq_h,
        coefficient_freq_l,
        coefficient_human,
        coefficient_move,
        reserv_6,
        temperature,
        humedity,
        light_lx,
        net_rssi,
        coefficient_freq,
        original_string,
        dt_cr,
        dt_up,
    ):
        self.imei = imei
        self.status = status
        self.light_lx_h = light_lx_h
        self.light_lx_l = light_lx_l
        self.grade_sound = grade_sound
        self.sound_db = sound_db
        self.noise_db = noise_db
        self.breath_rate = breath_rate
        self.heart_rate = heart_rate
        self.reserv_1 = reserv_1
        self.reserv_2 = reserv_2
        self.reserv_3 = reserv_3
        self.reserv_4 = reserv_4
        self.reserv_5 = reserv_5
        self.net_rssi_h = net_rssi_h
        self.net_rssi_l = net_rssi_l
        self.coefficient_freq_h = coefficient_freq_h
        self.coefficient_freq_l = coefficient_freq_l
        self.coefficient_human = coefficient_human
        self.coefficient_move = coefficient_move
        self.reserv_6 = reserv_6
        self.temperature = temperature
        self.humidity = humedity
        self.light_lx = light_lx
        self.net_rssi = net_rssi
        self.coefficient_freq = coefficient_freq
        self.original_string = original_string
        self.dt_cr = dt_cr
        self.dt_up = dt_up

    @staticmethod
    def get(imei, limit):
        connection = get_mysql_connection()
        cursor = connection.cursor()
        query =  """
            SELECT 
                imei,
                status, 
                light_lx_h, 
                light_lx_l, 
                grade_sound, 
                sound_db, 
                noise_db, 
                breath_rate, 
                heart_rate, 
                reserv_1, 
                reserv_2, 
                reserv_3, 
                reserv_4, 
                reserv_5, 
                net_rssi_h, 
                net_rssi_l, 
                coefficient_freq_h, 
                coefficient_freq_l, 
                coefficient_human, 
                coefficient_move, 
                reserv_6, 
                temperature, 
                humedity, 
                light_lx, 
                net_rssi, 
                coefficient_freq, 
                original_string,
                dt_cr,
                dt_up 
            FROM 
                airo_message
            WHERE 
                imei = %s
            ORDER BY 
                dt_cr DESC
            LIMIT %s
        """
        cursor.execute(query, (imei, limit))
        results = cursor.fetchall()
        connection.close()

        messages = [
            Message(
                imei,
                status,
                light_lx_h,
                light_lx_l,
                grade_sound,
                sound_db,
                noise_db,
                breath_rate,
                heart_rate,
                reserv_1,
                reserv_2,
                reserv_3,
                reserv_4,
                reserv_5,
                net_rssi_h,
                net_rssi_l,
                coefficient_freq_h,
                coefficient_freq_l,
                coefficient_human,
                coefficient_move,
                reserv_6,
                temperature,
                humedity,
                light_lx,
                net_rssi,
                coefficient_freq,
                original_string,
                dt_cr,
                dt_up,
            )
            for imei, status, light_lx_h, light_lx_l, grade_sound, sound_db, noise_db, breath_rate, heart_rate, reserv_1, reserv_2, reserv_3, reserv_4, reserv_5, net_rssi_h, net_rssi_l, coefficient_freq_h, coefficient_freq_l, coefficient_human, coefficient_move, reserv_6, temperature, humedity, light_lx, net_rssi, coefficient_freq, original_string, dt_cr, dt_up in results
        ]
        
        return messages

    
class airo_daily_stat:
    def __init__(
        self,
        ymd,
        imei,
        sound_db,
        noise_db,
        breath_rate,
        heart_rate,
        temperature,
        humedity,
        coefficient_human,
        coefficient_move,
        light_lx,
        net_rssi,
        dt_cr,
        dt_up,
    ):
        self.ymd = ymd
        self.imei = imei
        self.sound_db = sound_db
        self.noise_db = noise_db
        self.breath_rate = breath_rate
        self.heart_rate = heart_rate
        self.temperature = temperature
        self.humidity = humedity
        self.coefficient_human = coefficient_human
        self.coefficient_move = coefficient_move
        self.light_lx = light_lx
        self.net_rssi = net_rssi
        self.dt_cr = dt_cr
        self.dt_up = dt_up


    @staticmethod
    def get_weekly_data(imei, start_date, end_date):
        connection = get_mysql_connection()
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT 
                ymd,
                imei,
                sound_db, 
                noise_db, 
                breath_rate, 
                heart_rate, 
                temperature, 
                humedity, 
                coefficient_human, 
                coefficient_move, 
                light_lx, 
                net_rssi, 
                dt_cr,
                dt_up 
            FROM 
                airo_daily_stat
            WHERE 
                imei = %s AND
                dt_cr BETWEEN %s AND %s
            ORDER BY 
                dt_cr ASC
            limit 100
        """,
            (imei, start_date, end_date),
        )
        messages = [
            airo_daily_stat(
                ymd,
                imei,
                sound_db,
                noise_db,
                breath_rate,
                heart_rate,
                temperature,
                humedity,
                coefficient_human,
                coefficient_move,
                light_lx,
                net_rssi,
                dt_cr,
                dt_up,
            )
            for ymd, imei, sound_db, noise_db, breath_rate, heart_rate, temperature, humedity, coefficient_human, coefficient_move, light_lx, net_rssi, dt_cr, dt_up in cursor.fetchall()
        ]
        connection.close()
        return messages
    

    @staticmethod
    def get_monthly_data(imei, start_date, end_date):
        connection = get_mysql_connection()
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT 
                ymd,
                imei,
                sound_db, 
                noise_db, 
                breath_rate, 
                heart_rate, 
                temperature, 
                humedity, 
                coefficient_human, 
                coefficient_move, 
                light_lx, 
                net_rssi, 
                dt_cr,
                dt_up 
            FROM 
                airo_daily_stat
            WHERE 
                imei = %s AND
                dt_cr BETWEEN %s AND %s
            ORDER BY 
                dt_cr ASC
            limit 100
        """,
            (imei, start_date, end_date),
        )
        messages = [
            airo_daily_stat(
                ymd,
                imei,
                sound_db,
                noise_db,
                breath_rate,
                heart_rate,
                temperature,
                humedity,
                coefficient_human,
                coefficient_move,
                light_lx,
                net_rssi,
                dt_cr,
                dt_up,
            )
            for ymd, imei, sound_db, noise_db, breath_rate, heart_rate, temperature, humedity, coefficient_human, coefficient_move, light_lx, net_rssi, dt_cr, dt_up in cursor.fetchall()
        ]
        connection.close()
        return messages
    
    
class airo_monthly_stat:
    def __init__(
        self,
        ymd,
        imei,
        sound_db,
        noise_db,
        breath_rate,
        heart_rate,
        temperature,
        humedity,
        coefficient_human,
        coefficient_move,
        light_lx,
        net_rssi,
        dt_cr,
        dt_up,
    ):
        self.ymd = ymd
        self.imei = imei
        self.sound_db = sound_db
        self.noise_db = noise_db
        self.breath_rate = breath_rate
        self.heart_rate = heart_rate
        self.temperature = temperature
        self.humidity = humedity
        self.coefficient_human = coefficient_human
        self.coefficient_move = coefficient_move
        self.light_lx = light_lx
        self.net_rssi = net_rssi
        self.dt_cr = dt_cr
        self.dt_up = dt_up


    @staticmethod
    def get_yearly_data(imei, start_date, end_date):
        connection = get_mysql_connection()
        cursor = connection.cursor()
        cursor.execute(
            """
            SELECT 
                ymd,
                imei,
                sound_db, 
                noise_db, 
                breath_rate, 
                heart_rate, 
                temperature, 
                humedity, 
                coefficient_human, 
                coefficient_move, 
                light_lx, 
                net_rssi, 
                dt_cr,
                dt_up 
            FROM 
                airo_daily_stat
            WHERE 
                imei = %s AND
                dt_cr BETWEEN %s AND %s
            ORDER BY 
                dt_cr ASC
            limit 100
        """,
            (imei, start_date, end_date),
        )
        messages = [
            airo_monthly_stat(
                ymd,
                imei,
                sound_db,
                noise_db,
                breath_rate,
                heart_rate,
                temperature,
                humedity,
                coefficient_human,
                coefficient_move,
                light_lx,
                net_rssi,
                dt_cr,
                dt_up,
            )
            for ymd, imei, sound_db, noise_db, breath_rate, heart_rate, temperature, humedity, coefficient_human, coefficient_move, light_lx, net_rssi, dt_cr, dt_up in cursor.fetchall()
        ]
        connection.close()
        return messages