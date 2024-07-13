import pandas as pd

def parse_imei_data(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)

    # Exclude zeros for breath_rate and heart_rate
    df_non_zero_breath = df[df['breath_rate'] != 0]
    df_non_zero_heart = df[df['heart_rate'] != 0]

    # Calculate overall averages
    overall_averages = {
        'sound_db': df['sound_db'].mean(),
        'noise_db': df['noise_db'].mean(),
        'breath_rate': df_non_zero_breath['breath_rate'].mean(),
        'heart_rate': df_non_zero_heart['heart_rate'].mean(),
        'temperature': df['temperature'].mean(),
        'humedity': df['humedity'].mean()
    }
    
    # Calculate averages grouped by imei
    grouped_averages = df.groupby('imei').apply(lambda x: pd.Series({
        'sound_db': x['sound_db'].mean(),
        'noise_db': x['noise_db'].mean(),
        'breath_rate': x[x['breath_rate'] != 0]['breath_rate'].mean(),
        'heart_rate': x[x['heart_rate'] != 0]['heart_rate'].mean(),
        'temperature': x['temperature'].mean(),
        'humedity': x['humedity'].mean()
    }))
    
    return overall_averages, grouped_averages

def main():
    file_path = '/Users/heeshinkim/Desktop/Airosolution/ml//data/imei_data_last_month.csv'
    overall_averages, grouped_averages = parse_imei_data(file_path)
    
    print("Overall Averages:")
    print(overall_averages)
    print("\nAverages Grouped by IMEI:")
    print(grouped_averages)

if __name__ == "__main__":
    main()

