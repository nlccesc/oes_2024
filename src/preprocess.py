import pandas as pd

class DataPreprocessing:
    def __init__(self):
        pass

    def preprocess_telemetry(self, df):
        df['timestamp'] = pd.to_datetime(df['create_dt'], errors='coerce')
        
        if df['timestamp'].isnull().any():
            unparsed_rows = df[df['timestamp'].isnull()]
            print(f"Found {len(unparsed_rows)} unparsed rows, attempting alternative formats...")

            df.loc[df['timestamp'].isnull(), 'timestamp'] = pd.to_datetime(
                df.loc[df['timestamp'].isnull(), 'create_dt'], format='%Y-%m-%d', errors='coerce'
            )

        df.dropna(subset=['timestamp'], inplace=True)
        df.drop(columns=['create_dt'], inplace=True)
        
        return df

    def merge_with_labels(self, telemetry_df, labels_df):
        labels_df['start_time'] = pd.to_datetime(labels_df['start_time'])
        labels_df['end_time'] = pd.to_datetime(labels_df['end_time'])

        telemetry_df['timestamp'] = pd.to_datetime(telemetry_df['timestamp'])
        
        merged_df = pd.merge_asof(
            telemetry_df.sort_values('timestamp'),
            labels_df.sort_values('start_time'),
            left_on='timestamp',
            right_on='start_time',
            direction='backward'
        )
        return merged_df
