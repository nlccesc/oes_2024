class FeatureEngineering:
    def __init__(self):
        pass

    def create_features(self, df):
        required_columns = ['alt', 'lat', 'lon', 
                            'accel_forward_nn', 'accel_braking_nn', 
                            'accel_angular_nn', 'accel_vertical_nn']

        for column in required_columns:
            if column not in df.columns:
                raise KeyError(f"Required column '{column}' not found in DataFrame")

        df['altitude_diff'] = df['alt'].diff().fillna(0)
        df['latitude_diff'] = df['lat'].diff().fillna(0)
        df['longitude_diff'] = df['lon'].diff().fillna(0)
        df['distance'] = (df['latitude_diff']**2 + df['longitude_diff']**2).apply(lambda x: x**0.5)

        df['accel_forward_change'] = df['accel_forward_nn'].diff().fillna(0)
        df['accel_braking_change'] = df['accel_braking_nn'].diff().fillna(0)
        df['accel_angular_change'] = df['accel_angular_nn'].diff().fillna(0)
        df['accel_vertical_change'] = df['accel_vertical_nn'].diff().fillna(0)
        
        return df
