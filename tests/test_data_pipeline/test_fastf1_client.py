from src.data_pipeline.fastf1_client import FastF1Client

if __name__ == "__main__":
    client = FastF1Client()
    sessions = client.get_race_weekend(2023, 'Monaco')
    if 'R' in sessions:
        race_session = sessions['R']
        print("Session Results:")
        print(client.get_session_results(race_session).head())
        print("\nLap Data:")
        print(client.get_lap_data(race_session).head())
        print("\nTire Strategy Data:")
        print(client.get_tire_strategy_data(race_session).head())
        print("\nPosition Data:")
        print(client.get_position_data(race_session).head())
    else:
        print("Race session not found for Monaco 2023.") 