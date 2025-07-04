from src.data_pipeline.fastf1_client import FastF1Client

def save_df(df, name):
    if not df.empty:
        df.to_csv(f"{name}.csv", index=False)
        df.to_json(f"{name}.json", orient="records", lines=True)
        print(f"Saved {name}.csv and {name}.json")
    else:
        print(f"No data for {name}")

if __name__ == "__main__":
    client = FastF1Client()
    sessions = client.get_race_weekend(2024, 'Brazil')
    if 'R' in sessions:
        race_session = sessions['R']
        save_df(client.get_session_results(race_session), "brazil2024_results")
        save_df(client.get_lap_data(race_session), "brazil2024_laps")
        save_df(client.get_tire_strategy_data(race_session), "brazil2024_tires")
        save_df(client.get_position_data(race_session), "brazil2024_positions")
    else:
        print("Race session not found for Brazil 2024.")