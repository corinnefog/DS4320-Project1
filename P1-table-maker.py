import pandas as pd
import logging

# Logging setup 
logging.basicConfig(
    filename='data_creation.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

try:
    #  Load raw data 
    logging.info("Loading raw nflverse games CSV")
    df = pd.read_csv("http://www.habitatring.com/games.csv")
    df = df[(df['season'] >= 2014) & (df['season'] <= 2024) & (df['game_type'] == 'REG')]
    logging.info(f"Raw data loaded: {df.shape}")

    #  Table 1: games 
    # Core fact table — one row per game, foreign keys to other tables
    games = df[[
        'game_id', 'season', 'game_type', 'week', 'gameday',
        'away_team', 'home_team', 'away_score', 'home_score',
        'result', 'overtime', 'away_rest', 'home_rest',
        'away_moneyline', 'home_moneyline', 'spread_line',
        'total_line', 'div_game', 'stadium',
        'away_qb_id', 'home_qb_id'
    ]].copy()
    games.to_csv("games.csv", index=False)
    logging.info(f"games.csv saved: {games.shape}")

    # Table 2: teams 
    # One row per unique team — derived from all home/away appearances
    away_teams = df[['away_team']].rename(columns={'away_team': 'team_id'})
    home_teams = df[['home_team']].rename(columns={'home_team': 'team_id'})
    teams = pd.concat([away_teams, home_teams]).drop_duplicates().reset_index(drop=True)
    teams['conference'] = teams['team_id'].map({
        'BUF':'AFC','MIA':'AFC','NE':'AFC','NYJ':'AFC',
        'BAL':'AFC','CIN':'AFC','CLE':'AFC','PIT':'AFC',
        'HOU':'AFC','IND':'AFC','JAX':'AFC','TEN':'AFC',
        'DEN':'AFC','KC':'AFC','LV':'AFC','LAC':'AFC',
        'DAL':'NFC','NYG':'NFC','PHI':'NFC','WAS':'NFC',
        'CHI':'NFC','DET':'NFC','GB':'NFC','MIN':'NFC',
        'ATL':'NFC','CAR':'NFC','NO':'NFC','TB':'NFC',
        'ARI':'NFC','LAR':'NFC','SF':'NFC','SEA':'NFC',
        'OAK':'AFC','SD':'AFC','STL':'NFC'
    })
    teams['division'] = teams['team_id'].map({
        'BUF':'AFC East','MIA':'AFC East','NE':'AFC East','NYJ':'AFC East',
        'BAL':'AFC North','CIN':'AFC North','CLE':'AFC North','PIT':'AFC North',
        'HOU':'AFC South','IND':'AFC South','JAX':'AFC South','TEN':'AFC South',
        'DEN':'AFC West','KC':'AFC West','LV':'AFC West','LAC':'AFC West',
        'DAL':'NFC East','NYG':'NFC East','PHI':'NFC East','WAS':'NFC East',
        'CHI':'NFC North','DET':'NFC North','GB':'NFC North','MIN':'NFC North',
        'ATL':'NFC South','CAR':'NFC South','NO':'NFC South','TB':'NFC South',
        'ARI':'NFC West','LAR':'NFC West','SF':'NFC West','SEA':'NFC West',
        'OAK':'AFC West','SD':'AFC West','STL':'NFC West'
    })
    teams.to_csv("teams.csv", index=False)
    logging.info(f"teams.csv saved: {teams.shape}")

    #  Table 3: stadiums 
    # One row per unique stadium with surface and roof info
    stadiums = df[['stadium', 'roof', 'surface', 'temp', 'wind', 'home_team']].copy()
    stadiums = stadiums.groupby('stadium').agg(
        roof=('roof', 'first'),
        surface=('surface', 'first'),
        home_team=('home_team', 'first'),
        avg_temp=('temp', 'mean'),
        avg_wind=('wind', 'mean')
    ).reset_index()
    stadiums.to_csv("stadiums.csv", index=False)
    logging.info(f"stadiums.csv saved: {stadiums.shape}")

    # Table 4: quarterbacks
    # One row per QB appearance per game (away and home combined)
    away_qbs = df[['game_id', 'away_qb_id', 'away_qb_name', 'away_team']].copy()
    away_qbs.columns = ['game_id', 'qb_id', 'qb_name', 'team_id']
    away_qbs['role'] = 'away'

    home_qbs = df[['game_id', 'home_qb_id', 'home_qb_name', 'home_team']].copy()
    home_qbs.columns = ['game_id', 'qb_id', 'qb_name', 'team_id']
    home_qbs['role'] = 'home'

    quarterbacks = pd.concat([away_qbs, home_qbs]).dropna(subset=['qb_id'])
    quarterbacks = quarterbacks.drop_duplicates().reset_index(drop=True)
    quarterbacks.to_csv("quarterbacks.csv", index=False)
    logging.info(f"quarterbacks.csv saved: {quarterbacks.shape}")

    print("All 4 tables created successfully!")
    print(f"  games.csv:        {games.shape}")
    print(f"  teams.csv:        {teams.shape}")
    print(f"  stadiums.csv:     {stadiums.shape}")
    print(f"  quarterbacks.csv: {quarterbacks.shape}")

except Exception as e:
    logging.error(f"Error during table creation: {e}")
    raise
