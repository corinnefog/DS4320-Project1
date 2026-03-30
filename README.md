# DS4320-Project1 - NFL Point Spread Model

Name: Corinne Fogarty

NetID: qfr4cu

DOI:

[Press Release](/Press-Release.md)

Data: [Data](https://myuva-my.sharepoint.com/:f:/g/personal/qfr4cu_virginia_edu/IgCsxlm2yylQQKvGMnDODUY8AUfh3B7grQfh3l9pqinbWXI?e=gDn7aj)

Pipeline:

License


## Problem Definition

**Initial General Problem** - NFL game outcomes are inherently difficult to predict due to the complex interaction of team quality, game conditions, and situational factors. Fans, analysts, and sports bettors alike seek to understand what drives the margin of victory in NFL games. 

**Refined Specific Problem** - Using game-level data from the 2014-2024 NFL regular seasons, identify which measurable factors — including Vegas spread lines, rest days, weather conditions, home field advantage, and divisional matchups — are most strongly associated with the margin of victory (home score minus away score), and evaluate how well these factors predict game outcomes compared to the Vegas baseline.

**Refinement Rationale** - The general problem of predicting NFL outcomes was refined in several ways to make it tractable and analytically meaningful. First, the outcome was narrowed from a binary win/loss prediction to margin of victory (point spread), which preserves more information about game outcomes and aligns with how the sports betting market frames predictions. Second, the scope was limited to regular season games from 2014 to 2024, excluding playoff games where team composition and preparation differ systematically from the regular season. Third, rather than building a pure black-box prediction model, the project was framed as a factor analysis — identifying which variables drive outcomes — because this produces interpretable insights about the game that are useful beyond just making predictions. Finally, the Vegas spread line was retained as a benchmark rather than excluded as a feature, allowing the analysis to evaluate whether any observable factors add predictive value beyond what the betting market already prices in.

**Motivation** - The NFL is the most watched sports league in the United States, generating billions of dollars in revenue and attracting enormous interest from fans, media, and the rapidly growing sports betting industry. Despite this interest, there is surprisingly little transparent, data-driven public analysis of what actually drives game outcomes beyond general team quality. Understanding which factors — rest advantages, weather, home field, divisional familiarity — meaningfully influence margins of victory has practical value for analysts, coaches, and bettors, and contributes to the broader scientific understanding of performance in professional sports. This project uses publicly available, well-documented data to bring analytical rigor to a question that millions of people engage with every week during the NFL season.

[Jump to Press Release](/Press-Release.md)


## Domain Exposition

**Terminology** -
| Term | Definition |
|------|------------|
| **ATS (Against the Spread)** | Whether a team won by more than the Vegas point spread; a tougher benchmark than straight win/loss |
| **Point Spread** | The predicted margin of victory set by oddsmakers; used to handicap the favorite |
| **Moneyline** | A bet on which team wins outright, with odds reflecting implied probability |
| **Home Field Advantage** | The empirical tendency for home teams to win more often, attributed to crowd noise, travel fatigue, and familiarity |
| **Rest Differential** | Difference in days of rest between the two teams before the game |
| **DVOA (Defense-adjusted Value Over Average)** | An advanced NFL efficiency metric that adjusts team performance for opponent strength |
| **EPA (Expected Points Added)** | Measures how much a play increases or decreases a team's expected scoring; a key advanced stat |
| **Turnover Differential** | Difference between turnovers gained and lost; strongly correlated with game outcomes |
| **SRS (Simple Rating System)** | A team quality metric combining average margin of victory and strength of schedule |
| **Binary Classification** | A ML task where the model predicts one of two outcomes — here, home team win vs. loss |
| **Features / Predictors** | Input variables used by the model (e.g., rest days, recent form, spread) |
| **Implied Probability** | The win probability embedded in Vegas odds; a strong pre-game baseline to beat |

**Domain** - This project lives at the intersection of sports analytics and predictive modeling. Sports analytics is a rapidly growing field that applies statistical and machine learning methods to professional sports data in order to understand performance, inform strategy, and evaluate outcomes. Within sports analytics, NFL game prediction is a well-studied problem that draws on concepts from economics (betting markets and information efficiency), statistics (regression modeling and feature selection), and sports science (fatigue, home field effects, and weather impacts on performance). The NFL presents a particularly interesting domain because the betting market — represented by the Vegas spread line — provides a powerful baseline prediction that aggregates enormous amounts of expert knowledge. Any data-driven model must contend with this baseline, making the domain as much about understanding market efficiency as it is about predicting scores.


## Data Creation

**Provenance** - The dataset was acquired from the nflverse project, a community-maintained collection of NFL data. Game-level schedule and outcome data was loaded directly from the nflverse games CSV hosted at habitatring.com, originally compiled and maintained by Lee Sharpe. The data was filtered to include only regular season games (game_type == 'REG') from the 2014 through 2024 NFL seasons. The resulting dataset contains one row per game with scores, Vegas lines, rest days, weather, and stadium information. The raw CSV was downloaded on 03/22/26 and stored without modification before filtering.

**Code** - 

**Bias Identification** -
1. Survivorship bias — only teams/franchises that existed across the full period are fully represented; expansion teams or relocated franchises have fewer records
2. Recency bias — rule changes (e.g., pass interference rules, overtime rules) make older seasons less comparable to recent ones
3. Reporting bias — some advanced stats (e.g., air yards, pressure rates) are only available for more recent seasons, so older records will have more missing values
4. Home field advantage confounding — the dataset over-represents certain stadiums/climates if some teams host more games in extreme conditions
5. Vegas line bias — if you include the Vegas spread as a feature, it already encodes a lot of market information and could dominate the model

**Bias Mitigation** -
- Limit analysis to seasons with consistent rule sets, or include season as a control variable
- Standardize stats within season to account for rule-change-driven scoring inflation
- Use cross-validation with season-based splits (not random splits) to avoid data leakage across time
- Report model performance separately by season to detect temporal drift
- If using Vegas line, treat it as a baseline benchmark rather than a feature

**Rationale** -
1. **Data source:** The nflverse games dataset was selected over manual scraping from
Pro Football Reference because it provides a clean, one-row-per-game structure with
Vegas spread lines (spread_line) already included. Spread and total lines are not
easily available through the Pro Football Reference Team Game Finder interface, making
nflverse the more appropriate source for a point spread prediction project.

2. **Year range (2014-2024):** Ten seasons of regular season data were selected to
balance sample size against consistency. NFL rules have changed significantly over
time — particularly around pass interference, overtime rules, and kickoffs — making
data from before 2014 less comparable to modern play. Ten seasons provides
approximately 2,720 games, which is sufficient for training a predictive model while
keeping the data relatively homogeneous in terms of rules and playing conditions.

3. **Regular season only:** Playoff games were excluded because they differ
systematically from regular season games in ways that would confound the model.
Playoff teams are a non-random subset of the league, preparation time is longer,
and the stakes are higher — all of which affect game outcomes in ways that are
not captured by regular season statistics. Including playoffs would introduce
selection bias into the training data.

4. **Unit of analysis:** The game was chosen as the unit of analysis rather than
the team-game or the individual play. This is the natural unit for point spread
prediction, since a spread is defined at the game level. The nflverse dataset
provides one row per game with home and away scores already separated, making
no reshaping necessary.

5. **Target variable:** The result column (home_score minus away_score) was used
as the target variable rather than a binary win/loss outcome. This preserves more
information about game outcomes and aligns directly with the point spread, which
is a continuous prediction of margin of victory rather than a binary classification.

6. **No missing data imputation at this stage:** Some fields such as temp and wind
are only populated for outdoor games and are missing for dome or closed-roof stadiums.
No imputation was applied to the raw data — missing values will be handled at the
analysis stage, either by imputing a neutral value (e.g. 72 degrees for dome games)
or by dropping those columns depending on their predictive contribution. This decision
preserves full provenance of the raw data.

## Metadata

**Schema**
<img src="/images/ER-Diagram.png" alt="ER Diagram">

**Data**
| Table Name | Description | Link to File |
|------------|-------------|--------------|
| Stadiums  | One row per stadium,  |[Data]('https://myuva-my.sharepoint.com/:x:/r/personal/qfr4cu_virginia_edu/_layouts/15/Doc.aspx?sourcedoc=%7BB192DC75-E734-4DC1-BCEE-082A1785F860%7D&file=stadiums.csv&action=default&mobileredirect=true')  |
| Teams | One row per team,  |[Data]('https://myuva-my.sharepoint.com/:x:/r/personal/qfr4cu_virginia_edu/_layouts/15/Doc.aspx?sourcedoc=%7BD0844C54-C335-4B85-8C56-4A0D1B8A9AD0%7D&file=teams.csv&action=default&mobileredirect=true') |
| Games| One row per game,  |[Data]('https://myuva-my.sharepoint.com/:x:/r/personal/qfr4cu_virginia_edu/_layouts/15/Doc.aspx?sourcedoc=%7B9BF91CE5-4137-4015-A4ED-41AA986FE346%7D&file=games.csv&action=default&mobileredirect=true') | 
| Quarterbacks| One row per quarterback appearance per game| [Data]('https://myuva-my.sharepoint.com/:x:/r/personal/qfr4cu_virginia_edu/_layouts/15/Doc.aspx?sourcedoc=%7BD0DEC1F5-08CD-4298-B201-61B4BB323284%7D&file=quarterbacks.csv&action=default&mobileredirect=true') | 

**Data Dictionary** -
| Name | Data Type | Description | Example |
|------|-----------|-------------|---------|
| game_id | String | Unique game identifier (season_week_away_home) | 2022_01_BUF_LAR |
| season | Integer | NFL season year | 2022 |
| game_type | String | Season phase (REG = regular season) | REG |
| week | Integer | Week number within season | 7 |
| gameday | String | Date of game (YYYY-MM-DD) | 2022-10-16 |
| away_team | String | Away team abbreviation | KC |
| home_team | String | Home team abbreviation | LV |
| away_score | Integer | Final points scored by away team | 24 |
| home_score | Integer | Final points scored by home team | 30 |
| result | Integer | Target variable: home_score minus away_score | 6 |
| spread_line | Float | Vegas opening spread (positive = home favored) | -3.5 |
| total_line | Float | Vegas over/under total points line | 48.5 |
| location | String | Whether game is at home or neutral site | Home |
| away_rest | Integer | Days since away team's last game | 7 |
| home_rest | Integer | Days since home team's last game | 10 |
| away_moneyline | Integer | Moneyline odds for away team win | +145 |
| home_moneyline | Integer | Moneyline odds for home team win | -165 |
| roof | String | Stadium roof type | outdoors |
| surface | String | Playing surface type | grass |
| temp | Float | Temperature at kickoff in degrees F (outdoors/open only) | 52.0 |
| wind | Float | Wind speed at kickoff in mph | 12.0 |
| div_game | Integer | 1 if teams are in same division, else 0 | 1 |
| overtime | Integer | 1 if game went to overtime, else 0 | 0 |
| away_qb_id | String | nflverse player ID for starting away QB | 00-0033873 |
| home_qb_id | String | nflverse player ID for starting home QB | 00-0036355 |

**Data Dictionary** -

| Feature | Approximate Min | Approximate Max | Uncertainty Notes |
|---------|-----------------|-----------------|-------------------|
| result | -50 | 50 | Exact official final score, no uncertainty |
| away_score | 0 | 60 | Exact official final score, no uncertainty |
| home_score | 0 | 60 | Exact official final score, no uncertainty |
| spread_line | -16.5 | 16.5 | Opening line only — moves before kickoff; closing line is more predictive |
| total_line | 35.0 | 60.0 | Same timing uncertainty as spread_line |
| away_rest | 4 | 14 | Exact calendar days, no uncertainty |
| home_rest | 4 | 14 | Exact calendar days, no uncertainty |
| temp | 10.0 | 100.0 | Recorded near stadium — may not reflect exact on-field conditions |
| wind | 0.0 | 40.0 | Sustained speed only — gust vs. sustained distinction not captured |
| away_moneyline | varies | varies | Market snapshot; same timing uncertainty as spread_line |
| home_moneyline | varies | varies | Market snapshot; same timing uncertainty as spread_line |


