# PitchPulse

A player performance dashboard inspired by the World Cup. Search players, compare them head to head, explore market value trends, and cluster the league into playing-style archetypes, all backed by real Transfermarkt data.

Dedicated to Cape Verde's second goal against Argentina, the only match I watched start to finish. ty to haris. 

Built with Python, Streamlit, Plotly, and scikit-learn.

## Features

- **Dashboard** — headline numbers at a glance: players tracked, average market value, top scorer goals, clubs covered.
- **Player Search** — type-ahead search over the full player pool, with an animated stat card for whoever you select.
- **Compare Players** — head-to-head radar chart plus side-by-side stat cards for any two players.
- **League Overview** — top scorers, market value distribution by position, and age vs. value across the league.
- **Role Archetypes** — KMeans clustering (adjustable K) on scaled per-90 stats and market value, projected to 2D with PCA, with clusters auto-labeled by their own stat profile (Elite Attackers, Defensive Anchors, Creative Playmakers, and so on).
- **Explore Clusters** — browse which players fall into each archetype, with a summary of that archetype's average stats.

Stat bars and the head-to-head radar are both driven by **percentile rank** rather than raw values divided by a column max, so a single outlier player (for example, someone with one goal in five minutes played) can't quietly distort every other player's bar on that stat.

## Getting started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Using a virtual environment is recommended, so this doesn't touch any system-level pandas/numpy install.

### 2. Run the app

```bash
streamlit run app.py
```

This opens the dashboard at `http://localhost:8501`.

## Data source

By default, PitchPulse runs on a **synthetic sample of 350 players** with a realistic schema, so it works immediately with no setup.

To use real Transfermarkt data (via the [`player-scores`](https://www.kaggle.com/datasets/davidcariboo/player-scores) dataset on Kaggle) instead:

1. Go to [kaggle.com/settings](https://www.kaggle.com/settings) → **API** → *Create New Token*, which downloads a `kaggle.json` file.
2. Make those credentials available to the app, either:
   - Place `kaggle.json` at `~/.kaggle/kaggle.json` (the standard Kaggle CLI location), or
   - If deploying (for example on Streamlit Community Cloud), add them as secrets in `.streamlit/secrets.toml`:
     ```toml
     KAGGLE_USERNAME = "your_username"
     KAGGLE_KEY = "your_key"
     ```
3. Run the app. It authenticates via `kagglehub` and downloads the dataset automatically. If that fails for any reason (no credentials, no internet, dataset changed), it falls back to the synthetic sample and shows why in the sidebar.

**Note:** the free Transfermarkt dataset doesn't include per-touch stats like pass accuracy, dribbles, or tackles. Those three fields are simulated on top of real players for visualization purposes only. Everything else (goals, assists, minutes played, market value, position, club, nationality) is real when credentials are supplied.

## Project structure

```
streamlit_app/
├── app.py             # the full dashboard
├── football.png       # browser tab icon
├── requirements.txt   # Python dependencies
└── README.md
```

## Tech stack

- **Streamlit** for the app and UI
- **pandas / numpy** for data wrangling
- **Plotly** for interactive charts (bar, violin, scatter, radar)
- **scikit-learn** for KMeans clustering and PCA
- **kagglehub** for pulling the live dataset

## Notes and limitations

- Clustering results (and therefore the archetype labels) will shift slightly depending on which K you choose and whether you're on real or sample data.
- The UI is a personal project, built to learn more about data visualization along the way, not a production analytics tool.
