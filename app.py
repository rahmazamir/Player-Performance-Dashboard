"""
Player Performance Dashboard — Streamlit Edition
2026 World Cup / Transfermarkt data explorer.

Run locally:
    pip install -r requirements.txt
    streamlit run app.py
"""

import json
import os
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
import os as _os

_FOOTBALL_ICON_PATH = _os.path.join(_os.path.dirname(__file__), "football.png")
_PAGE_ICON = _FOOTBALL_ICON_PATH if _os.path.exists(_FOOTBALL_ICON_PATH) else "\u26bd"

st.set_page_config(
    page_title="PitchPulse",
    page_icon=_PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = {
    "amber": "#F3A400",
    "crimson": "#D11547",
    "blue": "#274192",
    "ink": "#302A40",
    "white": "#FAF7FC",
}
CLUSTER_COLORS = ["#F3A400", "#D11547", "#274192", "#7B5EA7", "#1FA189"]

POSITION_COLORS = {
    "Attack": PALETTE["amber"],
    "Midfield": PALETTE["crimson"],
    "Defender": PALETTE["blue"],
    "Goalkeeper": PALETTE["ink"],
    "Missing": "#9A93A8",
}

# ----------------------------------------------------------------------------
# GLOBAL CSS — bento-box layout, palette, fonts, micro-interactions
# ----------------------------------------------------------------------------
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Sora:wght@500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
h1, h2, h3, .pd-title { font-family: 'Sora', sans-serif !important; letter-spacing: -0.01em; }

/* --- App shell --- */
.stApp {
  background: #EEF0F3;
}
.block-container {
  padding-top: 2.2rem;
  max-width: 1250px;
}

/* --- Sidebar --- */
section[data-testid="stSidebar"] {
  background: #21212A;
  border-right: 1px solid rgba(255,255,255,0.05);
}
section[data-testid="stSidebar"][aria-expanded="true"] {
  min-width: 300px !important;
  width: 300px !important;
}
section[data-testid="stSidebar"][aria-expanded="true"] > div {
  width: 300px !important;
}
/* When collapsed, shrink all the way down so there's no leftover empty
   gap — Streamlit's native narrow rail + reopen arrow takes over. */
section[data-testid="stSidebar"][aria-expanded="false"] {
  min-width: 0 !important;
  width: 0 !important;
}
section[data-testid="stSidebar"][aria-expanded="false"] > div {
  width: 0 !important;
}
section[data-testid="stSidebar"] * { color: #E8E6EF !important; }
section[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.08); }

/* Shrink the sidebar's default inner side-padding so nav buttons span more
   of the (now narrower) sidebar width. */
section[data-testid="stSidebar"] div[data-testid="stSidebarUserContent"] {
  padding-left: 1rem !important;
  padding-right: 0rem !important;
}

/* Space between each nav icon and its label text */
section[data-testid="stSidebar"] div[role="radiogroup"] label [data-testid="stIconMaterial"] {
  margin-right: 45px;
}

.pd-brand {
  font-family: 'Sora', sans-serif;
  font-size: 19px;
  font-weight: 700;
  color: #FAF7FC;
  margin-bottom: 2px;
}
.pd-brand-sub {
  font-size: 12.5px;
  color: #9A93A8;
  margin-bottom: 4px;
}
.pd-section-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #7E7A8C;
  margin: 4px 0 10px 0;
}

/* Sidebar nav rectangles (built from st.radio, decorative dot removed) */
section[data-testid="stSidebar"] div[role="radiogroup"] {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label {
  background: rgba(255,255,255,0.035);
  border-radius: 10px;
  padding: 0 20px;
  height: 46px;
  width: calc(100% + 20px) !important;
  gap: 40px;
  display: flex;
  align-items: center;
  cursor: pointer;
  border: 1px solid transparent;
  transition: background 0.18s ease, transform 0.15s ease, border-color 0.18s ease, color 0.18s ease;
  font-weight: 500;
  font-size: 15px;
  box-sizing: border-box;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
  background: rgba(255,255,255,0.08);
  transform: translateX(3px);
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
  background: #F3A400;
  border-color: #F3A400;
  font-weight: 600;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked),
section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) * {
  color: #241F2E !important;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
  display: none;
}
section[data-testid="stSidebar"] input[type="radio"] {
  opacity: 0;
  width: 0;
  height: 0;
  position: absolute;
}

/* --- Keyframes --- */
@keyframes pd-fadeUp { from {opacity:0; transform: translateY(12px);} to {opacity:1; transform: translateY(0);} }
@keyframes pd-fadeIn { from {opacity:0;} to {opacity:1;} }
@keyframes pd-grow   { from {width:0%;} to {width: var(--target-width);} }

/* --- Bento boxes: every bordered container becomes a card ---
   Targeted via the `key` argument passed to st.container(), which Streamlit
   turns into a stable `st-key-<key>` class — more reliable across Streamlit
   versions than internal data-testid names. Each box's accent color is baked
   into its key (amber/crimson/blue) so the hover color is deterministic,
   rather than relying on CSS sibling position. */
div[class*="st-key-bento-"],
div[class*="st-key-bento-"] > div {
  background-color: #FFFFFF !important;
}
div[class*="st-key-bento-"] {
  border-radius: 18px;
  border: 1px solid rgba(48,42,64,0.10) !important;
  box-shadow: inset 0 0 0 1px rgba(48,42,64,0.035);
  padding: 22px 24px !important;
  transition: border-color 0.2s ease, border-width 0.2s ease, transform 0.22s ease;
  animation: pd-fadeUp 0.45s ease-out;
  overflow: hidden;
}
div[class*="st-key-bento-"]:hover {
  transform: translateY(-3px);
}
div[class*="st-key-bento-amber-"]:hover {
  border: 3.5px solid #F3A400 !important;
}
div[class*="st-key-bento-crimson-"]:hover {
  border: 3.5px solid #D11547 !important;
}
div[class*="st-key-bento-blue-"]:hover {
  border: 3.5px solid #274192 !important;
}

/* --- Player / stat cards (inline HTML) --- */
.pd-card {
  padding: 10px 6px 4px 6px;
}
.pd-badge {
  display: inline-block; padding: 4px 12px; border-radius: 999px;
  font-size: 11.5px; font-weight: 600; color: white; letter-spacing: 0.02em;
  animation: pd-fadeIn 0.5s ease-out;
}
.pd-tag {
  display: inline-block; background: #F1EEF7; color: #302A40; font-size: 11px;
  font-weight: 600; padding: 3px 10px; border-radius: 8px; margin-left: 8px;
}
.pd-name { font-family: 'Sora', sans-serif; font-size: 25px; font-weight: 700; color: #302A40; margin: 8px 0 2px 0; }
.pd-sub  { color: #837D93; font-size: 13.5px; margin-bottom: 14px; }

.pd-statrow { margin: 10px 0; animation: pd-fadeUp 0.55s ease-out; }
.pd-statlabel { display:flex; justify-content:space-between; font-size:12px; font-weight:600; color:#302A40; margin-bottom:4px; }
.pd-track { background:#EDE9F3; border-radius:8px; height:9px; overflow:hidden; }
.pd-fill  { height:100%; border-radius:8px; animation: pd-grow 1s cubic-bezier(.2,.8,.2,1) forwards;
            background: linear-gradient(90deg, var(--bar-color-a), var(--bar-color-b)); }

.pd-value-chip {
  display:inline-flex; align-items:center; gap:6px; background:#302A40; color:#FAF7FC;
  font-weight:600; padding:7px 16px; border-radius:10px; font-size:14px;
  transition: transform 0.2s ease;
}
.pd-value-label { font-size: 11px; color: #9A93A8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }

.pd-cluster-chip { display:inline-block; padding:3px 10px; border-radius:999px; font-size:11px; font-weight:600; color:white; margin-right:6px; }

/* --- Player table (Explore Clusters) with per-cell hover --- */
.pd-table { width: 100%; border-collapse: collapse; font-size: 13.5px; }
.pd-table th {
  text-align: left; padding: 10px 12px; color: #837D93; font-weight: 600;
  font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;
  border-bottom: 1px solid rgba(48,42,64,0.08);
}
.pd-table td {
  padding: 10px 12px; color: #302A40;
  border-bottom: 1px solid rgba(48,42,64,0.05);
  transition: background 0.15s ease, color 0.15s ease;
}
.pd-table tbody tr:hover td { background: #FBF3E4; }
.pd-table td:nth-child(1):hover {
  background: #F3A400 !important;
  color: #241F2E;
  font-weight: 600;
}
.pd-table td:nth-child(2):hover {
  background: #D11547 !important;
  color: #FAF7FC;
  font-weight: 600;
}
.pd-table td:nth-child(3):hover {
  background: #274192 !important;
  color: #FAF7FC;
  font-weight: 600;
}
.pd-table td:nth-child(4):hover {
  background: #302A40 !important;
  color: #FAF7FC;
  font-weight: 600;
}

/* --- Buttons / inputs micro-interactions --- */
div.stButton > button, div.stDownloadButton > button {
  background: linear-gradient(90deg, #F3A400, #D11547);
  color: white; border: none; border-radius: 10px; font-weight: 600;
  padding: 0.5em 1.4em; transition: transform 0.18s ease, box-shadow 0.18s ease;
}
div.stButton > button:hover, div.stDownloadButton > button:hover {
  transform: translateY(-2px) scale(1.02);
  box-shadow: 0 8px 18px rgba(209,21,71,0.3);
}
div[data-baseweb="select"] > div, .stTextInput input {
  border-radius: 10px !important;
  transition: border-color 0.18s ease, box-shadow 0.18s ease;
}
div[data-baseweb="select"] > div:hover, .stTextInput input:hover {
  border-color: #F3A400 !important;
}

/* --- Metric widgets --- */
div[data-testid="stMetric"] { padding: 4px 2px; }
div[data-testid="stMetricValue"] { font-family: 'Sora', sans-serif; color: #302A40; }
div[data-testid="stMetricLabel"] { color: #837D93; }

hr { margin: 0.6rem 0; }

/* --- Cluster-count slider: one solid blue, not part red/part blue ---
   Streamlit's default red theme shows up as inline styles containing
   "rgb(255...)" wherever it's used (filled track, thumb, value bubble) —
   whatever the exact property name/spacing, matching on that substring
   catches it everywhere and replaces it with a flat blue instead of a
   red-to-blue gradient. */
div[class*="st-key-cluster_k_slider"] [style*="rgb(255"] {
  background: #274192 !important;
  background-color: #274192 !important;
  border-color: #274192 !important;
}
div[class*="st-key-cluster_k_slider"] div[role="slider"] {
  background-color: #274192 !important;
  border-color: #274192 !important;
}
div[class*="st-key-cluster_k_slider"] div[role="slider"] > div {
  background-color: #274192 !important;
}
</style>
"""
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# HTML RENDER HELPERS
# ----------------------------------------------------------------------------
def stat_bar_html(label, value, pct, color_a, color_b, delay=0.0):
    pct = 0 if pd.isna(pct) else max(0, min(100, round(pct, 1)))
    return f"""
    <div class="pd-statrow" style="animation-delay:{delay}s">
      <div class="pd-statlabel"><span>{label}</span><span>{value}</span></div>
      <div class="pd-track">
        <div class="pd-fill" style="--target-width:{pct}%; --bar-color-a:{color_a}; --bar-color-b:{color_b};"></div>
      </div>
    </div>"""


def format_value(v):
    if pd.isna(v) or v == 0:
        return "\u20ac0"
    if v >= 1_000_000:
        return f"\u20ac{v/1_000_000:.1f}M"
    return f"\u20ac{v/1_000:.0f}K"


_bento_counter = {"n": 0}
_BENTO_ACCENTS = ["amber", "crimson", "blue"]

def bento():
    """A bordered container with a stable, unique key so our CSS can reliably
    target it (Streamlit's internal DOM structure/testids can change between
    versions, but the `key` argument generates a fixed `st-key-<key>` class).
    Each box is also tagged with a deterministic accent color (amber, crimson,
    blue, repeating) baked into the key itself, since CSS :nth-of-type can't
    reliably count across Streamlit's nested column/container DOM."""
    _bento_counter["n"] += 1
    accent = _BENTO_ACCENTS[(_bento_counter["n"] - 1) % 3]
    return st.container(border=True, key=f"bento-{accent}-{_bento_counter['n']}")


def player_card_html(row, stat_defs, color_a=None, color_b=None):
    color = POSITION_COLORS.get(row["position"], PALETTE["ink"])
    bar_color_a = color_a or PALETTE["amber"]
    bar_color_b = color_b or PALETTE["crimson"]
    bars = "".join(
        stat_bar_html(label, row[col], row[pct_col], bar_color_a, bar_color_b, delay=i * 0.06)
        for i, (label, col, pct_col) in enumerate(stat_defs)
    )
    return f"""
    <div class="pd-card">
      <span class="pd-badge" style="background:{color}">{row['position']} &middot; {row.get('sub_position','')}</span>
      <span class="pd-tag">{row.get('nationality','—')}</span>
      <div class="pd-name">{row['name']}</div>
      <div class="pd-sub">{row.get('club','—')} &nbsp;&middot;&nbsp; Age {int(row['age'])} &nbsp;&middot;&nbsp; {row.get('height','—')} cm &nbsp;&middot;&nbsp; {int(row['appearances'])} apps</div>
      <div class="pd-value-label">Market value</div>
      <div class="pd-value-chip">{format_value(row['market_value'])}</div>
      <div style="margin-top:16px;">{bars}</div>
    </div>"""


def player_table_html(subset):
    rows_html = "".join(
        f"<tr><td>{r['name']}</td><td>{r['position']}</td><td>{r.get('club','—')}</td>"
        f"<td>{format_value(r['market_value'])}</td></tr>"
        for _, r in subset.iterrows()
    )
    return f"""
    <table class="pd-table">
      <thead><tr><th>Name</th><th>Position</th><th>Club</th><th>Market value</th></tr></thead>
      <tbody>{rows_html}</tbody>
    </table>"""


# ----------------------------------------------------------------------------
# DATA LOADING (real Kaggle data via kagglehub, else synthetic fallback)
# — unchanged from the previous version —
# ----------------------------------------------------------------------------
def generate_sample_data(n=350, seed=42):
    rng = np.random.default_rng(seed)
    first = ["Lucas","Mateo","Kenji","Liam","Youssef","Diego","Ivan","Noah","Marco","Kwame",
             "Elias","Ryo","Omar","Bruno","Sami","Theo","Aleksander","Rafael","Kofi","Milan",
             "Tomas","Andres","Jamal","Felix","Hiroshi","Carlos","Dario","Amir","Jonas","Pablo"]
    last = ["Silva","Nakamura","Andersson","Diallo","Fernandez","Kowalski","Rossi","Okafor",
            "Martinez","Novak","Costa","Yamada","Berisha","Traore","Petrov","Garcia",
            "Larsson","Haidari","Moreira","Popescu","Nilsson","Ferreira","Osei","Vidal"]
    clubs = ["Real Sotano","FC Nordheim","Atletico Marea","Berg United","Porto Vermelho",
             "AS Lumiere","Kaze FC","River Plata Sur","Continental City","Estrella Roja",
             "Nordic Athletic","Sahara Kings","Bahia Stars","Highland Rovers","Zenith SC"]
    nationalities = ["Brazil","Japan","Sweden","Mali","Argentina","Poland","Italy","Nigeria",
                      "Spain","Croatia","Portugal","Senegal","Serbia","France","Morocco",
                      "Ghana","Netherlands","Colombia","Iran","England"]
    positions = {
        "Goalkeeper": ["Goalkeeper"],
        "Defender":   ["Centre-Back", "Left-Back", "Right-Back"],
        "Midfield":   ["Defensive Midfield", "Central Midfield", "Attacking Midfield"],
        "Attack":     ["Left Winger", "Right Winger", "Centre-Forward"],
    }
    pos_choices = rng.choice(list(positions.keys()), size=n, p=[0.10, 0.30, 0.32, 0.28])

    rows = []
    for i in range(n):
        pos = pos_choices[i]
        sub = rng.choice(positions[pos])
        age = int(rng.integers(18, 36))
        minutes = int(rng.integers(300, 3400))
        apps_ct = max(3, int(minutes / rng.integers(55, 90)))

        if pos == "Attack":
            goals, assists = int(rng.poisson(0.55*apps_ct)), int(rng.poisson(0.30*apps_ct))
            tackles, dribbles, pass_acc = int(rng.integers(0,30)), int(rng.integers(30,160)), rng.normal(78,7)
        elif pos == "Midfield":
            goals, assists = int(rng.poisson(0.18*apps_ct)), int(rng.poisson(0.28*apps_ct))
            tackles, dribbles, pass_acc = int(rng.integers(20,110)), int(rng.integers(20,110)), rng.normal(85,6)
        elif pos == "Defender":
            goals, assists = int(rng.poisson(0.04*apps_ct)), int(rng.poisson(0.06*apps_ct))
            tackles, dribbles, pass_acc = int(rng.integers(40,160)), int(rng.integers(5,45)), rng.normal(84,6)
        else:
            goals, assists = 0, int(rng.poisson(0.02*apps_ct))
            tackles, dribbles, pass_acc = int(rng.integers(0,5)), int(rng.integers(0,5)), rng.normal(72,8)

        base_value = {"Goalkeeper":6e6,"Defender":10e6,"Midfield":14e6,"Attack":18e6}[pos]
        skill = goals*3 + assists*2 + pass_acc/10 - age*0.4
        market_value = max(0.3e6, base_value + skill*6e5 + rng.normal(0,4e6))

        rows.append({
            "player_id": i+1, "name": f"{rng.choice(first)} {rng.choice(last)}",
            "position": pos, "sub_position": sub, "nationality": rng.choice(nationalities),
            "club": rng.choice(clubs), "height": int(rng.normal(183 if pos!="Goalkeeper" else 190, 6)),
            "age": age, "goals": goals, "assists": assists, "minutes_played": minutes,
            "appearances": apps_ct, "yellow_cards": int(rng.poisson(0.15*apps_ct)),
            "red_cards": int(rng.poisson(0.01*apps_ct)), "market_value": round(market_value, -3),
            "pass_accuracy": round(float(np.clip(pass_acc,45,99)),1),
            "dribbles": dribbles, "tackles": tackles,
        })

    out = pd.DataFrame(rows)
    dupe = out["name"].duplicated(keep=False)
    if dupe.any():
        initials = out.loc[dupe, "club"].str.split().str[0].str[:3].str.upper()
        out.loc[dupe, "name"] = out.loc[dupe, "name"] + " (" + initials + ")"
        still = out["name"].duplicated(keep=False)
        out.loc[still, "name"] = out.loc[still, "name"] + " #" + out.loc[still, "player_id"].astype(str)
    return out


def load_real_data(players_raw, appearances_raw, valuations_raw, seed=7):
    rng = np.random.default_rng(seed)
    players, apps, vals = players_raw.copy(), appearances_raw.copy(), valuations_raw.copy()

    agg = apps.groupby("player_id").agg(
        goals=("goals","sum"), assists=("assists","sum"),
        minutes_played=("minutes_played","sum"), appearances=("player_id","count"),
        yellow_cards=("yellow_cards","sum"), red_cards=("red_cards","sum"),
    ).reset_index()

    latest_val = (vals.sort_values("date").groupby("player_id")["market_value_in_eur"]
                  .last().reset_index().rename(columns={"market_value_in_eur": "market_value"}))

    df = players.merge(agg, on="player_id", how="left").merge(latest_val, on="player_id", how="left")
    df = df.rename(columns={
        "country_of_citizenship": "nationality", "current_club_name": "club",
        "height_in_cm": "height",
    })
    keep = ["player_id","name","position","sub_position","nationality","club","height",
            "date_of_birth","goals","assists","minutes_played","appearances",
            "yellow_cards","red_cards","market_value"]
    for c in keep:
        if c not in df.columns:
            df[c] = np.nan
    df = df[keep].dropna(subset=["name"])
    df["position"] = df["position"].fillna("Missing")

    today = pd.Timestamp("2026-01-01")
    df["age"] = ((today - pd.to_datetime(df["date_of_birth"], errors="coerce")).dt.days // 365).fillna(25).astype(int)

    # Transfermarkt's free dataset has no per-touch stats (passes, dribbles, tackles) —
    # these three are simulated for visualization purposes only.
    df["pass_accuracy"] = rng.normal(80, 8, len(df)).clip(45, 99).round(1)
    df["dribbles"] = rng.integers(5, 120, len(df))
    df["tackles"] = rng.integers(0, 140, len(df))

    df = df.fillna({
        "goals": 0, "assists": 0, "minutes_played": 1, "appearances": 1,
        "market_value": df["market_value"].median() if df["market_value"].notna().any() else 2_000_000,
    })
    return df.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_data():
    """Try kagglehub (using credentials from ~/.kaggle/kaggle.json, env vars,
    or st.secrets); fall back to synthetic sample data."""
    try:
        if "KAGGLE_USERNAME" in st.secrets and "KAGGLE_KEY" in st.secrets:
            os.environ["KAGGLE_USERNAME"] = st.secrets["KAGGLE_USERNAME"]
            os.environ["KAGGLE_KEY"] = st.secrets["KAGGLE_KEY"]
    except Exception:
        pass  # no secrets.toml configured locally — that's fine

    try:
        import kagglehub

        dataset_path = kagglehub.dataset_download("davidcariboo/player-scores")
        players_raw = pd.read_csv(os.path.join(dataset_path, "players.csv"))
        appearances_raw = pd.read_csv(os.path.join(dataset_path, "appearances.csv"))
        valuations_raw = pd.read_csv(os.path.join(dataset_path, "player_valuations.csv"))
        df = load_real_data(players_raw, appearances_raw, valuations_raw)
        return df, True, None
    except Exception as e:
        return generate_sample_data(), False, str(e)


def engineer_features(df):
    df = df.copy()
    df["minutes_played"] = df["minutes_played"].replace(0, 1)
    df["goals_per_90"] = (df["goals"] / df["minutes_played"] * 90).round(2)
    df["assists_per_90"] = (df["assists"] / df["minutes_played"] * 90).round(2)
    df["tackles_per_90"] = (df["tackles"] / df["minutes_played"] * 90).round(2)
    df["dribbles_per_90"] = (df["dribbles"] / df["minutes_played"] * 90).round(2)
    # Aggregating real data with .sum()/.fillna() promotes these to float64
    # (e.g. displaying as "458.0" instead of "458") even though they're
    # always whole numbers — cast back to int for clean display.
    for col in ["goals", "assists", "minutes_played", "appearances", "dribbles", "tackles"]:
        if col in df.columns:
            df[col] = df[col].round().astype(int)
    return df


# ----------------------------------------------------------------------------
# SIDEBAR — data source + navigation
# ----------------------------------------------------------------------------
NAV_ITEMS = [
    ("dashboard", "Dashboard"),
    ("search", "Player Search"),
    ("people", "Compare Players"),
    ("leaderboard", "League Overview"),
    ("bubble_chart", "Role Archetypes"),
    ("grid_view", "Explore Clusters"),
]
_NBSP = "\u00A0" * 6
NAV_OPTIONS = [f":material/{icon}: {'\u00A0' * 6}{label}" for icon, label in NAV_ITEMS]

with st.sidebar:
    st.markdown("<div class='pd-brand'>PitchPulse</div>", unsafe_allow_html=True)
    st.markdown("<div class='pd-brand-sub'>Football intelligence powered by data</div>", unsafe_allow_html=True)
    st.markdown("---")

    page_choice = st.radio(
        "Navigate",
        NAV_OPTIONS,
        label_visibility="collapsed",
    )
    # Strip the ":material/xxx: " icon shortcode back off so the rest of the
    # app can keep comparing against plain page names like "Dashboard".
    page = page_choice.split(": ", 1)[1] if ": " in page_choice else page_choice
    page = page.replace("\u00A0", "").strip()

    st.markdown("---")
    st.markdown("<div class='pd-section-label'>Data source</div>", unsafe_allow_html=True)


with st.spinner("Loading player data..."):
    df_raw, used_real_data, load_error = load_data()
    df = engineer_features(df_raw)

with st.sidebar:
    if used_real_data:
        st.markdown(
            f"""
            <div style='font-size:13px; color:#B7F0C9;'>
                <strong>Connected</strong><br>
                {len(df)} players synced from Transfermarkt
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        ...
        st.markdown(
            f"<div style='font-size:13px; color:#9A93A8;'>Using a modeled sample of {len(df)} players. "
            f"Live data syncs automatically once Kaggle credentials are found in "
            f"<code>~/.kaggle/kaggle.json</code>.</div>",
            unsafe_allow_html=True,
        )
        if load_error:
            with st.expander("Connection details"):
                st.code(load_error)


# ----------------------------------------------------------------------------
# GLOBAL HEADING — plain text, shown on every page, not inside a bento box
# ----------------------------------------------------------------------------
st.markdown(
    "<h1 class='pd-title' style='color:#302A40; margin:0 0 2px 0;'>Player Performance Dashboard</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div style='color:#837D93; margin-bottom:20px;'>Search individual players, compare them head-to-head, "
    "and surface playing-style archetypes across the squad.</div>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# SHARED STAT DEFINITIONS (used across multiple pages)
# ----------------------------------------------------------------------------
# Stat bars are driven by each player's PERCENTILE RANK for that stat across
# the whole player pool, not a raw value / raw max. Dividing by the raw max
# made the bars misleading: a single outlier (e.g. a player with 1 goal in a
# handful of minutes) could set an inflated max, making every other player's
# bar look short even when their number was genuinely excellent. Percentile
# rank fixes this the same way we already fixed the head-to-head radar chart.
_PCT_STAT_COLS = ["goals", "assists", "pass_accuracy", "dribbles", "tackles", "minutes_played"]
for _col in _PCT_STAT_COLS:
    df[f"{_col}_pctile"] = (df[_col].rank(pct=True) * 100).round(1)

STAT_DEFS = [
    ("Goals", "goals", "goals_pctile"),
    ("Assists", "assists", "assists_pctile"),
    ("Pass Accuracy", "pass_accuracy", "pass_accuracy_pctile"),
    ("Dribbles", "dribbles", "dribbles_pctile"),
    ("Tackles", "tackles", "tackles_pctile"),
    ("Minutes Played", "minutes_played", "minutes_played_pctile"),
]

# ----------------------------------------------------------------------------
# PAGE: DASHBOARD — headline metrics only (heading above is shared by all pages)
# ----------------------------------------------------------------------------
if page == "Dashboard":
    m1, m2, m3, m4 = st.columns([1.2, 1, 1, 0.9])
    with m1:
        with bento():
            st.metric("Players tracked", f"{len(df):,}")
    with m2:
        with bento():
            st.metric("Avg. market value", format_value(df["market_value"].mean()))
    with m3:
        with bento():
            st.metric("Top scorer goals", int(df["goals"].max()))
    with m4:
        with bento():
            st.metric("Clubs covered", df["club"].nunique() if "club" in df else "—")

# ----------------------------------------------------------------------------
# PAGE: SEARCH
# ----------------------------------------------------------------------------
elif page == "Player Search":
    col_search, col_card = st.columns([1, 1.6])

    with col_search:
        with bento():
            st.markdown("<div class='pd-section-label'>Find a player</div>", unsafe_allow_html=True)
            query = st.text_input("Search by name", placeholder="Type a player name...", label_visibility="collapsed")
            names = sorted(df["name"].tolist())
            if query.strip():
                names = sorted(df[df["name"].str.lower().str.contains(query.strip().lower())]["name"].tolist())

            if not names:
                st.warning("No players match that search.")
                selected = None
            else:
                default_name = "Erling Haaland"
                default_idx = names.index(default_name) if (not query.strip() and default_name in names) else 0
                selected = st.selectbox("Select a player", names, index=default_idx)

    with col_card:
        with bento():
            if selected:
                row = df[df["name"] == selected].iloc[0]
                st.markdown(player_card_html(row, STAT_DEFS), unsafe_allow_html=True)
                st.toast(f"Loaded {selected}'s profile")
            else:
                st.markdown("<div class='pd-card'><div class='pd-sub'>Select a player to see their profile.</div></div>", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# PAGE: COMPARE
# ----------------------------------------------------------------------------
elif page == "Compare Players":
    names = sorted(df["name"].tolist())

    with bento():
        st.markdown("<div class='pd-section-label'>Head-to-head comparison</div>", unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        default_a, default_b = "Nereo Champagne", "Lionel Messi"
        idx_a = names.index(default_a) if default_a in names else 0
        idx_b = names.index(default_b) if default_b in names else min(1, len(names)-1)
        name_a = col_a.selectbox("Player A", names, index=idx_a)
        name_b = col_b.selectbox("Player B", names, index=idx_b)

        RADAR_STATS = ["goals_per_90","assists_per_90","pass_accuracy","dribbles_per_90","tackles_per_90","minutes_played"]
        RADAR_LABELS = ["Goals/90","Assists/90","Pass Acc.","Dribbles/90","Tackles/90","Minutes"]

        # Percentile-rank each stat across the whole player pool instead of
        # dividing by the raw max. Per-90 rates explode for players with only
        # a handful of minutes played (e.g. 1 goal in 5 minutes = huge
        # goals_per_90), and using that as the denominator squashed every
        # normal player's bar down near zero on that axis — which is why
        # most radar comparisons looked identical. Percentile rank is immune
        # to that: it reflects where a player stands relative to everyone
        # else, so the chart stays readable and varied no matter who you pick.
        radar_percentiles = df[RADAR_STATS].rank(pct=True) * 100

        def normalized_row(name):
            idx = df.index[df["name"] == name][0]
            vals = [round(radar_percentiles.loc[idx, s], 1) for s in RADAR_STATS]
            return vals, df.loc[idx]

        vals_a, row_a = normalized_row(name_a)
        vals_b, row_b = normalized_row(name_b)

        hover_a = [f"{name_a}<br>{lbl}: {row_a[s]} (top {100-p:.0f}%)"
                   for lbl, s, p in zip(RADAR_LABELS, RADAR_STATS, vals_a)]
        hover_b = [f"{name_b}<br>{lbl}: {row_b[s]} (top {100-p:.0f}%)"
                   for lbl, s, p in zip(RADAR_LABELS, RADAR_STATS, vals_b)]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(r=vals_a+vals_a[:1], theta=RADAR_LABELS+RADAR_LABELS[:1], fill="toself",
                                       name=name_a, line=dict(color=PALETTE["crimson"], width=3),
                                       fillcolor="rgba(209,21,71,0.28)",
                                       text=hover_a+hover_a[:1], hoverinfo="text"))
        fig.add_trace(go.Scatterpolar(r=vals_b+vals_b[:1], theta=RADAR_LABELS+RADAR_LABELS[:1], fill="toself",
                                       name=name_b, line=dict(color=PALETTE["blue"], width=3),
                                       fillcolor="rgba(39,65,146,0.25)",
                                       text=hover_b+hover_b[:1], hoverinfo="text"))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,100], showticklabels=False)),
            showlegend=True, title=f"{name_a} vs {name_b}",
            font=dict(family="Inter"), title_font=dict(family="Sora", size=19, color=PALETTE["ink"]),
            paper_bgcolor="rgba(0,0,0,0)", transition=dict(duration=500, easing="cubic-in-out"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Each axis shows the player's percentile rank across the full player pool for that stat, not a raw value — so the shape stays informative regardless of outliers elsewhere in the data.")

    col_a, col_b = st.columns(2)
    with col_a:
        with bento():
            st.markdown(
                player_card_html(row_a, STAT_DEFS, color_a=PALETTE["crimson"], color_b=PALETTE["blue"]),
                unsafe_allow_html=True,
            )
    with col_b:
        with bento():
            st.markdown(
                player_card_html(row_b, STAT_DEFS, color_a=PALETTE["blue"], color_b=PALETTE["amber"]),
                unsafe_allow_html=True,
            )

# ----------------------------------------------------------------------------
# PAGE: LEAGUE OVERVIEW
# ----------------------------------------------------------------------------
elif page == "League Overview":
    with bento():
        st.markdown("<div class='pd-section-label'>Top scorers</div>", unsafe_allow_html=True)
        top_scorers = df.sort_values("goals", ascending=False).head(12).sort_values("goals")
        fig1 = px.bar(top_scorers, x="goals", y="name", orientation="h",
                      color_discrete_sequence=[PALETTE["crimson"]], title="Top 12 Goal Scorers",
                      labels={"goals":"Goals","name":""})
        fig1.update_traces(marker_line_width=0)
        fig1.update_layout(font=dict(family="Inter"), title_font=dict(family="Sora", size=19, color=PALETTE["ink"]),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            transition=dict(duration=500, easing="cubic-in-out"))
        st.plotly_chart(fig1, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        with bento():
            st.markdown("<div class='pd-section-label'>Market value by position</div>", unsafe_allow_html=True)
            # Values are log-transformed before plotting (not just the axis),
            # since market value is heavily right-skewed.
            plot_df = df[df["market_value"] > 0].copy()
            plot_df["market_value_log"] = np.log10(plot_df["market_value"])
            fig2 = px.violin(plot_df, x="position", y="market_value_log", color="position",
                              color_discrete_map=POSITION_COLORS, box=True, points=False,
                              title="Market Value Distribution by Position",
                              labels={"market_value_log": "Market Value (log scale)", "position": ""})
            tick_vals = [5.3, 5.7, 6, 6.3, 6.7, 7, 7.3, 7.7, 8, 8.3]
            tick_text = ["200K","500K","1M","2M","5M","10M","20M","50M","100M","200M"]
            fig2.update_yaxes(tickvals=tick_vals, ticktext=tick_text)
            fig2.update_layout(showlegend=False, font=dict(family="Inter"),
                                title_font=dict(family="Sora", size=17, color=PALETTE["ink"]),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        with bento():
            st.markdown("<div class='pd-section-label'>Age vs. value</div>", unsafe_allow_html=True)
            fig3 = px.scatter(df, x="age", y="market_value", color="position", size="goals",
                               color_discrete_map=POSITION_COLORS, hover_name="name",
                               title="Age vs Market Value (bubble size = goals)",
                               labels={"market_value":"Market Value (€)","age":"Age"})
            fig3.update_layout(font=dict(family="Inter"), title_font=dict(family="Sora", size=17, color=PALETTE["ink"]),
                                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig3, use_container_width=True)

# ----------------------------------------------------------------------------
# PAGE: CLUSTERING
# ----------------------------------------------------------------------------
elif page == "Role Archetypes":
    col_controls, col_scatter = st.columns([1, 2.2])

    CLUSTER_FEATURES = ["goals_per_90","assists_per_90","pass_accuracy","dribbles_per_90","tackles_per_90","market_value"]
    X = df[CLUSTER_FEATURES].fillna(0).copy()
    X_scaled = StandardScaler().fit_transform(X)

    with col_controls:
        with bento():
            st.markdown("<div class='pd-section-label'>Model controls</div>", unsafe_allow_html=True)
            st.caption("KMeans on scaled per-90 stats and market value, projected to 2D with PCA.")
            k = st.slider("Number of clusters", 2, 8, 4, key="cluster_k_slider")

    with st.spinner("Running clustering..."):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_scaled)
    df["cluster"] = kmeans.labels_

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    df["pca_x"], df["pca_y"] = coords[:,0], coords[:,1]

    cluster_profiles = df.groupby("cluster")[CLUSTER_FEATURES].mean()

    def label_cluster(row):
        if row["goals_per_90"] > cluster_profiles["goals_per_90"].median()*1.3:
            return "Elite Attackers"
        if row["tackles_per_90"] > cluster_profiles["tackles_per_90"].median()*1.3:
            return "Defensive Anchors"
        if row["assists_per_90"] > cluster_profiles["assists_per_90"].median()*1.2:
            return "Creative Playmakers"
        if row["market_value"] > cluster_profiles["market_value"].median()*1.4:
            return "High-Value Stars"
        return "Squad Players"

    label_map = {idx: label_cluster(r) for idx, r in cluster_profiles.iterrows()}
    df["cluster_label"] = df["cluster"].map(label_map)
    color_map = {lbl: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, lbl in enumerate(sorted(df["cluster_label"].unique()))}

    with col_scatter:
        with bento():
            fig = px.scatter(df, x="pca_x", y="pca_y", color="cluster_label", hover_name="name",
                              hover_data={"position": True, "goals": True, "assists": True,
                                          "pca_x": False, "pca_y": False},
                              color_discrete_map=color_map,
                              title=f"Player Archetypes — KMeans (K={k}) via PCA",
                              labels={"pca_x":"PCA 1","pca_y":"PCA 2","cluster_label":"Archetype"})
            fig.update_traces(marker=dict(size=10, opacity=0.85, line=dict(width=1, color="white")))
            fig.update_layout(font=dict(family="Inter"), title_font=dict(family="Sora", size=19, color=PALETTE["ink"]),
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               legend=dict(bgcolor="rgba(255,255,255,0.6)"))
            st.plotly_chart(fig, use_container_width=True)

    st.session_state["df_clustered"] = df
    st.session_state["color_map"] = color_map

    with bento():
        st.markdown("<div class='pd-section-label'>Archetype sizes</div>", unsafe_allow_html=True)
        st.bar_chart(df["cluster_label"].value_counts(), color=PALETTE["blue"])

# ----------------------------------------------------------------------------
# PAGE: CLUSTER EXPLORER
# ----------------------------------------------------------------------------
elif page == "Explore Clusters":
    if "df_clustered" not in st.session_state:
        with bento():
            st.warning("Run Role Archetypes first to generate clusters.")
    else:
        cdf = st.session_state["df_clustered"]
        color_map = st.session_state["color_map"]

        col_select, col_stats = st.columns([1, 1.4])

        with col_select:
            with bento():
                st.markdown("<div class='pd-section-label'>Choose an archetype</div>", unsafe_allow_html=True)
                archetype_options = sorted(cdf["cluster_label"].unique())
                default_archetype = "Elite Attackers"
                default_idx = archetype_options.index(default_archetype) if default_archetype in archetype_options else 0
                label = st.selectbox("Archetype", archetype_options, index=default_idx, label_visibility="collapsed")
                subset = cdf[cdf["cluster_label"] == label].sort_values("market_value", ascending=False)
                color = color_map.get(label, PALETTE["ink"])
                st.markdown(
                    f"<span class='pd-cluster-chip' style='background:{color}'>{label}</span> "
                    f"<b>{len(subset)} players</b>",
                    unsafe_allow_html=True,
                )

        with col_stats:
            with bento():
                st.markdown("<div class='pd-section-label'>Archetype profile</div>", unsafe_allow_html=True)
                avg_stats = subset[["goals_per_90","assists_per_90","pass_accuracy","dribbles_per_90","tackles_per_90"]].mean()
                archetype_stat_items = [
                    ("Avg Goals/90","goals_per_90"), ("Avg Assists/90","assists_per_90"),
                    ("Avg Pass Acc.","pass_accuracy"), ("Avg Dribbles/90","dribbles_per_90"),
                    ("Avg Tackles/90","tackles_per_90"),
                ]
                bars = "".join(
                    stat_bar_html(
                        lbl,
                        round(avg_stats[col_name], 2),
                        (df[col_name] < avg_stats[col_name]).mean() * 100,  # where this archetype's average sits among all players
                        color, PALETTE["ink"], delay=i * 0.07,
                    )
                    for i, (lbl, col_name) in enumerate(archetype_stat_items)
                )
                st.markdown(bars, unsafe_allow_html=True)

        with bento():
            st.markdown("<div class='pd-section-label'>Players in this archetype</div>", unsafe_allow_html=True)
            st.markdown(
                player_table_html(subset[["name", "position", "club", "market_value"]].head(20)),
                unsafe_allow_html=True,
            )
