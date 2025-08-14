import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# =========================================================
# PAGE CONFIG & STYLING
# =========================================================
st.set_page_config(layout="wide")

background = "#2c2f38"
text_color = "#f4f4f9"
actual_color = "#f08a5d"
xg_color = "#b83b5e"
gap_fill_color = "#f9ed69"
guide_line_color = "#e4e4e4"
accent3 = "#31748f" 

plt.rcParams.update({
    'axes.facecolor': background,
    'figure.facecolor': background,
    'axes.edgecolor': background,
    'axes.labelcolor': text_color,
    'text.color': text_color,
    'xtick.color': text_color,
    'ytick.color': text_color,
    'grid.color': guide_line_color,
    'font.size': 10,
    'axes.titlesize': 10,
    'axes.labelsize': 10,
})

# =========================================================
# DATA LOADING & PREPARATION
# =========================================================
CSV_PATH = "premier_league_clinicality_with_standings.csv"
data = pd.read_csv(CSV_PATH, encoding='utf-8')
teams = sorted(data['team'].unique())

team_totals = (
    data.groupby('team')[['goals', 'xg']]
    .sum()
    .reset_index()
)
team_totals['goal_diff'] = team_totals['goals'] - team_totals['xg']

final_positions = (
    data[data['week'] == 38][['team', 'position']]
    .rename(columns={'position': 'final_position'})
)

team_totals = (
    team_totals
    .merge(final_positions, on='team')
    .sort_values('final_position', ascending=True)
    .reset_index(drop=True)
)

# =========================================================
# SIDEBAR CONTROLS
# =========================================================

st.sidebar.markdown(
    """
    <hr style="margin-top: 0.5em; margin-bottom: 0.5em; border: 0; border-top: 1px solid #666;">
    <p style='font-size:14px; color:#ccc; text-align:center;'>
    Built by <strong>Kalungi Analytics</strong><br>
    <a href='https://www.linkedin.com/in/ben-sharpe-49659a207//' target='_blank' style='color:#4FA2B4;'>Connect on LinkedIn</a>
    </p>
    <hr style="margin-top: 0.5em; margin-bottom: 0.5em; border: 0; border-top: 1px solid #666;">
    """,
    unsafe_allow_html=True
)
st.sidebar.title("xG Adjustment Method")
csv_option = st.sidebar.radio(
    "Select xG Adjustment Method:",
    options=[
        "Adjustment A: xG Floor Correction",
        "Adjustment B: xG Threshold Correction (≥ 0.5)"
    ],
    index=0
)

if csv_option.startswith("Adjustment A"):
    st.sidebar.caption("Goals adjusted up to xG floor (no fractions).")
    CSV_PATH = "premier_league_clinicality_with_standings.csv"
else:
    st.sidebar.caption("Goals rounded up if xG exceeds by ≥ 0.5.")
    CSV_PATH = "premier_league_clinicality_with_standings_049.csv"

data = pd.read_csv(CSV_PATH, encoding='utf-8')
teams = sorted(data['team'].unique())

# =========================================================
# PLOT FUNCTIONS
# =========================================================
def plot_goals_vs_xg():
    fig, ax = plt.subplots(figsize=(10, 3))
    x = np.arange(len(team_totals))
    ax.plot(x, team_totals['goals'], label='Goals', marker='o', color=actual_color)
    ax.plot(x, team_totals['xg'], label='xG', marker='o', color=xg_color)
    ax.set_xticks(x)
    ax.set_xticklabels(team_totals['team'], rotation=45, ha='right')
    ax.set_ylabel("Total Goals / xG")
    ax.grid(True, linestyle='--', alpha=0.15)
    ax.legend(frameon=False, loc='upper right')
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig

def plot_clinicality():
    fig, ax = plt.subplots(figsize=(10, 3))
    x = np.arange(len(team_totals))
    ax.plot(x, team_totals['goal_diff'], label='Goals – xG',
            linestyle=':', linewidth=1.2, marker='o', alpha=1, color=xg_color)
    ax.set_xticks(x)
    ax.set_xticklabels(team_totals['team'], rotation=45, ha='right')
    ax.set_ylabel("Goals – xG")
    ax.axhline(0, linestyle='--', color='gray', alpha=0.3)
    ax.grid(True, linestyle='--', alpha=0.15)
    ax.legend(frameon=False, loc='upper right')
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig

def plot_cumulative_xg_diff_all_teams(df):
    fig, ax = plt.subplots(figsize=(6.5, 3))
    df = df.copy()
    df['xG_Diff'] = df['goals'] - df['xg']
    team_colors = [
        "#2A5F9E", "#B66323", "#2B682D", "#8A2E2E", "#7B5EA3",
        "#7E6B64", "#C18BCC", "#7E7E7E", "#B1B23E", "#4FA2B4",
        "#2B5F9E", "#B66323", "#2B682D", "#8A2E2E", "#7B5EA3",
        "#7E6B64", "#C18BCC", "#7E7E7E", "#B1B23E", "#4FA2B4"
    ]
    teams = list(df['team'].unique())
    for i, team in enumerate(teams):
        team_df = df[df['team'] == team].sort_values(by='week').copy()
        team_df['Cumulative xG_Diff'] = team_df['xG_Diff'].cumsum()
        color = team_colors[i % len(team_colors)]
        linestyle = '-' if i < 10 else '--'
        alpha = 1.0 if i < 10 else 0.5
        ax.plot(team_df['week'], team_df['Cumulative xG_Diff'],
                color=color, linewidth=0.8, alpha=alpha, linestyle=linestyle)
    ax.set_xlabel("Matchweek", fontsize=5)
    ax.set_ylabel("Cumulative Goals – xG", fontsize=5)
    min_diff = int(df['xG_Diff'].min())
    max_diff = int(df['xG_Diff'].max())
    for y in range(min_diff - 5, max_diff + 5, 5):
        ax.axhline(y=y, color=guide_line_color, linestyle='-', linewidth=0.3, alpha=0.05)
    ax.axhline(0, color=guide_line_color, linestyle='-', linewidth=0.5, alpha=0.3)
    ax.tick_params(axis='x', labelsize=5)
    ax.tick_params(axis='y', labelsize=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.legend(teams, frameon=False, fontsize=5, loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    return fig

# =========================================================
# HEADLINE STATISTICS DISPLAY
# =========================================================
a, b, c, d, e, f = 'Liverpool', 'Southampton', 'Chelsea', 'Bournemouth', 'Forest', 'Crystal Palace'

st.title("Premier League xG Headline Statistics")
st.subheader("Total Goals scored vs xG with no adjustments - Sorted by final ACTUAL league position")
st.markdown(
    f"<p style='font-size:18px;'>Premier League Champions <strong><span style='color:{actual_color};'>{a}</span></strong> "
    f"scored 60 more goals than the team that finished bottom, <strong><span style='color:{actual_color};'>{b}</span></strong>.</p>",
    unsafe_allow_html=True
)
st.pyplot(plot_goals_vs_xg())

st.subheader("Total variance on xG - Sorted by final ACTUAL league position")
st.markdown(
    f"<p style='font-size:18px;'>8 out of the top 10 teams exceeded their xG with only "
    f"<strong><span style='color:{actual_color};'>{c}</span></strong> & "
    f"<strong><span style='color:{actual_color};'>{d}</span></strong> failing to do so.</p>",
    unsafe_allow_html=True
)
st.pyplot(plot_clinicality())

st.subheader("Cumulative xG Difference Across All Teams")
st.markdown(
    f"<p style='font-size:18px;'>At opposite ends of the scale "
    f"<strong><span style='color:{actual_color};'>{e}</span></strong> exceeded xG by c.13 goals "
    f"whilst <strong><span style='color:{actual_color};'>{f}</span></strong> underperformed xG by c.9 goals.</p>",
    unsafe_allow_html=True
)
st.pyplot(plot_cumulative_xg_diff_all_teams(data))

# =========================================================
# TEAM SELECTION & DATA FILTERING
# =========================================================
st.markdown("---")
st.title("Premier League xG Clinicality Dashboard - Select Team from the sidebar")
st.sidebar.markdown("---")
st.sidebar.title("Select a Team")
selected_team = st.sidebar.selectbox("", teams)

df = data[data['team'] == selected_team].sort_values("week").reset_index(drop=True)
df['xg_points_gain'] = df['xg_points'] - df['points']
df['additional_goals'] = df['adj_goals'] - df['goals']

# =========================================================
# xG ADJUSTED POSITION CALCULATION
# =========================================================
all_teams_df = data.copy()
xg_positions = []

# Only replace goals when the adjusted value is GREATER than actual
def adjusted_goals_for_match(row, method):
    actual = int(row['goals'])
    if method.startswith("Adjustment A"):          # Floor method
        adj = int(np.floor(row['xg']))
    else:                                          # Adjustment B: round .5 up (no bankers rounding)
        adj = int(np.floor(row['xg'] + 0.5))
    return adj if adj > actual else actual

method_choice = csv_option

for week in sorted(df['week'].unique()):
    # cumulative trackers up to this week
    teams_points = {t: 0 for t in all_teams_df['team'].unique()}
    teams_gd = {t: 0 for t in all_teams_df['team'].unique()}

    processed_pairs = set()

    for w in range(1, week + 1):
        week_rows = all_teams_df[all_teams_df['week'] == w]

        for _, match in week_rows.iterrows():
            home_team = match['team']
            away_team = match['opponent']

            pair_key = (w, tuple(sorted([home_team, away_team])))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            # Find reverse row for the opponent
            opp_rows = week_rows[
                (week_rows['team'] == away_team) & (week_rows['opponent'] == home_team)
            ]
            if opp_rows.empty:
                # Fallback if reverse row isn’t present (shouldn’t happen in your CSVs)
                home_goals = int(match['goals'])
                away_goals = int(match['goals_conceded'])
                away_xg = away_goals  # no xG available; use actual as safe fallback
                opp_row = None
            else:
                opp_row = opp_rows.iloc[0]
                home_goals = int(match['goals'])
                away_goals = int(opp_row['goals'])

            # Adjust ONLY if the selected team is involved
            if home_team == selected_team:
                home_adj_goals = adjusted_goals_for_match(match, method_choice)
            else:
                home_adj_goals = home_goals

            if away_team == selected_team and opp_row is not None:
                # Use opponent's row (their own xG) when selected team is the away side
                away_adj_goals = adjusted_goals_for_match(opp_row, method_choice)
            else:
                away_adj_goals = away_goals

            # Determine points from the (possibly) adjusted scoreline
            if home_adj_goals > away_adj_goals:
                teams_points[home_team] += 3
            elif home_adj_goals == away_adj_goals:
                teams_points[home_team] += 1
                teams_points[away_team] += 1
            else:
                teams_points[away_team] += 3

            # Update GD with the adjusted tally (only differs if selected team involved)
            teams_gd[home_team] += home_adj_goals - away_adj_goals
            teams_gd[away_team] += away_adj_goals - home_adj_goals

    # Build table and rank
    standings = pd.DataFrame({
        'team': list(teams_points.keys()),
        'temp_points': list(teams_points.values()),
        'temp_gd': list(teams_gd.values())
    }).sort_values(
        by=['temp_points', 'temp_gd', 'team'],
        ascending=[False, False, True]
    ).reset_index(drop=True)

    standings['xg_position'] = standings.index + 1
    xg_positions.append(int(standings.loc[standings['team'] == selected_team, 'xg_position'].values[0]))

df['xg_adjusted_position'] = xg_positions

# =========================================================
# CLINICALITY SUMMARY LINE
# =========================================================
actual_goals = df['goals'].sum()
adjusted_goals = df['adj_goals'].sum()
goal_diff = int(adjusted_goals - actual_goals)

# Recompute adjusted points week-by-week using the same single-processing + “never decrease goals” logic
adjusted_points_weekly = []
teams_points_tracker = {t: 0 for t in data['team'].unique()}
processed_pairs_cum = set()

for w in range(1, df['week'].max() + 1):
    week_rows = data[data['week'] == w]

    for _, match in week_rows.iterrows():
        home_team = match['team']
        away_team = match['opponent']

        pair_key = (w, tuple(sorted([home_team, away_team])))
        if pair_key in processed_pairs_cum:
            continue
        processed_pairs_cum.add(pair_key)

        opp_rows = week_rows[
            (week_rows['team'] == away_team) & (week_rows['opponent'] == home_team)
        ]
        if opp_rows.empty:
            home_goals = int(match['goals'])
            away_goals = int(match['goals_conceded'])
            opp_row = None
        else:
            opp_row = opp_rows.iloc[0]
            home_goals = int(match['goals'])
            away_goals = int(opp_row['goals'])

        if home_team == selected_team:
            home_adj_goals = adjusted_goals_for_match(match, method_choice)
        else:
            home_adj_goals = home_goals

        if away_team == selected_team and opp_row is not None:
            away_adj_goals = adjusted_goals_for_match(opp_row, method_choice)
        else:
            away_adj_goals = away_goals

        if home_adj_goals > away_adj_goals:
            teams_points_tracker[home_team] += 3
        elif home_adj_goals == away_adj_goals:
            teams_points_tracker[home_team] += 1
            teams_points_tracker[away_team] += 1
        else:
            teams_points_tracker[away_team] += 3

    adjusted_points_weekly.append(teams_points_tracker[selected_team])

actual_points = int(df['cumulative_points'].iloc[-1])
adjusted_points = int(adjusted_points_weekly[-1])

points_diff = int(adjusted_points - actual_points)
position_diff = int(df['position'].iloc[-1] - df['xg_adjusted_position'].iloc[-1])
team_name = selected_team

# Display (unchanged wording)
if points_diff > 0 and position_diff > 0:
    st.markdown(
        f"""
        <p style='font-size:18px;'>
        🔍 By being more clinical, <strong><span style='color:{actual_color};'>{team_name}</span></strong> would have scored 
        <span style='color:{actual_color};'><strong>{goal_diff} more goals</strong></span>, gained 
        <span style='color:{actual_color};'><strong>{points_diff} more points</strong></span> and finished 
        <span style='color:{actual_color};'><strong>{position_diff} place{'s' if position_diff > 1 else ''} higher</strong></span> in the league.
        </p>
        """, unsafe_allow_html=True
    )
elif points_diff > 0:
    st.markdown(
        f"""
        <p style='font-size:18px;'>
        🔍 By being more clinical, <strong><span style='color:{actual_color};'>{team_name}</span></strong> would have scored 
        <span style='color:{actual_color};'><strong>{goal_diff} more goals</strong></span> and gained 
        <span style='color:{actual_color};'><strong>{points_diff} more points</strong></span>, 
        but it wouldn’t have helped them finish any higher in the league.
        </p>
        """, unsafe_allow_html=True
    )
else:
    st.markdown(
        f"""
        <p style='font-size:18px;'>
        🔍 By being more clinical, <strong><span style='color:{actual_color};'>{team_name}</span></strong> would have scored 
        <span style='color:{actual_color};'><strong>{goal_diff} more goals</strong></span>, 
        but it wouldn’t have gained them more points or improved their league position.
        </p>
        """, unsafe_allow_html=True
    )

# =========================================================
# KPI METRICS
# =========================================================
final_adjusted_points = adjusted_points  # From the summary calculation
final_xg_position = int(df['xg_adjusted_position'].iloc[-1])
final_position_diff = int(df['position'].iloc[-1] - final_xg_position)

c1, c2, c3, c4, c5, c6, c7, c8, c9 = st.columns(9)
with c1:
    st.metric("Actual Points", int(df['cumulative_points'].iloc[-1]))
with c2:
    st.metric("xG Adjusted Points", final_adjusted_points)
with c3:
    st.metric("Points Difference", final_adjusted_points - int(df['cumulative_points'].iloc[-1]))
with c4:
    st.metric("Actual Goals", int(df['goals'].sum()))
with c5:
    st.metric("Adj_Goals", int(df['adj_goals'].sum()))
with c6:
    st.metric("Goals Difference", goal_diff)
with c7:
    st.metric("League Position", int(df['position'].iloc[-1]))
with c8:
    st.metric("xG Adjusted Position", final_xg_position)
with c9:
    st.metric("League Pos – xG Pos", final_position_diff)

# =========================================================
# ADDITIONAL PLOTS
# =========================================================
def plot_cumulative_points():
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.fill_between(df['week'], df['cumulative_points'], df['cumulative_xg_points'],
                    where=(df['cumulative_xg_points'] > df['cumulative_points']),
                    color=gap_fill_color, alpha=0.15)
    ax.plot(df['week'], df['cumulative_points'], label='Actual Points', marker='o', markersize=3, color=actual_color)
    ax.plot(df['week'], df['cumulative_xg_points'], label='xG Adjusted Points', marker='o', markersize=3, color=xg_color)
    #ax.set_xlabel('Matchweek')
    #ax.set_ylabel('Points')
    ax.set_title(f'{selected_team} – Actual Points vs what could have been', fontsize=10)
    y_max = max(df['cumulative_points'].max(), df['cumulative_xg_points'].max())
    for y in range(20, int(y_max) + 20, 20):
        ax.axhline(y=y, color=guide_line_color, linestyle='-', linewidth=0.3, alpha=0.1)
    ax.legend(frameon=False, fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.label.set_size(10)  # smaller x label
    ax.yaxis.label.set_size(10)  # smaller y label
    ax.tick_params(axis='x', labelsize=9)  # smaller x tick labels
    ax.tick_params(axis='y', labelsize=9)  # smaller y tick labels
    return fig

def plot_league_position():
    fig, ax = plt.subplots(figsize=(6, 2))
        # Set a custom font family and size for the entire plot
    font = {'family': 'DejaVu Sans', 'size': 9}  # Change 'DejaVu Sans' to your preferred font

    # Apply font settings globally on this axis
    plt.rcParams.update({'font.family': font['family'], 'font.size': font['size']})
    ax.fill_between(df['week'], df['xg_adjusted_position'], df['position'],
                    where=(df['xg_adjusted_position'] < df['position']),
                    color=actual_color, alpha=0.15)
    ax.plot(df['week'], df['position'], label='Official Position', marker='o', markersize=3, color=actual_color)
    ax.plot(df['week'], df['xg_adjusted_position'], label='xG Position', marker='o', markersize=3, color=xg_color)
    #ax.set_xlabel('Matchweek')
    #ax.set_ylabel('League Position')
    ax.set_title(f'{selected_team} – League Position', fontsize=10)
    ax.set_ylim(20.5, 0.5)
    ax.xaxis.label.set_size(8)  # smaller x label
    ax.yaxis.label.set_size(8)  # smaller y label
    ax.tick_params(axis='x', labelsize=8)  # smaller x tick labels
    ax.tick_params(axis='y', labelsize=8)  # smaller y tick labels

    # Show only odd y-ticks (1,3,5,...19)
    odd_ticks = list(range(1, 21, 2))
    ax.set_yticks(odd_ticks)

    for y in odd_ticks:
        ax.axhline(y=y, color=guide_line_color, linestyle='-', linewidth=0.3, alpha=0.05)

    ax.legend(frameon=False, fontsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig


def plot_weekly_xg_gain():
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.bar(df['week'], df['xg_points_gain'], color=actual_color)

    # Title & axis limits
    ax.set_title(f'{selected_team} – Additional Points by Week', fontsize=10)
    ax.set_ylim(0, 3)
    ax.set_yticks([0, 1, 2, 3])

    # Match guide line style from plot_league_position()
    for y in [0, 1, 2, 3]:
        ax.axhline(y=y, color=guide_line_color, linestyle='-', linewidth=0.3, alpha=0.05)

    # Label & tick sizes
    ax.xaxis.label.set_size(8)
    ax.yaxis.label.set_size(8)
    ax.tick_params(axis='x', labelsize=9)
    ax.tick_params(axis='y', labelsize=9)

    # Remove plot borders
    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig


def plot_additional_goals():
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.bar(df['week'], df['additional_goals'], color=xg_color)
    #ax.axhline(0, color=xg_color, linewidth=0.8)
    #ax.set_xlabel('Matchweek')
    #ax.set_ylabel('Additional Goals (xG - Actual)')
    ax.set_title(f'{selected_team} – Additional Goals by Week', fontsize=10)
    ax.set_ylim(0, 3)
    ax.set_yticks([0, 1, 2, 3])

    # Match guide line style from plot_league_position()
    for y in [0, 1, 2, 3]:
        ax.axhline(y=y, color=guide_line_color, linestyle='-', linewidth=0.3, alpha=0.05)

    ax.xaxis.label.set_size(8)  # smaller x label
    ax.yaxis.label.set_size(8)  # smaller y label
    ax.tick_params(axis='x', labelsize=9)  # smaller x tick labels
    ax.tick_params(axis='y', labelsize=9)  # smaller y tick labels
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig

# =========================================================
# DISPLAY PLOTS
# =========================================================
col1, col2 = st.columns(2)
with col1:
    st.pyplot(plot_cumulative_points())
with col2:
    st.pyplot(plot_league_position())

col3, col4 = st.columns(2)
with col3:
    st.pyplot(plot_weekly_xg_gain())
with col4:
    st.pyplot(plot_additional_goals())

# =========================================================
# FLAGS & DETAIL TABLE
# =========================================================
# === Flags ===
df['result_mismatch_flag'] = (df['result'] != df['xg_result']).apply(lambda x: '✅' if x else '')
df['goal_adjustment_emoji'] = (df['goals'] != df['adj_goals']).apply(lambda x: '⚠️' if x else '')

# === Detail Table ===
st.write("\n\n")
st.markdown("---")  # divider
st.subheader("📊 Matchweek Detail Table")
df['actual_gd'] = df['goals'].cumsum() - df['goals_conceded'].cumsum()
df['xg_gd'] = df['adj_goals'].cumsum() - df['goals_conceded'].cumsum()

summary_df = df[[
    'week', 'opponent', 'home_away', 'xg', 'result', 'xg_result', 'score', 'xg_score', 
    'goal_adjustment_emoji', 'result_mismatch_flag',  
    'points', 'xg_points', 
    'cumulative_points', 'cumulative_xg_points',
    'actual_gd', 'xg_gd',
]]

column_aliases = {
    'week': 'Wk', 'opponent': 'Opponent', 'home_away': 'H/A', 'xg': 'xG', 'result': 'Result',
    'xg_result': 'xG Result', 'score': 'Score', 'xg_score': 'xG_Score',
    'goal_adjustment_emoji': 'Adj⚠️', 'result_mismatch_flag': '✅Res',
    'points': 'Points', 'xg_points': 'xG Points',
    'cumulative_points': 'cu. Points', 'cumulative_xg_points': 'xG cu. Points',
    'actual_gd': 'cu. GD', 'xg_gd': 'cu. xG GD',
}
summary_df = summary_df.rename(columns=column_aliases)

col1, col2 = st.columns(2)
with col1:
    show_goal_adj = st.checkbox("Show only Adjusted Goal differences (⚠️)")
with col2:
    show_only_mismatches = st.checkbox("Show only Result mismatches (✅)")

if show_goal_adj:
    summary_df = summary_df[summary_df['Adj⚠️'] == '⚠️']
if show_only_mismatches:
    summary_df = summary_df[summary_df['✅Res'] == '✅']

st.dataframe(summary_df, use_container_width=True)
