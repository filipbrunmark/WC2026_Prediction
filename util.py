from collections import defaultdict
import pandas as pd
import math
import numpy as np
from sklearn.linear_model import LinearRegression

# ---------------------------------------
# 🔹 ELO CALCULATION UTILITIES
# ---------------------------------------

def calculate_team_a_expected_result(a_elo, b_elo):
    """
    Calculate the expected result of team A vs team B using their ELO ratings.
    """
    return 1 / (1 + pow(10, -(a_elo - b_elo) / 600))


def update_elo(a_elo, b_elo, score_diff, i):
    """
    Update the ELO scores of two teams based on the score difference.
    Delegates to win/draw handling functions.
    Parameter:
    score_diff = [Team A score] - [Team B score]
    Returns:
    tuple: (new_a_elo, new_b_elo)
    """
    if score_diff > 0:
        return update_elo_win(a_elo, b_elo, i)
    elif score_diff == 0:
        return update_elo_draw(a_elo, b_elo, i)
    else:
        return update_elo_win(b_elo, a_elo, i)[::-1]  # reverse result


def update_elo_draw(a_elo, b_elo, i):
    """
    Update ELO ratings after a draw.
    """
    w = 0.5
    new_a_elo = a_elo + i * (w - calculate_team_a_expected_result(a_elo, b_elo))
    new_b_elo = b_elo + i * (w - calculate_team_a_expected_result(b_elo, a_elo))
    return new_a_elo, new_b_elo


def update_elo_win(win_elo, loss_elo, i):
    """
    Update ELO ratings after a win/loss.
    """
    new_win_elo = win_elo + i * (1 - calculate_team_a_expected_result(win_elo, loss_elo))
    new_loss_elo = loss_elo + i * (0 - calculate_team_a_expected_result(loss_elo, win_elo))
    return new_win_elo, new_loss_elo

# ---------------------------------------
# 🔹 TOURNAMENT IMPORTANCE SETUP
# ---------------------------------------

# Match categories
friendlies = ['friendly']
nations_league = ['uefa nations leauge']
qualifications = [
    'fifa world cup qualification', 'afc asian cup qualification', 'copa américa qualification',
    'uefa euro qualification', 'concacaf championship qualification', 'african cup of nations qualification'
]
confederation_finals = [
    'copa américa', 'uefa euro', 'african cup of nations', 'concaf championship', 'afc asian cup',
    'oceania nations cup', 'confederations cup'
]
world_cup = ['fifa world cup']


def game_importance_score(row):
    """
    Assign an importance score to a match based on its tournament.
    """
    tournament = row['tournament'].lower()

    if tournament in friendlies:
        return 10
    if tournament in nations_league:
        return 15
    if tournament in qualifications:
        return 25
    if tournament in confederation_finals:
        return 35
    if tournament in world_cup:
        return 60

    return 10  # default for unknown tournaments

# ---------------------------------------
# 🔹 POISSON MATCH RESULT MODELING
# ---------------------------------------

MAX_GOALS = 15

def poisson_pmf(k, lam):
    """
    Poisson probability mass function for scoring k goals given expected value lambda.
    """
    return (lam ** k * math.exp(-lam)) / math.factorial(k)


def get_match_res_prob(lambda_home, lambda_away):
    """
    Compute probabilities for home win, draw, and away win using Poisson-distributed goals.
    """
    prob_home_win = 0
    prob_draw = 0
    prob_home_loss = 0

    for h_goals in range(MAX_GOALS):
        for a_goals in range(MAX_GOALS):
            prob_outcome = poisson_pmf(h_goals, lambda_home) * poisson_pmf(a_goals, lambda_away)
            if h_goals > a_goals:
                prob_home_win += prob_outcome
            elif h_goals == a_goals:
                prob_draw += prob_outcome
            else:
                prob_home_loss += prob_outcome

    return prob_home_win, prob_draw, prob_home_loss

# ---------------------------------------
# 🔹 POISSON-BASED MATCH PREDICTION MODEL
# ---------------------------------------

class MRP_Poisson_Dist:
    """
    Match Result Prediction using Poisson Distribution and linear regression based on ELO difference.
    """

    def fit(self, elo_diff, home_score, away_score):
        """
        Fit linear models to predict home and away goals from ELO difference.
        """
        self.model_home = LinearRegression()
        self.model_away = LinearRegression()
        self.model_home.fit(elo_diff, home_score)
        self.model_away.fit(elo_diff, away_score)

    def random_res(self, elo_diff):
        """
        Generate a random match result (goals) using fitted Poisson models.
        """
        lambda_h = self.model_home.predict(elo_diff)
        lambda_a = self.model_away.predict(elo_diff)

        home_goals = np.random.poisson(lambda_h)
        away_goals = np.random.poisson(lambda_a)

        return home_goals, away_goals

# ---------------------------------------
# 🔹 STANDINGS / TABLE PLACEMENT
# ---------------------------------------

def get_placement(df):
    """
    Calculate final standings from match results based on points, goal difference, and goals scored.
    """
    points = defaultdict(int)
    goal_diff = defaultdict(int)
    goals_for = defaultdict(int)

    for _, row in df.iterrows():
        home = row['home_team']
        away = row['away_team']
        home_score = row['home_score']
        away_score = row['away_score']

        # Update points
        if home_score > away_score:
            points[home] += 3
        elif home_score < away_score:
            points[away] += 3
        else:
            points[home] += 1
            points[away] += 1

        # Update goal stats
        goal_diff[home] += home_score - away_score
        goal_diff[away] += away_score - home_score
        goals_for[home] += home_score
        goals_for[away] += away_score

    # Create and sort table
    table = pd.DataFrame({
        'team': list(points.keys()),
        'points': [points[t] for t in points],
        'goal_diff': [goal_diff[t] for t in points],
        'goals_for': [goals_for[t] for t in points]
    })

    table = table.sort_values(by=['points', 'goal_diff', 'goals_for'], ascending=False)
    return table['team'].to_numpy()

# ---------------------------------------
# 🔹 GROUP MATCH SIMULATION
# ---------------------------------------

def simulate_group_play(df_group: pd.DataFrame, starting_elo: dict, mrp_model: MRP_Poisson_Dist):
    """
    Simulate the results of a full group play using a match result prediction model.

    Parameters:
        df_group (DataFrame): All group games (ordered), with 'home_team' and 'away_team'.
        starting_elo (dict): Current ELO ratings for each team.
        mrp_model (MRP_Poisson_Dist): Model to generate match results.

    Returns:
        DataFrame: Original df_group with added 'home_score' and 'away_score' columns.
    """
    current_elo = starting_elo.copy()
    df_group_copy = df_group.copy()
    home_scores = []
    away_scores = []
    importance = 25  # Constant qualification importance

    for _, row in df_group.iterrows():
        home = row['home_team']
        away = row['away_team']
        home_elo = current_elo[home]
        away_elo = current_elo[away]

        # Predict result
        home_goals, away_goals = mrp_model.random_res(pd.DataFrame([[home_elo - away_elo]], columns=["elo_diff"]))
        home_scores.append(home_goals[0])
        away_scores.append(away_goals[0])

        # Update elo
        new_home_elo, new_away_elo = update_elo(home_elo, away_elo, home_goals - away_goals, importance)
        current_elo[home] = new_home_elo
        current_elo[away] = new_away_elo

    df_group_copy['home_score'] = home_scores
    df_group_copy['away_score'] = away_scores

    return df_group_copy
