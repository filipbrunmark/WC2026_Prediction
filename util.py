import math

def calculate_team_a_expected_result(a_elo,b_elo):
    return 1 / (1 + pow(10,-(a_elo-b_elo)/600))

def update_elo_draw(a_elo, b_elo, i):
    w = 0.5

    new_a_elo = a_elo + i*(w - calculate_team_a_expected_result(a_elo, b_elo))
    new_b_elo = b_elo + i*(w - calculate_team_a_expected_result(b_elo,a_elo))
    return (new_a_elo, new_b_elo)

def update_elo_win(win_elo, loss_elo, i):
    w_win = 1
    w_loss = 0

    new_win_elo = win_elo + i*(w_win - calculate_team_a_expected_result(win_elo, loss_elo))
    new_loss_elo = loss_elo + i*(w_loss - calculate_team_a_expected_result(loss_elo,win_elo))
    return (new_win_elo, new_loss_elo)

friendlies = ['friendly']
nations_leauge = ['uefa nations leauge']
qualifications = ['fifa world cup qualification', 'afc asian cup qualification', 'copa américa qualification',
                   'uefa euro qualification', 'concacaf championship qualification', 'african cup of nations qualification']
confederation_finals = ['copa américa', 'uefa euro', 'african cup of nations', 'concaf championship', 'afc asian cup', 
        'oceania nations cup', 'confederations cup']
world_cup = ['fifa world cup']

#retrieve match type and assign importance of game based on the match partitioning above
def game_importance_score(row):
    tournament = row['tournament'].lower()
    
    if tournament in friendlies:
        return 10

    if tournament in nations_leauge:
        return 15
    
    if tournament in qualifications:
        return 25

    if tournament in confederation_finals:
        return 35
    
    if tournament in world_cup:
        return 60

    
    return 10

MAX_GOALS = 15

# PMF: Probability Mass Function 
# k = Number of goals
# lamda = Number of expected goals
def poisson_pmf(k, lam):
    return (lam**k * math.exp(-lam)) / math.factorial(k)

def get_match_res_prob(lambda_home,lambda_away):
    prob_home_win = 0
    prob_draw = 0
    prob_home_loss = 0

    for h_goals in range(0,MAX_GOALS):
        for a_goals in range(0,MAX_GOALS):
            prob_home_goals = poisson_pmf(h_goals, lambda_home)
            prob_away_goals = poisson_pmf(a_goals, lambda_away)
            prob_outcome = prob_home_goals*prob_away_goals
            if h_goals > a_goals:
                prob_home_win += prob_outcome
            elif h_goals == a_goals:
                prob_draw += prob_outcome
            else:
                prob_home_loss += prob_outcome
                
    return (prob_home_win, prob_draw, prob_home_loss)