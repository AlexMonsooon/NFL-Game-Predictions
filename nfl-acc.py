import pandas as pd 
import numpy as np
from sqlalchemy import create_engine
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import BaseCrossValidator
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif

engine = create_engine('mysql+pymysql://root:password@127.0.0.1:3306/sys')    
print("Connection successful.")

################################################################################
def fetch_existing_data(
    engine, 
    table_name, 
    cols='*', 
    joins=None, 
    groupby=None, 
    orderby=None
):
    """
    Fetch data from a database with optional JOINs, GROUP BY, and ORDER BY clauses.

    Parameters:
    - engine: SQLAlchemy engine for database connection.
    - table_name: Main table name to fetch data from.
    - cols: Columns to select (default is '*').
    - joins: List of tuples for JOINs [(join_table, join_condition)].
    - groupby: Columns for GROUP BY clause.
    - orderby: Columns for ORDER BY clause.

    Returns:
    - DataFrame: Query result as a pandas DataFrame.
    """
    
    query = f"SELECT {cols} FROM {table_name}"

    if joins:
        for join_table, join_condition in joins:
            query += f" JOIN {join_table} ON {join_condition}"

    if groupby:
        groupby_clause = ", ".join(groupby)
        query += f" GROUP BY {groupby_clause}"

    if orderby:
        orderby_clause = ", ".join(orderby)
        query += f" ORDER BY {orderby_clause}"
        
    return pd.read_sql(query, engine)

################################################################################
# Convert 'Record' column to a percentage, -1 for first week
def record_to_percentage(record):
    try:
        wins, games = map(int, record.split('/'))
        return (wins / games) * 100 if games > 0 else 0
    except:
        return -1  # Handle invalid or missing data gracefully

################################################################################
def custom_shift(group):
    return group.shift(1) if len(group) > 1 else group

###############################################################################
def shift_vals(df, cols):
    results_df = pd.DataFrame(index=df.index)
    df_sorted = df.sort_values(by=['FullTeam', 'Season', 'Game_Week'])
    grouped = df_sorted.groupby(['FullTeam', 'Season'])
    
    prcolumns = [col for col in cols]
    prefixed_columns = ['Shifted_' + col for col in prcolumns]
    
    results_df[prefixed_columns] = grouped[cols].transform(custom_shift)
    results_df.dropna(inplace=True)
    return results_df

###############################################################################
### ema tested 8/18 confirmed to work and usese 4th previous to predict 5th
### must shift both ema and trailing avg

def rolling_window_ema(group, columns):
    ema_df = pd.DataFrame()
    for col in columns:
        # Compute EMA for the column
        ema_col = group[col].ewm(alpha=0.6).mean()
        # Shift the EMA values by 1 within the group
        ema_df[f'EMA_{col}'] = ema_col.shift(1)
        
    return ema_df

###############################################################################
def calculate_ema_average(df, cols):
    # Sort and group the data
    df_sorted = df.sort_values(by=['FullTeam', 'Season', 'Game_Week'])
    grouped = df_sorted.groupby(['FullTeam', 'Season'])

    # Apply rolling_window_ema to each group
    trailing_ema_df = grouped.apply(lambda g: rolling_window_ema(g, cols), include_groups=False)
    
    ## resest index
    trailing_ema_df.reset_index(inplace=True)

    ## make level_2 the index which is what it was in original df
    trailing_ema_df.set_index('level_2', inplace=True)
    return trailing_ema_df

################################################################################
###############################################################################
def target_encoder(train_df, val_df, cats_cols, target):
    """
    Encodes categorical columns with target encoding, ensuring no data leakage.

    Parameters:
    train_df (pd.DataFrame): Training DataFrame.
    val_df (pd.DataFrame): Validation DataFrame.
    cats_cols (list): List of categorical column names to encode.
    target (str): Target column name for encoding.

    Returns:
    pd.DataFrame, pd.DataFrame: Encoded training and validation DataFrames.
    """
    train_encoded = train_df.copy()
    val_encoded = val_df.copy()

    # Encode 'FullTeam' and 'Opp' with same values
    if 'FullTeam' in cats_cols and 'Opp' in cats_cols:
        team_target_means = train_df.groupby('FullTeam')[target].mean()
        train_encoded['FullTeam'] = train_df['FullTeam'].map(team_target_means)
        train_encoded['Opp'] = train_df['Opp'].map(team_target_means)
        val_encoded['FullTeam'] = val_df['FullTeam'].map(team_target_means)
        val_encoded['Opp'] = val_df['Opp'].map(team_target_means)

        cats_cols = [col for col in cats_cols if col not in ['FullTeam', 'Opp']]

    # Encode remaining categorical columns
    for col in cats_cols:
        col_means = train_df.groupby(col)[target].mean()
        
        train_encoded[col] = train_df[col].map(col_means)
        val_encoded[col] = val_df[col].map(col_means)

    return train_encoded, val_encoded

###############################################################################
class TemporalSeasonSplitter(BaseCrossValidator):
    def __init__(self, n_train_weeks):
        self.n_train_weeks = n_train_weeks  # Number of weeks to use for training

    def split(self, X, y=None, groups=None):
        """
        groups: Unique identifiers for season_week (e.g., '2021_1').
        """
        unique_seasons = X['Season'].unique()

        for season in unique_seasons:
            # Sort by season and week to guarantee chronological order
            season_data = X[X['Season'] == season].sort_values(by='Game_Week')
            
            # Split into training and validation based on weeks
            train_data = season_data[season_data['Game_Week'] <= self.n_train_weeks]
            val_data = season_data[season_data['Game_Week'] > self.n_train_weeks]

            train_idx = train_data.index
            val_idx = val_data.index

            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(X['Season'].unique())  # One split per season


###############################################################################
###############################################################################
###############################################################################
games_df = fetch_existing_data(engine, 'games', orderby=['Game_Date'])

## can drop games_id and FullTeam cols, have Tm to use as our main
games_df.drop(columns=['games_id', 'Tm'], inplace=True)

day_mapping = {'Sun': 0, 'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6}
games_df['Game_Day'] = games_df.loc[:, 'Game_Day'].map(day_mapping)
    
# Change x/x to % won and shift records by 1
games_df['Record'] = games_df.groupby(['FullTeam', 'Season'])['Record'].transform(custom_shift)
games_df['Record'] = games_df['Record'].apply(record_to_percentage)

games_df['Beat_Spread'] = games_df.groupby(['FullTeam', 'Season'])['Beat_Spread'].transform(custom_shift)

shift_cols = ['Duration', 'PF', 'PA', 'First_Downs', 'Total_Yards', 'Turnovers',
 'Third_Down_Conv_', 'Fourth_Down_Conv_', 'Time_of_Possession', 'Penalties',
 'Penalty_Yards', 'Spread']

### ema of shifted cols
ema_df = calculate_ema_average(games_df, shift_cols)
ema_df.dropna(inplace=True)

shift_by_1 = ['Beat_Spread', 'Record']

# concat games_df no_shift cols back with index 'level_2' of games_df 
# FullTeam and Season are already in the ema_df so they are taken out
# add shift_by_1 after shifting those records
no_shift = ['Opp', 'Coach', 'Stadium', 'Surface', 'Roof', 'Game_Week', 'Game_Date',
            'Game_Day', 'Link', 'Start_Time', 'Attendance', 'Spread', 'Over_Under',
            'Temperature', 'Humidity', 'Wind', 'Rest', 'HA', 'Result'] + shift_by_1

no_shift_data = games_df[no_shift]

no_shift_data_reindexed = no_shift_data.reindex(ema_df.index)
ema_df_combined = pd.concat([ema_df, no_shift_data_reindexed], axis=1)
ema_df_combined.drop(columns=['Game_Date', 'Start_Time'], inplace=True)

################################################################################
# add FullTeam + 'Stadium', 'Surface', 'Roof' 
# 'Coach' and Record are very similar
cats_cols = ['FullTeam', 'Opp', 'Coach', 'Stadium', 'Surface', 'Roof']
# str_encoder = TargetEncoder()

discrete_cols = ['Season', 'Game_Week', 'Game_Day', 'Beat_Spread', 'Result', 'Turnovers', 'Rest', 'Penalties', 'HA']

# TOOK OUT EMA_First_Downs, EMA_Penalties
continous_cols = ['EMA_Duration', 'EMA_PF', 'EMA_PA',
       'EMA_Total_Yards', 'EMA_Turnovers',
       'EMA_Third_Down_Conv_', 'EMA_Fourth_Down_Conv_',
       'EMA_Time_of_Possession', 'EMA_Penalty_Yards',
       'EMA_Spread'] # 'Over_Under', 'Record'


test_df = ema_df_combined[['EMA_Duration', 'EMA_PF', 'EMA_PA',
       'EMA_First_Downs', 'EMA_Total_Yards', 'EMA_Turnovers',
       'EMA_Third_Down_Conv_', 'EMA_Fourth_Down_Conv_', 'HA', 'Game_Day',
       'EMA_Time_of_Possession', 'EMA_Penalties', 'EMA_Penalty_Yards',
       'EMA_Spread', 'Result', 'Season', 'Game_Week', 'Spread',
       'Rest', 'FullTeam', 'Opp', 'Coach', 'Stadium', 'Surface', 'Roof']]


test_df = test_df.sort_values(by=['Season', 'Game_Week']).reset_index(drop=True)
target = test_df['Result']
#test_df.drop(columns=['Result', 'Link', 'Attendance', 'FullTeam', 'Season'], inplace=True)

temporal_splitter = TemporalSeasonSplitter(n_train_weeks=8)

logreg = LogisticRegression(random_state=0, max_iter=1000)

feature_columns = [
    'EMA_Duration', 'EMA_PF', 'EMA_PA', 'EMA_Total_Yards',
    'EMA_Turnovers', 'EMA_Third_Down_Conv_', 'EMA_Fourth_Down_Conv_',
    'EMA_Time_of_Possession', 'EMA_Penalty_Yards', 'Rest',
    'EMA_Spread', 'Spread']

accuracies, precisions, recalls, f1_scores, aucs = [], [], [], [], []

# Number of features to select
k = 1  

## result for 1 feature (Spread), simplest model with highest accuracy
# Mean Accuracy: 0.6962 ± 0.0412
# Mean Precision: 0.6970 ± 0.0387
# Mean Recall: 0.6976 ± 0.0435
# Mean F1 Score: 0.6973 ± 0.0410
# Mean ROC AUC: 0.7508 ± 0.0484


# Loop through each Season 1-8 is train
# 9-18 is test set
# however many seasons is how many folds
for train_idx, val_idx in temporal_splitter.split(test_df):
    # Split into training and validation sets
    train_data = test_df.iloc[train_idx]
    val_data = test_df.iloc[val_idx]

    # Separate features and target
    X_train = train_data[feature_columns]
    y_train = train_data['Result']
    X_val = val_data[feature_columns]
    y_val = val_data['Result']

    # StandardScaler mean = 0, std = 1 > only works on continous
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Feature selection with SelectKBest... f_classif = ANOVA
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_val_selected = selector.transform(X_val_scaled)
    
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_selected, y_train)

    y_pred = model.predict(X_val_selected)
    y_pred_proba = model.predict_proba(X_val_selected)[:, 1]

    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, zero_division=1)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)

    print("Fold Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"ROC AUC: {auc:.4f}")
    print("Selected Features:", [feature_columns[i] for i in selector.get_support(indices=True)])
    print("-" * 50)
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    aucs.append(auc)
    print("-" * 50)

    
# Print average metrics across folds
print(f"Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
print(f"Mean Precision: {np.mean(precisions):.4f} ± {np.std(precisions):.4f}")
print(f"Mean Recall: {np.mean(recalls):.4f} ± {np.std(recalls):.4f}")
print(f"Mean F1 Score: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")
print(f"Mean ROC AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
