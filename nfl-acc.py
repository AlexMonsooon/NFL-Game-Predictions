import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, \
    RobustScaler, StandardScaler, Normalizer
from scipy.stats import zscore
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, \
    RobustScaler, StandardScaler, Normalizer
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, \
    HistGradientBoostingClassifier, IsolationForest
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import PassiveAggressiveClassifier, Perceptron, RidgeClassifier, SGDClassifier, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import BernoulliNB, ComplementNB, MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier



data = pd.read_csv(r'C:\Users\Alex\Downloads\NFLData3.csv', low_memory=False).drop(columns=['Unnamed: 0', 'LTime', 'Rk', 'PC', 'PD'])
data['Date'] = pd.to_datetime(data['Date'])

# selecting only data from 2007 - present (2019)
data = data[data['Date'].dt.year >= 2007]
data = data.rename(columns={'Unnamed: 6': 'HomeAway'})

ot_mapping = {'OT': 1, np.NaN: 0}
data['OT'] = data.loc[data.index, 'OT'].map(ot_mapping)
data['HomeAway'] = data.loc[data.index, 'HomeAway'].fillna(0).replace('@', 1)
data['TO'] = data.loc[data.index, 'TO'].fillna(0)

# Select categorical data
cats = data.select_dtypes(include='object')
cats.dropna(inplace=True)
cats.drop(columns=['ToP', 'Time.1', 'Time'], inplace=True)

# Remove Data that has multicollinearity / not as importnant
data.drop(columns=['Cmp', 'Att', 'Yds', 'Cmp%', 'Int', 'Att.1', 'Yds.2', 'Tot', 'Sk', 'TD', 'TD.1', 'Rate'], inplace=True)

#cumulative columns
cumcols = ['PF', 'PA', 'Result']
exclude = ['Tm', 'Year', 'Week']


# shifting groups by 1, so x = x + 1
def custom_shift(group):
  return group.shift(1) if len(group) > 1 else group

# Wins or loses
def convert_result(result):
        if result.startswith('W'):
            return 1
        elif result.startswith('L'):
            return 0
        else:
            return 1  # tie game but counted as win... can change


data['Result'] = data.loc[data.index, 'Result'].apply(convert_result)

# Converting ToP, Time.1, Time from x:x to integers
def convert_to_seconds(time_str, desc=0):
        if pd.notnull(time_str) and desc == 0:
            hours, minutes = map(int, time_str.split(':'))
            return (hours * 3600) + (minutes * 60)
        elif desc == 1:
            minutes, seconds = map(int, time_str.split(':'))
            return (minutes * 60) + seconds
        else:
            return None

data['Time'] = data.loc[data.index, 'Time'].apply(lambda x: convert_to_seconds(x))
data['Time.1'] = data.loc[data.index, 'Time.1'].apply(lambda x: convert_to_seconds(x))
data['ToP'] = data.loc[data.index, 'ToP'].apply(lambda x: convert_to_seconds(x, 1))

# 4 games to predict 5th
min_games = 4
def calculate_trailing_average(df, cols):
      df_sorted = df.sort_values(by=['Tm', 'Year', 'G#'])
      grouped = df_sorted.groupby(['Tm', 'Year'])

      # average for selected numerical cols
      trailing_avg_df = grouped[cols].rolling(window=min_games, min_periods=min_games).mean()


      # Findng cumsum for select cum cols 
      csum =  grouped[cumcols].cumsum()
      csum = csum.add_prefix('cum_')
      trailing_avg_df.reset_index(inplace=True)
      trailing_avg_df.set_index('level_2', inplace=True)

      # concating cumsums to trailingavg on cols
      trailing_avg_df = pd.concat([trailing_avg_df, csum], axis=1)

      # exclude these cols from the shift
      excl = [col for col in trailing_avg_df.columns if col not in exclude]
      trailing_avg_df[excl] = trailing_avg_df.groupby(['Tm', 'Year']).transform(custom_shift)

      trailing_avg_df.dropna(inplace=True)

      return trailing_avg_df

# transform the data so we can predict with it
def trans(df):
    # Select only numeric columns
    passnums = df.select_dtypes(include=['int', 'float'])
    passnums.dropna(inplace=True)
    # CHECK DROPPED COLUMNS
    passnums.drop(columns=['Week', 'G#', 'Year', 'Result'], inplace=True)

    trailing = calculate_trailing_average(df, passnums.columns)

    columns_to_include = [col for col in trailing.columns if col not in exclude]

    # remove outliers
    z_scores = zscore(trailing[columns_to_include])
    abs_z_scores = abs(z_scores)
    z_score_threshold = 3
    outliers = (abs_z_scores > z_score_threshold).any(axis=1)
    
    trailing = trailing[~outliers]
    # getting the categorical data for trailing
    trailing[cats.columns] = df.loc[trailing.index, cats.columns]


    # Convert categorical columns to numerical codes
    day_mapping = {'Sun': 0, 'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6}
    trailing['Day'] = df.loc[trailing.index, 'Day'].map(day_mapping)
    trailing['Tm'] = df.loc[trailing.index, 'Tm'].astype('category').cat.codes + 1
    trailing['Opp'] = df.loc[trailing.index, 'Opp'].astype('category').cat.codes + 1
    trailing['Year'] = df.loc[trailing.index, 'Year']
    trailing['Time'] = df.loc[trailing.index, 'Time']
    trailing['Time.1'] = df.loc[trailing.index, 'Time.1']
    trailing['ToP'] = df.loc[trailing.index, 'ToP']
    trailing['Result'] = df.loc[trailing.index, 'Result']
    trailing['Week'] = df.loc[trailing.index, 'Week']


    # Rank columns
    wy = trailing.groupby(['Year', 'Week'])
    trailing['PA_rank'] = wy['cum_PA'].rank(ascending=False)
    trailing['PF_rank'] = wy['cum_PF'].rank(ascending=True)
    trailing['Win_rank'] = wy['cum_Result'].rank(ascending=True)
    
    
    # Make sure all Data less than min games is removed
    trailing = trailing.query(f'Week > {min_games}')


    #ADD OPP STATS
    for index, row in trailing.iterrows():
      oppteam = row['Opp']

      # extract opponent PF, PA, Winrank
      opp_row = trailing.loc[(oppteam == trailing['Tm']) & (trailing['Week'] == row['Week']) & (trailing['Year'] == row['Year']), ['PF', 'PA', 'Win_rank']]

      if not opp_row.empty:
        trailing.at[index, 'Opp_PF'] = opp_row.iloc[0]['PF']
        trailing.at[index, 'Opp_PA'] = opp_row.iloc[0]['PA']
        trailing.at[index, 'Opp_WinRank'] = opp_row.iloc[0]['Win_rank']

    return trailing


# clean and avg data for predictions
data = trans(data)
data.dropna(inplace=True)
#drop cumcols
data.drop(columns=['cum_PF', 'cum_PA', 'cum_Result'], inplace=True)
target = data['Result']


# scaling techniques used
# rob removes the median and scales the data according to the quantile range.
# The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).
# stan z = (x - u) / s and rob 
def scale_num(nums):
    scaled_data = {}
    scalers = {
        'rob': RobustScaler(),
        'stan': StandardScaler()
    }

    # scaling data with selected scalers and returing df
    for key, scaler in scalers.items():
        scaled_data[key] = pd.DataFrame(scaler.fit_transform(nums), columns=nums.columns)
        
    return scaled_data

#selecting all numbers to scale
nums = data.select_dtypes(include=[np.number])
# DROP TARGET VAR
nums.drop(columns='Result', inplace=True)
scalie = scale_num(nums)

# check for multicollinearity
def vif(data, threshold_vif):
    results = {}
    for key, df in data.items():
        vif = pd.DataFrame()
        vif['Variable'] = df.columns
        vif['VIF'] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        high_vif_variables = vif[vif['VIF'] >= threshold_vif]['Variable'].tolist()
        selected_features = vif[~vif['Variable'].isin(high_vif_variables)]['Variable'].tolist()
        results[key] = df[selected_features]
    return results



# Linear classifiers
lin = {
        'sgd': SGDClassifier(), 'lr': LogisticRegression(), 'pa': PassiveAggressiveClassifier(),
'per': Perceptron(), 'rd': RidgeClassifier(),
    }

# Discriminant Analysis
disc = {
    'ld': LinearDiscriminantAnalysis(),
    'quad': QuadraticDiscriminantAnalysis()
}

# Nearest Neighbors
knear = {
    'knn': KNeighborsClassifier(),
    'ncen': NearestCentroid()
}
#', 'rad': RadiusNeighborsClassifier(),

# Neural network models
nn = {
    'bnb':  BernoulliNB(),
    'ml':  MLPClassifier()
}

# Support Vector Machines
svm = {
    'svc': SVC(),
    'linsvc':LinearSVC(),
}
#'nsvc':NuSVC(),

#Gaussian Processes Naive Bayes
bay = {
    'gauspc': GaussianProcessClassifier(),
    'gausnb': GaussianNB()
}
#NO NEGATIVE VALUES
# 'comp':ComplementNB(),
# 'mnom': MultinomialNB(),

dtree = {
    'dt': DecisionTreeClassifier(),
    'et': ExtraTreeClassifier(),
    'rfc': RandomForestClassifier()
}

glm = {
    'gradboost': GradientBoostingClassifier(),
    'hist': HistGradientBoostingClassifier()
}

d = {
    'iso': IsolationForest()
}


lr_grid = {
        'penalty': ['l1', 'l2'],
        'C': [0.001, 1, 10],
        'solver': ['liblinear'],
        'class_weight': [None, 'balanced'],
        'max_iter': [1000]
    }

pa_grid = {
    'C': [.01, 1, 1.5],
    'loss': ['hinge', 'squared_hinge'],
    'class_weight': [None, 'balanced']
}

per_grid = {
    'alpha': [.01, 1, 2],
    'penalty': ['l1', 'l2', 'elasticnet', None],
    'class_weight': [None, 'balanced']
}

rd_grid = {
    'alpha': [.01, .2, .8],
    'solver': ['auto', 'svd', 'lsqr'],
    'class_weight': [None, 'balanced']
}

sgd_grid = {
    'loss': ['perceptron', 'modified_huber', 'squared_hinge', 'hinge', 'log_loss'],
    'penalty': ['l1', 'l2', 'elasticnet'],
}

ld_grid = {
    'solver': ['svd', 'lsqr', 'eigen']
}

quad_grid = {
    'reg_param': [0, .5, 1]
}

knn_grid = {
    'n_neighbors': [10,11],
    'weights': ['uniform', 'distance'],
    'algorithm': ['ball_tree'], # , 'kd_tree', 'brute'
    'leaf_size': [5, 10],
    'metric': ['euclidean', 'manhattan'], # , 'chebyshev', 'minkowski'
    'p': [1, 2]
}


ncen_grid = {
    'metric': ['euclidean', 'manhattan']
}

bnb_grid = {
    'alpha': [.1, .5, 1, 1.5]
}

#takes a long time lbfgs failed
ml_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
    'activation': ['logistic', 'tanh', 'relu', 'identity'],
    'solver': ['lbfgs', 'adam', 'sgd'],
    'learning_rate': ['constant', 'adaptive', 'invscaling'],
    'max_iter': [500]
}

svc_grid = {
    'C': [.1, .5, 1],
    'gamma': ['scale', 'auto'],
    'class_weight': [None, 'balanced'],
}

# 'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],


linsvc_grid = {
    'penalty': ['l1', 'l2'],
    'loss': ['hinge', 'squared_hinge'],
    'C': [.1, .5, .9],
    'class_weight': [None, 'balanced']
}


dt_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 30, 100],
    'max_features': [None, 'sqrt', 'log2']
}

et_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 30, 100],
    'max_features': [None, 'sqrt', 'log2']
}

rfc_grid = {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'max_depth': [None, 30, 100],
    'max_features': [None, 'sqrt', 'log2']
}


featsel = {
    'kb6': SelectKBest(k=6),
    'kb7': SelectKBest(k=7),
    'kb8': SelectKBest(k=8),
    'kb9': SelectKBest(k=9),
}


def train_and_evaluate(features, target, model):
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    ypred = model.predict(X_test)
    acc = accuracy_score(y_test, ypred)
    roc = roc_auc_score(y_test, ypred)
    f1 = f1_score(y_test, ypred)

    return {'accuracy': acc, 'roc_auc': roc, 'f1_score': f1}

vif(scalie, 8)
def itry(featsel, vif_features, target, models):
    results = {}

    for key, value in featsel.items():
        for prekey, prevalue in vif_features.items():
            redue_key = f'{key}_{prekey}'
            selected_features = value.fit_transform(prevalue, target)
            selected_mask = value.get_support()

            selected_names = prevalue.columns[selected_mask]
            selected_data = pd.DataFrame(selected_features, columns=selected_names)

            for mod, tech in models.items():
                model = tech
                nkey = f'{mod}_{redue_key}'

                model_results = train_and_evaluate(selected_data, target, model)
                results[nkey] = model_results

    return results


# models >>> lin, disc, knear, nn, svm, bay, dtree
results = itry(featsel, vif(scalie, 8), target, disc)

result_rows = []
for key, values in results.items():
    result_rows.append([key, values['accuracy'], values['roc_auc'], values['f1_score']])  # Assuming key is nkey

df = pd.DataFrame(result_rows, columns=['Model', 'Accuracy', 'ROC_AUC', 'F1'])
df_sorted = df.sort_values(by='Accuracy', ascending=False)

print(df_sorted.to_string())



