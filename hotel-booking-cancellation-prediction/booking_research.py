import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import warnings
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import re
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, RocCurveDisplay
from sklearn.model_selection import learning_curve, ShuffleSplit
import joblib
from sklearn.model_selection import train_test_split



pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', None)

warnings.simplefilter(action='ignore', category=Warning)

df = pd.read_csv("booking.csv")
df.head()


################################################
# 1. Exploratory Data Analysis
################################################
def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Number uniqe #####################")
    print(dataframe.nunique())
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

check_df(df)

df = df.drop(["Booking_ID","P-C","P-not-C"],axis=1)
df['booking status'] = df['booking status'].replace({'Canceled':1,'Not_Canceled':0}).astype(int)
df["date of reservation"] = pd.to_datetime(df["date of reservation"], format="%m/%d/%Y", errors='coerce')
df = df.dropna(axis=0)
df = df[df['date of reservation'] > '2017-06-30']
df = df[df["type of meal"] != "Meal Plan 3"]

def grab_col_names(dataframe, cat_th=8, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in cat_cols.copy():
    if re.search(r"number", str(col)):
        num_cols.append(col)
        cat_cols.remove(col)

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col)

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col,plot=True)

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df,"booking status",col)

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df,"booking status",col)

for col in cat_cols:
    target_summary_with_cat(df,"average price",col)

################################################
# 2. Data Preprocessing & Feature Engineering
################################################


def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(f"total outliers of {col_name} : " + str(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0]))
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(f"total outliers of {col_name} : " + str(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0]))
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    print("##########################################")
    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

def check_outlier(dataframe, col_name, q1=0.01, q3=0.99):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name, q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    print( col, check_outlier(df, col))

for col in num_cols:
    print(col, grab_outliers(df, col))

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


for col in num_cols:
    df = remove_outlier(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))


df["new total number of customers"] = df["number of adults"] + df["number of children"]

df["new number of total nights"] = df["number of weekend nights"] + df["number of week nights"]

df["new total price"] = df["new number of total nights"] * df["average price"]

df.loc[(df['number of weekend nights'] == 0) & (df['number of week nights'] == 0), 'new night type'] = 'Same day visitor'
df.loc[(df['number of weekend nights'] == 0) & (df['number of week nights'] != 0), 'new night type'] = 'only weekdays'
df.loc[(df['number of weekend nights'] != 0) & (df['number of week nights'] == 0), 'new night type'] = 'only weekends'
df.loc[(df['number of weekend nights'] != 0) & (df['number of week nights'] != 0), 'new night type'] = 'mixed'

df.loc[(df['number of adults'] == 0) & (df['number of children'] > 0), 'new guest type'] = 'only children'
df.loc[(df['number of adults'] == 1) & (df['number of children'] == 0), 'new guest type'] = 'single'
df.loc[(df['number of adults'] == 2) & (df['number of children'] == 0), 'new guest type'] = 'couple'
df.loc[(df['number of adults'] > 0) & (df['number of children'] > 0), 'new guest type'] = 'family'
df.loc[(df['number of adults'] > 2) & (df['number of children'] == 0), 'new guest type'] = 'a group of guest'


df['new date of transaction'] = df.apply(lambda row: row['date of reservation'] - timedelta(days=row['lead time']), axis=1)

df['new reservation month'] = df['date of reservation'].dt.month_name()

bins = [-1,1, 50, 100, 200, 500]
Labels = ["Free",'Less than 50','50 - 100',"100 - 199", "200 and above"]
df['new Average Price Range'] = pd.cut(df['average price'], bins=bins,labels=Labels,right=False)


def determine_season(date):
    month = date.month
    if 3 <= month <= 5:
        return 'Spring'
    elif 6 <= month <= 8:
        return 'Summer'
    elif 9 <= month <= 11:
        return 'Autumn'
    else:
        return 'Winter'

df['new reservation season'] = df['date of reservation'].apply(determine_season)

df['new transaction month'] = df['new date of transaction'].dt.month_name()

df['new transaction season'] = df['new date of transaction'].apply(determine_season)


bins = [-1,92, 183, 274, 500]
Labels = ["0-3 months",'3-6 months','6-9 months',"over 9 months"]
df['new lead time Range'] = pd.cut(df['lead time'], bins=bins,labels=Labels,right=False)

df = df.drop(["date of reservation","new date of transaction"],axis=1)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in cat_cols.copy():
    if re.search(r"number", str(col)):
        num_cols.append(col)
        cat_cols.remove(col)

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    corr_matrix = corr.abs()
    upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc={"figure.figsize": (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

drop_list = high_correlated_cols(df[num_cols],plot=True,corr_th=0.80)

df_dropped = df.drop(drop_list,axis=1)

cat_cols, num_cols, cat_but_car = grab_col_names(df_dropped)

for col in cat_cols.copy():
    if re.search(r"number", str(col)):
        num_cols.append(col)
        cat_cols.remove(col)

df_1 =df_dropped.copy()

def label_encoder(dataframe,binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes not in [int,float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df_1, col)


def rare_analyser(dataframe,target,cat_cols): ############
    for col in cat_cols:
        print(col,":",len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT":dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts()/ len(dataframe),
                            "TARGET_MEAN":dataframe.groupby(col)[target].mean()}),end="\n\n\n")
rare_analyser(df_1, "booking status", cat_cols)
def rare_encoder(dataframe,rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O"
                    and (temp_df[col].value_counts()/ len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels),"Rare",temp_df[var])

    return temp_df
df_1 = rare_encoder(df_1, 0.01)


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first,dtype= int)
    return dataframe

ohe_cols = [col for col in df[cat_cols].columns if 12 >= df[col].nunique() > 2]

df_1 = one_hot_encoder(df_1, ohe_cols)


rs = RobustScaler()

df_1[num_cols] = rs.fit_transform(df_1[num_cols])

def high_correlated_cols(dataframe, plot=False, corr_th=0.80):
    corr = dataframe.corr()
    corr_matrix = corr.abs()
    upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc={"figure.figsize": (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(df_1)
drop_list = high_correlated_cols(df_1,plot=True,corr_th=0.80)

df_1= df_1.drop(drop_list,axis=1)

y = df_1["booking status"]
X = df_1.drop(["booking status"], axis=1)

################# RFE Recursive Feature Elimination ########################

estimator = DecisionTreeClassifier()

rfe = RFE(estimator)

param_grid = {'n_features_to_select': [20, 25, 30, 35]}

grid_search = GridSearchCV(rfe, param_grid, cv=5)

grid_search.fit(X, y)

best_n_features_to_select = grid_search.best_params_['n_features_to_select']

rfe_best = RFE(estimator, n_features_to_select=best_n_features_to_select)

rfe_best.fit(X, y)

X = rfe_best.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)

############### Feature Importance ###############################
def plot_feature_importance(importance, names, model_type):
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])

    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')


selected_models = [
    ('CART', DecisionTreeClassifier()),
    ('RF', RandomForestClassifier()),
    ("LightGBM", LGBMClassifier(verbose=-1)),
    ('CatBoost', CatBoostClassifier(verbose=False))
]

for name, model in selected_models:
    model.fit(X_train, y_train)
    if name == "CatBoost":
        importance = model.feature_importances_
    else:
        importance = model.feature_importances_

    # Selecting feature names
    names = X.columns.tolist()
    plot_feature_importance(importance, names, model_type=name)

#################################################################################################

def base_models(X, y, scoring="roc_auc"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=40)
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
            model = classifier.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            roc_auc = roc_auc_score(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            print(f"{classifier} ROC AUC: {roc_auc}")
            print(f"{classifier} accuracy: {accuracy}")
            print(f"{classifier} precision: {precision}")
            print(f"{classifier} recall: {recall}")
            print(f"{classifier} f1: {f1}")

base_models(X, y, scoring="ROC AUC")


knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

lightgbm_params = {
    "learning_rate": [0.05, 0.1],
    "n_estimators": [300, 500, 750],
    "colsample_bytree": [0.7, 1],
    "max_depth": [3, 5, 7],
    "min_child_samples": [10, 20, 30],
    "subsample": [0.7, 0.8, 0.9],
    "reg_alpha": [0, 0.1, 0.5],
    "reg_lambda": [0, 0.1, 0.5],
    "num_leaves": [15, 31, 63]
}

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}


classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(verbose=-1), lightgbm_params),
               ('CatBoost', CatBoostClassifier(verbose=False))
]

def hyperparameter_optimization(X, y, scoring="roc_auc"):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        model = classifier.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        roc_auc = roc_auc_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"{classifier} (Before): ROC AUC: {roc_auc}")
        print(f"{classifier} (Before): accuracy: {accuracy}")
        print(f"{classifier} (Before): precision: {precision}")
        print(f"{classifier} (Before): recall: {recall}")
        print(f"{classifier} (Before): f1: {f1}")

        gs_best = GridSearchCV(classifier, params, n_jobs=-1, verbose=False).fit(X, y)
        final_model = gs_best.best_estimator_

        model = final_model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        roc_auc = roc_auc_score(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        print(f"{classifier} (After): ROC AUC: {roc_auc}")
        print(f"{classifier} (After): accuracy: {accuracy}")
        print(f"{classifier} (After): precision: {precision}")
        print(f"{classifier} (After): recall: {recall}")
        print(f"{classifier} (After): f1: {f1}")

        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

best_models = hyperparameter_optimization(X, y)

def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('KNN', best_models["KNN"]),
                                              ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

voting_clf = voting_classifier(best_models, X, y)

######################### Confusion Matrix ########################################

lgbm_model1 = joblib.load("lgbm.pkl")

RocCurveDisplay.from_estimator(lgbm_model1, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

y_pred = lgbm_model1.predict(X_test)

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y,y_pred),2)
    cm =confusion_matrix(y,y_pred)
    sns.heatmap(cm,annot=True,fmt=".0f")
    plt.xlabel("Predicted")
    plt.ylabel("y")
    plt.show()

plot_confusion_matrix(y_test,y_pred)

print(classification_report(y_test, y_pred))



############## Bias Variance Tradeoff ############################################

def plot_bias_variance_tradeoff(estimator, X, y, ylim=None, cv=None,
                                n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title("Bias-Variance Tradeoff")
    if ylim is not None:
        plt.ylim(*ylim)
    else:
        plt.ylim(0.80, 1.00)  # Adjusted ylim to start from 0.80 and increase by 0.04

    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.xticks(np.arange(0, 28973, 2500))  # Adjusted x-axis ticks
    plt.yticks(np.arange(0.80, 1.01, 0.04))  # Adjusted y-axis ticks

    plt.legend(loc="best")
    return plt

cv = ShuffleSplit(n_splits=20, test_size=0.2, random_state=42)

plot_bias_variance_tradeoff(lgbm_model1, X, y, cv=cv, n_jobs=-1)
plt.show()
