import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import  GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import re
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier


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

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    # print(f"Observations: {dataframe.shape[0]}")
    # print(f"Variables: {dataframe.shape[1]}")
    # print(f'cat_cols: {len(cat_cols)}')
    # print(f'num_cols: {len(num_cols)}')
    # print(f'cat_but_car: {len(cat_but_car)}')
    # print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def high_correlated_cols(dataframe, plot=False, corr_th=0.75):
    corr = dataframe.corr()
    corr_matrix = corr.abs()
    upper_triangle_matrix = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool_))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc={"figure.figsize": (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

def remove_outlier(dataframe, num_cols):
    low_limit, up_limit = outlier_thresholds(dataframe, num_cols)
    df_without_outliers = dataframe[~((dataframe[num_cols] < low_limit) | (dataframe[num_cols] > up_limit))]
    return df_without_outliers

def rare_encoder(dataframe, rare_perc):
        temp_df = dataframe.copy()

        rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O"
                        and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

        for var in rare_columns:
            tmp = temp_df[var].value_counts() / len(temp_df)
            rare_labels = tmp[tmp < rare_perc].index
            temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])

        return temp_df
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

def label_encoder(dataframe,binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

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

def booking_data_prep(dataframe):

    dataframe = dataframe.drop(["Booking_ID","P-C", "P-not-C"], axis=1)
    dataframe['booking status'] = dataframe['booking status'].replace({'Canceled': 1, 'Not_Canceled': 0}).astype(int)
    dataframe["date of reservation"] = pd.to_datetime(dataframe["date of reservation"], format="%m/%d/%Y",
                                                      errors='coerce')
    dataframe = dataframe.dropna(axis=0)
    dataframe = dataframe[dataframe['date of reservation'] > '2017-06-30']
    dataframe = dataframe[dataframe["type of meal"] != "Meal Plan 3"]

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    for col in cat_cols.copy():
        if re.search(r"number", str(col)):
            num_cols.append(col)
            cat_cols.remove(col)


    for col in num_cols:
        dataframe = remove_outlier(dataframe, col)

    dataframe["total number of customers"] = dataframe["number of adults"] + dataframe["number of children"]

    dataframe["number of total nights"] = dataframe["number of weekend nights"] + dataframe["number of week nights"]

    dataframe["total price"] = dataframe["number of total nights"] * dataframe["average price"]

    dataframe.loc[(dataframe['number of adults'] == 0) & (dataframe['number of children'] > 0), 'guest type'] = 'only children'
    dataframe.loc[(dataframe['number of adults'] == 1) & (dataframe['number of children'] == 0), 'guest type'] = 'single'
    dataframe.loc[(dataframe['number of adults'] == 2) & (dataframe['number of children'] == 0), 'guest type'] = 'couple'
    dataframe.loc[(dataframe['number of adults'] > 0) & (dataframe['number of children'] > 0), 'guest type'] = 'family'
    dataframe.loc[(dataframe['number of adults'] > 2) & (dataframe['number of children'] == 0), 'guest type'] = 'a group of guest'

    dataframe.loc[(dataframe['number of weekend nights'] == 0) & (dataframe['number of week nights'] == 0), 'night type'] = 'daily'
    dataframe.loc[(dataframe['number of weekend nights'] == 0) & (dataframe['number of week nights'] != 0), 'night type'] = 'only weekdays'
    dataframe.loc[(dataframe['number of weekend nights'] != 0) & (dataframe['number of week nights'] == 0), 'night type'] = 'only weekends'
    dataframe.loc[(dataframe['number of weekend nights'] != 0) & (dataframe['number of week nights'] != 0), 'night type'] = 'mixed'

    dataframe['date of transaction'] = dataframe.apply(lambda row: row['date of reservation'] - timedelta(days=row['lead time']),
                                                       axis=1)

    dataframe['reservation month'] = dataframe['date of reservation'].dt.month_name()
    dataframe['reservation season'] = dataframe['date of reservation'].apply(determine_season)

    dataframe['transaction month'] = dataframe['date of transaction'].dt.month_name()
    dataframe['transaction season'] = dataframe['date of transaction'].apply(determine_season)

    bins = [-1, 1, 50, 100, 200, 500]
    Labels = ["Free", 'Less than 50', '50 - 100', "100 - 199", "200 and above"]
    dataframe['Average Price Range'] = pd.cut(dataframe['average price'], bins=bins, labels=Labels, right=False)

    bins = [-1, 92, 183, 274, 500]
    Labels = ["0-3 months", '3-6 months', '6-9 months', "over 9 months"]
    dataframe['lead time Range'] = pd.cut(dataframe['lead time'], bins=bins, labels=Labels, right=False)

    dataframe = dataframe.drop(["date of reservation", "date of transaction"], axis=1)

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    for col in cat_cols.copy():
        if re.search(r"number", str(col)):
            num_cols.append(col)
            cat_cols.remove(col)

    drop_list = high_correlated_cols(dataframe[num_cols])
    dataframe = dataframe.drop(drop_list, axis=1)

    dataframe = rare_encoder(dataframe, 0.01)

    binary_cols = [col for col in dataframe if dataframe[col].dtypes not in [int, float]
                   and dataframe[col].nunique() == 2]

    for col in binary_cols:
        label_encoder(dataframe, col)

    ohe_cols = [col for col in dataframe[cat_cols].columns if 12 >= dataframe[col].nunique() > 2]
    dataframe = one_hot_encoder(dataframe, ohe_cols)

    rs = RobustScaler()

    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)

    for col in cat_cols.copy():
        if re.search(r"number", str(col)):
            num_cols.append(col)
            cat_cols.remove(col)

    dataframe[num_cols] = rs.fit_transform(dataframe[num_cols])


    y = dataframe["booking status"]
    X = dataframe.drop(["booking status"], axis=1)

    estimator = DecisionTreeClassifier()

    rfe = RFE(estimator)

    param_grid = {'n_features_to_select': [20,25, 30, 35]}

    grid_search = GridSearchCV(rfe, param_grid, cv=5)

    grid_search.fit(X, y)

    best_n_features_to_select = grid_search.best_params_['n_features_to_select']

    rfe_best = RFE(estimator, n_features_to_select=best_n_features_to_select)

    rfe_best.fit(X, y)

    X = rfe_best.transform(X)

    return X, y

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



knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 1]}

catboost_params = {"iterations": [200, 500],
                   "learning_rate": [0.01, 0.1],
                   "depth": [3, 6]}


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



classifiers = [('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ("CatBoost", CatBoostClassifier(verbose=False), catboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]

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

def lgbm_model_save(best_models):
    lgbm_model = best_models["LightGBM"]
    return lgbm_model


################################################
# Pipeline Main Function
################################################


def main():
    df = pd.read_csv("booking.csv")
    X, y = booking_data_prep(df)
    base_models(X, y)
    best_models = hyperparameter_optimization(X, y)
    model = lgbm_model_save(best_models)
    joblib.dump(model, "lgbm.pkl")
    return model

if __name__ == "__main__":
    print("İşlem başladı")
    main()

