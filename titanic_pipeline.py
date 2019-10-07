#!/usr/bin/env python

from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC


class PipelineDebugger(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def transform(self, df, y=None):
        print(f"Shape: {df.shape}")
        return df
    
    def fit(self, df, y=None):
        """Pass"""
        return self


class OneHotEncoding(BaseEstimator, TransformerMixin):
    """Takes in dataframe and give one hot encoding for categorical features """

    def __init__(self, column_names=[]):
        self.column_names = column_names

    def transform(self, df, y=None):
        """transform a categorical feature into one-hot-encoding"""
        return pd.get_dummies(df, columns=self.column_names)

    def fit(self, df, y=None):
        """Pass"""
        return self


class ColumnExtractor(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts a columns as feture """

    def __init__(self, column_names=[]):
        self.column_names = column_names

    def transform(self, df, y=None):
        """Return the columns"""
        return df.loc[:, self.column_names]

    def fit(self, df, y=None):
        """Pass"""
        return self


class SexBinarizer(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts a columns as feture """

    def __init__(self, column_names=[]):
        pass

    def transform(self, df, y=None):
        """female maps to 0 and male maps to 1"""
        df.loc[:, "sex"] = df.loc[:, "sex"].map({"male": 0, "female": 1})
        return df

    def fit(self, df, y=None):
        """pass"""
        return self    


class FeatureNormalizer(BaseEstimator, TransformerMixin):
    """Takes in dataframe, extracts a columns as feture """

    def __init__(self, column_names=[]):
        self.column_names = column_names
        self.min_max_scalar = MinMaxScaler()

    def transform(self, df, y=None):
        """Min Max Scalar"""
        df.loc[:, self.column_names] = self.min_max_scalar.transform(df[self.column_names].as_matrix())
        return df

    def fit(self, df, y=None):
        """FItting Min Max Scalar"""
        self.min_max_scalar.fit(df[self.column_names].as_matrix())
        return self


class FillNa(BaseEstimator, TransformerMixin):
    """Takes in dataframe, fill NaN values in a given columns """

    def __init__(self, method="mean"):
        self.method = method

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        if self.method == "zeros":
            df.fillna(0)
        elif self.method == "mean":
            df.fillna(df.mean(), inplace=True)
        elif self.method == "median":
            df.fillna(df.median(), inplace=True)
        else:
            raise ValueError("Method should be 'mean' or 'zeros'")
        return df

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self


class AddTwoCategoricalVariables(BaseEstimator, TransformerMixin):
    def __init__(self, column_1, column_2):
        self.column_1 = column_1
        self.column_2 = column_2
    
    def transform(self, df):
        df[self.column_1 + "_" + self.column_2] = (df[self.column_1].astype(float) + 
                                                (len(df[self.column_1].unique()) * 
                                                (df[self.column_2].astype(float)))).astype("category")
        return df
    
    def fit(self, df, y=None):
        return self

    
class Numerical2Categorical(BaseEstimator, TransformerMixin):
    def __init__(self, column, ranges, labels):
        self.column = column
        self.ranges = ranges
        self.labels = labels
        
    def transform(self, df):
        df.loc[:, self.column + "_" + "cat"] = (pd
                                                .cut(df.loc[:, self.column], 
                                                     self.ranges, labels=self.labels))
        return df
    
    def fit(self, df, y=None):
        return self
    

def evaluate_model(X_test, y_test, clf, threshold=0.5):
    print(f"Evaluating with threshold: {threshold}")
    y_pred = clf.predict_proba(X_test)[:, 1] > threshold
    # print AUC
    print(f"AUC: {roc_auc_score(y_test, clf.predict_proba(X_test)[:,1]):.3f}")
    # print Accuracy
    print(f"TN: {sum(~y_test & ~y_pred)} FP: {sum(~y_test & y_pred)}")
    print(f"FN: {sum(y_test & ~y_pred)} TP: {sum(y_test & y_pred)}")

    # plot distributions of probabilities
    probs = clf.predict_proba(X_test)[:, 0]
    sns.distplot(probs[y_test == 0], color='g', norm_hist=False)
    sns.distplot(probs[y_test == 1], color='r', norm_hist=False)
    plt.axvline(x=threshold, color="orange")
    plt.show()
    pass

def preprocess_data(titanic_df):

    categorical_columns = ["class", "sex", "embarked", "who", "alone"]

    for cat in categorical_columns:
        titanic_df[cat] = titanic_df[cat].astype('category')

    X = titanic_df.drop(["survived"], axis=1)
    y = titanic_df["survived"]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

    print(f"Features: {len(X_train.columns)}")
    print(f"Train/test size: {len(X_train), len(X_test)}")
    
    return(X_train, X_test, y_train, y_test)

def create_classifier():
    feature_columns = ["class", "sex", "age", "sibsp", "parch", "fare", "embarked", "who", "alone"]
    normalize_features = ["fare", "sibsp", "parch"]
    categorical_columns = ["class", "sex", "embarked", "who", "alone"]

    age_range = [0, 15, 35, 50, 80]
    age_label = [0, 1, 2, 3]

    pipeline = Pipeline([
                ("column_extractor", ColumnExtractor(feature_columns)),
                ("ohe", OneHotEncoding(categorical_columns)),
                ("imputer", SimpleImputer()),
                ("rfc", RandomForestClassifier()),
    ])
    
    return pipeline