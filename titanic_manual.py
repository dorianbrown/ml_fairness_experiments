from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, confusion_matrix
import pandas as pd

def CreateInput(titanic_df, train_params=[]):
    """
    Parameters for normalisation are set using the training set,
    these values are used for test preprocessing (to have stable mean, std)
    """

    # Separate label from dataset
    X = titanic_df.drop(["survived"], axis=1)
    y = titanic_df["survived"]

    # --- linear dependent information in variables ---
    # 'alive' textual label, content covered in label 'survived'
    # 'pclass' numerical class indicator, content covered by 'class'
    # 'adult_male', start with standard features, this is a combined feature
    # 'embarked', content covered by 'embarked_town'
    # 'who' is combination of sex and age (containing child, woman, man)

    # --- information about variables ---
    # sibsp - number of siblings/spouses aboard
    # parch - number of parents/children abourd

    # --- adjust variables ---
    # create categorical age variable, drop age, there are no people above 80
    X['age_cat'] = pd.cut(X.age,bins=[0, 15, 35, 60, 100], labels=['0_15','15_35','35_60','60_100'])
    # normalise ticket fare
    if len(train_params)==0:
        fare_mean = X['fare'].mean()
        fare_std = X['fare'].std()
    else:
        fare_mean = train_params[0]
        fare_std = train_params[0]

    X['fare_norm'] = (X['fare']-fare_mean)/fare_std
    print('Variables sibsp and parch are capped 3')
    X['sibsp'][X['sibsp']>=3] = 3
    X['parch'][X['parch']>=3] = 3

    # --- create input data ---
    X_input = pd.DataFrame()

    categorical_columns = ['class', 'sex', 'age_cat','parch', 'sibsp', 'deck', 'embark_town', 'alone']
    for col in categorical_columns:
        new_dummies = pd.get_dummies(X[col])
        new_dummies.columns = [col + '_' + str(c) for c in new_dummies.columns]
        # drop one category to prevent multicolinearity
        new_dummies = new_dummies.drop(new_dummies.columns[0], axis=1)

        for c in new_dummies.columns:
            X_input[c] = new_dummies[c]

    # information needed to normalise test set
    if len(train_params)==0:
        train_params.append(fare_mean)
        train_params.append(fare_std)

    X_input['fare_norm'] = X['fare_norm']
    X_input['intercept'] = 1

    return X_input, y, train_params

def ConsistentIndex(train_df, test_df):
    """ make index of test_df consistent with train_df by adding missings and dropping redundant indices """

    missing = train_df.columns.difference(test_df.columns)
    for c in missing:
        test_df[c] = 0

    redundant = test_df.columns.difference(train_df.columns)
    for c in redundant:
        test_df = test_df.drop([c], axis=1)

    # rearange order of columns in test_df
    test_df = test_df[train_df.columns]

    return train_df, test_df

def ManualPreprocess(titanic_df, train_params=[]):
    """ Preprocess dataframe, including train test split """

    train, test = train_test_split(titanic_df, test_size=0.2)
    print('Split; Train: ' + str(train.shape[0]) + ', Test: ' + str(test.shape[0]))

    X_train, y_train, train_params = CreateInput(train)
    # use information from train set normalise test set
    X_test, y_test, _ = CreateInput(test, train_params)

    X_train, X_test = ConsistentIndex(X_train, X_test)

    return X_train, y_train, X_test, y_test

def Evaluate(y, y_pred):
    print('AUROC score: %f' %roc_auc_score(y, y_pred))
    threshold = 0.5
    print('confusion matrix trheshold: %f' %threshold)
    y_pred_round = y_pred.copy()
    y_pred_round[y_pred_round>threshold] = 1
    y_pred_round[y_pred_round<threshold] = 0

    conf_mat = confusion_matrix(y, y_pred_round)
    print('TP: %i , FP: %i' %(conf_mat[1,1], conf_mat[0,1]))
    print('FN: %i , TN: %i' %(conf_mat[1,0], conf_mat[1,1]))

    
