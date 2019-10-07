def eqaualised_odds(X, y_true, clf, sens_attr):
    pred = clf.predict(X)
    for val in X[sens_attr].unique():
        p_ind = (X[sens_attr] == val) & (y_true == 1)
        n_ind = (X[sens_attr] == val) & (y_true == 0)
        tpr = pred[p_ind].sum()/p_ind.sum()
        tnr = (1 - pred[n_ind]).sum()/n_ind.sum()
        print(f"TP rate for {val}: {tpr:.2f}")
        print(f"TN rate for {val}: {tnr:.2f}")

def calibration(X, y_true, clf, sens_attr):
    pred = clf.predict(X)
    print("# POSITIVE OUTCOME")
    for val in X[sens_attr].unique():
        val_ind = (pred == 1) & (X[sens_attr] == val)
        frac = y_true[val_ind].sum()/val_ind.sum()
        print(f"Fraction of {val} that were predicted as 1 that are actually 1: {frac:.2f}")
    
    print("\n# NEGATIVE OUTCOME")
    for val in X[sens_attr].unique():
        val_ind = (pred == 0) & (X[sens_attr] == val)
        frac = (1 - y_true[val_ind]).sum()/val_ind.sum()
        print(f"Fraction of {val} that were predicted as 0 that are actually 0: {frac:.2f}")

def demographic_parity(X, clf, sens_attr):
    pred = clf.predict(X)
    for val in X[sens_attr].unique():
        val_ind = (X[sens_attr] == val)
        frac = pred[val_ind].sum()/val_ind.sum()
        print(f"Fraction of {val} that were predicted 1: {frac:.2f}")