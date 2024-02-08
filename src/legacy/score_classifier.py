from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score
import pickle

def print_scores(model, X_train, X_test, y_train, y_test):

    '''
    Produces classification scores for a given model and train/test dataset.
    '''
    
    # with open(f'../models/{model}.pkl', 'rb') as file:  
    #          model = pickle.load(file)

    # get scores and probabilities
    cv = cross_val_score(model, X_train, y_train, cv=3).mean()
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    probs = model.predict_proba(X_test)
    pred = model.predict(X_test)
    f1 = f1_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred) 
    probs_pos = probs[:, 1]
    roc_auc = roc_auc_score(y_test, probs_pos)

    print(f'cv: {round(cv,4)}')
    print(f'train: {round(train_score,4)}')
    print(f'test: {round(test_score,4)}')
    print(f'roc_auc: {round(roc_auc,4)}')
    print(f'precision: {round(precision,4)}')
    print(f'recall: {round(recall,4)}')
    print(f'f1: {round(f1,4)}')

    return None