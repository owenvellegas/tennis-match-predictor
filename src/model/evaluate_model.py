from sklearn.metrics import accuracy_score, precision_score, brier_score_loss, log_loss, f1_score
from sklearn.model_selection import cross_val_score, KFold
import numpy as np

def evaluate_model(
    model,
    X, y,
    X_test, y_test,
    X_train, y_train,
):
    
    test_pred = model.predict(X_test)
    train_pred = model.predict(X_train)
    test_acc = accuracy_score(y_test, test_pred)
    train_acc = accuracy_score(y_train, train_pred)

    print(f"\n----- Results -----")
    print(f"Test Accuracy:   {test_acc  * 100:.2f}%")
    print(f"Train Accuracy:  {train_acc * 100:.2f}%")
    
    kf = KFold(n_splits=8, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    print(f"\n8-fold CV Mean Accuracy: {np.mean(scores) * 100:.2f}%")