from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

class BaselineClassifier:
    """
    Wrapper for non-neural baseline classifiers.
    """
    def __init__(self, model_type='random_forest', random_state=42, **kwargs):
        self.model_type = model_type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=random_state, **kwargs)
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=random_state, max_iter=1000, **kwargs)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
            
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
        
    def predict(self, X):
        return self.model.predict(X)
        
    def predict_proba(self, X):
        return self.model.predict_proba(X)
        
    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, output_dict=True)
        return {
            'accuracy': acc,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'report': report
        }

def train_baseline_rf(X_train, y_train, X_val, y_val, X_test, y_test, random_state=42):
    """
    Helper function to fit and evaluate a Random Forest baseline.
    """
    clf = BaselineClassifier(model_type='random_forest', random_state=random_state)
    clf.fit(X_train, y_train)
    
    # Train / Val Acc
    train_preds = clf.predict(X_train)
    val_preds = clf.predict(X_val)
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    
    # Evaluate
    results = clf.evaluate(X_test, y_test)
    
    print(f"Random Forest - Train Accuracy: {train_acc:.4f}")
    print(f"Random Forest - Val Accuracy: {val_acc:.4f}")
    print(f"Random Forest - Test Accuracy: {results['accuracy']:.4f}")
    
    return clf.model, results['report']
