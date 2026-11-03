import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

from load_data import load_and_clean

def preprocess_text(text_series):
    """Lowercase, strip whitespace, remove extra spaces."""
    return text_series.str.lower().str.strip().str.replace(r'\s+', ' ', regex=True)

def build_pipeline():
    """
    Create a pipeline with TF‑IDF and RandomForestClassifier.
    This automates the workflow.
    """
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    return pipeline

def train_category_classifier(df, tune_hyperparams=False):
    """
    Trains a classifier to predict Category from Description.
    Returns the best model, vectorizer (inside pipeline), and accuracy.
    """
    # Sirf expenses
    expenses = df[df['Type'] == 'Debit'].copy()
    
    # X = preprocessed descriptions
    X = preprocess_text(expenses['Description'])
    y = expenses['Category']
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Build pipeline
    pipe = build_pipeline()
    
    # Hyperparameter tuning (optional, chala ke dekh sakte ho)
    if tune_hyperparams:
        param_grid = {
            'vectorizer__ngram_range': [(1,1), (1,2)],
            'vectorizer__max_df': [0.8, 0.9],
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_depth': [None, 10, 20]
        }
        grid = GridSearchCV(pipe, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_pipe = grid.best_estimator_
        print("Best params:", grid.best_params_)
    else:
        # Default fit
        best_pipe = pipe
        best_pipe.fit(X_train, y_train)
    
    # Predictions
    y_pred = best_pipe.predict(X_test)
    
    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Categorization Accuracy: {acc*100:.2f}%")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Cross-validation score (more robust)
    cv_scores = cross_val_score(best_pipe, X, y, cv=5, scoring='accuracy')
    print(f"📈 5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
    
    return best_pipe, acc

def predict_category(description, pipeline):
    """
    Predict category with probability score.
    Returns (category, confidence).
    """
    clean_desc = preprocess_text(pd.Series([description]))
    vec = pipeline.named_steps['vectorizer'].transform(clean_desc)
    proba = pipeline.named_steps['classifier'].predict_proba(vec)[0]
    pred = pipeline.named_steps['classifier'].classes_[np.argmax(proba)]
    confidence = np.max(proba)
    return pred, confidence

if __name__ == "__main__":
    # Data load
    df, expenses, income = load_and_clean("pakistan_statement_30k_enhanced.csv")
    
    # Train model (set tune_hyperparams=True to search best params)
    model_pipe, accuracy = train_category_classifier(df, tune_hyperparams=False)
    
    # Test on new descriptions
    new_data = [
        "KFC Phase 6 DHA",
        "Daraz.pk Karachi",
        "Shell Petrol Pump",
        "Sui Southern Gas bill",
        "Careem Ride",
        "FoodPanda karachi"
    ]
    
    print("\n🔮 New Transactions Predictions:")
    for desc in new_data:
        cat, conf = predict_category(desc, model_pipe)
        print(f" '{desc}' → {cat} (Confidence: {conf:.2f})")
