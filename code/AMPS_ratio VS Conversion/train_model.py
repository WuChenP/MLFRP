from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
import numpy as np
from sklearn.svm import SVR
import processing
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
import os
from sklearn.metrics import mean_squared_error, r2_score

# -------------------------- 1. Core Auxiliary Functions --------------------------
def get_feature_importance(model, model_name, feature_names):
    """Multi-output feature importance (averaged over outputs if needed)"""
    try:
        if hasattr(model, 'estimators_'):
            importances = []
            for est in model.estimators_:
                if hasattr(est, 'feature_importances_'):
                    importances.append(est.feature_importances_)
                elif hasattr(est, 'coef_'):
                    importances.append(np.abs(est.coef_))
            return np.mean(importances, axis=0)
        elif hasattr(model, 'feature_importances_'):
            return model.feature_importances_
        elif hasattr(model, 'coef_'):
            return np.abs(model.coef_)
    except Exception as e:
        print(f"  {model_name} feature importance calculation failed: {str(e)}")
    return None


def get_sample_size_range(X_train):
    total_samples = len(X_train)
    sample_sizes = list(range(40, total_samples + 1, 5))
    return sample_sizes if sample_sizes else [total_samples]


# -------------------------- 2. Main Training Process --------------------------
if __name__ == "__main__":

    random_state = 55
    data_path = '../../data/Data for Machine Learning.xlsx'
    split_data_save_dir = './split_data'
    model_save_dir = './model'

    correct_columns = [
        'C_i/C_m((g/L)/M)',
        'C_ci/C_m((g/L)/M)',
        'C_t/C_m((g/L)/M)',
        'C_m (M)',
        'T (°C)',
        'AMPS feed ratio (mol%)',
        'C_c (mg/L)',
        'AMPS ratio (mol%)',
        'conversion（%）'
    ]

    target_cols = ['AMPS ratio (mol%)', 'conversion（%）']

    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(split_data_save_dir, exist_ok=True)

    # -------------------------- Step 1: Load data --------------------------
    data = pd.read_excel(
        data_path,
        sheet_name='AMPS_ratio VS Conversion',
        skiprows=[0],
        names=correct_columns
    )

    # -------------------------- Step 2: Split data --------------------------
    train_data, test_data = processing.split_data(
        data, train_size=0.8, random_state=random_state
    )

    train_data.to_csv(
        f"{split_data_save_dir}/train_data_randomstate_{random_state}.csv"
    )
    test_data.to_csv(
        f"{split_data_save_dir}/test_data_randomstate_{random_state}.csv"
    )

    # -------------------------- Step 3: Target  --------------------------
    y_train = train_data[target_cols].copy()
    y_test = test_data[target_cols].copy()

    # -------------------------- Step 4: Features --------------------------
    X_train = train_data.drop(columns=target_cols)
    X_test = test_data.drop(columns=target_cols)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -------------------------- Step 5: Feature selection--------------------------
    #  你的 select_features 不支持多输出，这里保持所有特征
    X_train_selected = X_train_scaled
    X_test_selected = X_test_scaled

    X_train_selected = pd.DataFrame(
        X_train_selected, columns=X_train.columns, index=train_data.index
    )
    X_test_selected = pd.DataFrame(
        X_test_selected, columns=X_train.columns, index=test_data.index
    )

    # -------------------------- Step 6: Save training stats --------------------------
    training_stats = {
        'feature_names': X_train.columns.tolist(),
        'scaler_min': scaler.data_min_.tolist(),
        'scaler_max': scaler.data_max_.tolist(),
        'target_names': target_cols
    }
    joblib.dump(training_stats, f"{model_save_dir}/training_stats.pkl")

    # -------------------------- Step 7: Models (Multi-output) --------------------------
    models = {
        'Random Forest': MultiOutputRegressor(
            RandomForestRegressor(n_estimators=50, random_state=random_state)
        ),

        'Decision Tree': MultiOutputRegressor(
            DecisionTreeRegressor(random_state=random_state)
        ),

        'GBT': MultiOutputRegressor(
            GradientBoostingRegressor(n_estimators=100, random_state=random_state)
        ),

        'XGB': MultiOutputRegressor(
            XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=random_state)
        ),

        'SVR': MultiOutputRegressor(
            GridSearchCV(
                SVR(),
                param_grid={
                    'C': [0.1, 1, 10, 50],
                    'gamma': ['scale', 'auto'],
                    'epsilon': [0.01, 0.1],
                    'kernel': ['rbf']
                },
                cv=5,
                scoring='r2',
                n_jobs=-1
            )
        ),

        'Linear Regression': MultiOutputRegressor(
            LinearRegression()
        )
    }

    model_performance = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # -------------------------- Step 8: Training & Evaluation --------------------------
    for model_name, model in models.items():

        print(f"\n{'='*70}")
        print(f" Evaluating model: {model_name}")
        print(f"{'='*70}")

        sample_sizes = get_sample_size_range(X_train_selected)

        for sample_size in sample_sizes:
            np.random.seed(random_state + sample_size)
            idx = np.random.choice(
                len(X_train_selected), size=sample_size, replace=False
            )

            X_sample = X_train_selected.iloc[idx]
            y_sample = y_train.iloc[idx]

            model.fit(X_sample, y_sample)

            train_pred = model.predict(X_sample)
            train_r2 = r2_score(y_sample, train_pred, multioutput='uniform_average')

            val_r2 = np.mean(
                cross_val_score(
                    model, X_sample, y_sample,
                    cv=kf, scoring='r2'
                )
            )

            print(
                f"  Sample size={sample_size:4d} | "
                f"Train R²={train_r2:.4f} | Val R²={val_r2:.4f}"
            )

        # -------- Test set --------
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)

        test_r2 = r2_score(y_test, y_pred, multioutput='raw_values')
        test_mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')

        print("\n  Test set performance:")
        for i, col in enumerate(target_cols):
            print(
                f"    {col}: R²={test_r2[i]:.4f} | MSE={test_mse[i]:.4f}"
            )

        model_performance[model_name] = {
            'model': model,
            'test_r2': dict(zip(target_cols, test_r2)),
            'test_mse': dict(zip(target_cols, test_mse))
        }

    # -------------------------- Step 9: Save models --------------------------
    for i, (model_name, info) in enumerate(model_performance.items(), 1):
        model_path = f"{model_save_dir}/{i}_{model_name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(info['model'], model_path)

    joblib.dump(
        {'feature_names': X_train.columns.tolist(), 'scaler': scaler},
        f"{model_save_dir}/feature_info.pkl"
    )

    print("\n All multi-output models trained successfully!")
