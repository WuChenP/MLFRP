from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from sklearn.svm import SVR
import processing
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor  # sklearn-style interface
import os
from sklearn.metrics import mean_squared_error


# -------------------------- 1. Core Auxiliary Functions --------------------------
def get_feature_importance(model, model_name, feature_names):
    """Calculate feature importance, adapted to different model types"""
    try:
        if model_name == 'SVR':
            return np.abs(model.coef_[0]) if hasattr(model, 'coef_') else None
        elif model_name == 'Linear Regression':
            return np.abs(model.coef_)
        elif model_name == 'XGB' and isinstance(model, XGBRegressor):
            return model.feature_importances_
        elif hasattr(model, 'feature_importances_'):
            return model.feature_importances_
    except Exception as e:
        print(f"  {model_name} feature importance calculation failed: {str(e)}")
        return None


def get_sample_size_range(X_train):
    """Generate training sample size sequence (starting from 40 to avoid invalid training with small samples)"""
    total_samples = len(X_train)
    sample_sizes = list(range(40, total_samples + 1, 5))
    return sample_sizes if sample_sizes else [total_samples]


# -------------------------- 2. Main Training Process --------------------------
if __name__ == "__main__":
    # Configuration parameters
    random_state = 42
    data_path = '../../data/Data for Machine Learning.xlsx'
    split_data_save_dir = './split_data'  # Directory for saving train/test sets
    model_save_dir = './model'
    correct_columns = ['C_i/C_m((g/L)/M)', 'C_ci/C_m((g/L)/M)', 'C_t/C_m((g/L)/M)', 'C_m (M)', 'T (°C)',
                       'AMPS feed ratio (mol%)', 'C_c (mg/L)', '[η] (mL/g)']

    # Create model saving directory
    os.makedirs(model_save_dir, exist_ok=True)
    print(f" Models will be saved to: {model_save_dir}")

    # Create split data saving directory
    os.makedirs(split_data_save_dir, exist_ok=True)
    print(f" Split train/test sets will be saved to: {split_data_save_dir}")

    # -------------------------- Step 1: Load and validate data --------------------------
    try:
        data = pd.read_excel(data_path, skiprows=[0], names=correct_columns)
        print(f"\n Data loaded successfully, total {data.shape[0]} samples, {data.shape[1]} columns")
        print("First 5 rows of data preview:")
        print(data[correct_columns].head(5))
    except Exception as e:
        print(f" Data loading failed: {str(e)}")
        exit()

    # -------------------------- Step 2: Split train/test sets first --------------------------
    try:
        train_data, test_data = processing.split_data(
            data, train_size=0.8, random_state=random_state
        )
        print(f"\n Data splitting completed:")
        print(f"  - Training set: {train_data.shape[0]} samples")
        print(f"  - Test set: {test_data.shape[0]} samples")

        # -------------------------- Save split train and test sets --------------------------
        # Save as CSV
        train_csv_path = f"{split_data_save_dir}/train_data_randomstate_{random_state}.csv"
        test_csv_path = f"{split_data_save_dir}/test_data_randomstate_{random_state}.csv"
        train_data.to_csv(train_csv_path, index=True, encoding="utf-8")
        test_data.to_csv(test_csv_path, index=True, encoding="utf-8")

        print(f"  Training set saved to: \n      - {train_csv_path}")
        print(f"  Test set saved to: \n      - {test_csv_path}")
    except Exception as e:
        print(f" Data splitting failed: {str(e)}")
        exit()

    # -------------------------- Step 3: Target variable processing (log transformation only) --------------------------
    y_train = np.log(train_data['[η] (mL/g)'] + 1e-9)
    y_test = np.log(test_data['[η] (mL/g)'] + 1e-9)
    print(f"\n Target variable processing completed")

    # -------------------------- Step 4: Feature processing (MinMaxScaler) --------------------------
    X_train = train_data.drop('[η] (mL/g)', axis=1)
    X_test = test_data.drop('[η] (mL/g)', axis=1)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f" Feature scaling completed (MinMaxScaler)")

    # -------------------------- Step 5: Feature selection --------------------------
    try:
        X_train_selected, X_test_selected = processing.select_features(
            X_train_scaled, y_train, X_test_scaled, n_features=7
        )
        print(f" Feature selection completed, retained {X_train_selected.shape[1]} features")
    except Exception as e:
        print(f" Feature selection failed: {str(e)}")
        exit()

    # -------------------------- Step 6: Format unification --------------------------
    X_train_selected = pd.DataFrame(
        X_train_selected, columns=X_train.columns, index=train_data.index
    )
    X_test_selected = pd.DataFrame(
        X_test_selected, columns=X_train.columns, index=test_data.index
    )
    y_train = pd.Series(y_train, name='log_[η] (mL/g)', index=train_data.index)
    y_test = pd.Series(y_test, name='log_[η] (mL/g)', index=test_data.index)

    # -------------------------- Step 7: Save training statistics --------------------------
    training_stats = {
        'feature_names': X_train.columns.tolist(),
        'scaler_min': scaler.data_min_.tolist(),
        'scaler_max': scaler.data_max_.tolist(),
        'target_epsilon': 1e-9
    }
    joblib.dump(training_stats, f"{model_save_dir}/training_stats.pkl")
    print(f"\n Training statistics saved to: {model_save_dir}/training_stats.pkl")

    # -------------------------- Step 8: Define models --------------------------
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=random_state),
        'Decision Tree': DecisionTreeRegressor(random_state=random_state),
        'GBT': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
        'XGB': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=random_state),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'Linear Regression': LinearRegression(n_jobs=-1)
    }
    print(f"\n Models to evaluate: {list(models.keys())}")

    model_performance = {}

    # -------------------------- Step 9: Train and evaluate models --------------------------
    for model_name, model in models.items():
        print(f"\n{'='*70}")
        print(f" Evaluating model: {model_name}")
        print(f"{'='*70}")

        sample_sizes = get_sample_size_range(X_train_selected)
        print(f"  Sample size sequence: {sample_sizes} (total {len(sample_sizes)} points)")

        train_r2_list = []
        val_r2_list = []
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

        # Train with increasing sample sizes
        for sample_size in sample_sizes:
            # Use np.random.seed instead of random_state parameter to ensure consistent results across runs
            np.random.seed(random_state + sample_size)  # +sample_size to avoid duplicate random results for different sample sizes
            sample_indices = np.random.choice(
                len(X_train_selected), size=sample_size, replace=False  # Remove random_state parameter
            )
            X_sample = X_train_selected.iloc[sample_indices]
            y_sample = y_train.iloc[sample_indices]

            # Train and evaluate
            model.fit(X_sample, y_sample)
            train_r2 = round(model.score(X_sample, y_sample), 4)
            val_r2 = round(np.mean(cross_val_score(model, X_sample, y_sample, cv=kf, scoring='r2')), 4)

            train_r2_list.append(train_r2)
            val_r2_list.append(val_r2)

            print(f"  Sample size={sample_size:4d} | Train R²={train_r2:6.4f} | Validation R²={val_r2:6.4f}")

        # Cross-validation with full dataset
        full_cv_r2 = cross_val_score(model, X_train_selected, y_train, cv=kf, scoring='r2')
        full_cv_mean = round(np.mean(full_cv_r2), 4)
        full_cv_std = round(np.std(full_cv_r2), 4)
        print(f"\n  5-fold CV with full dataset:")
        print(f"    R² scores: {np.round(full_cv_r2, 4)}")
        print(f"    Mean R²: {full_cv_mean} (±{full_cv_std})")

        # Test set evaluation
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)
        test_r2 = round(model.score(X_test_selected, y_test), 4)
        test_mse = round(mean_squared_error(y_test, y_pred), 4)
        print(f"  Test set performance:")
        print(f"    R²: {test_r2} | MSE: {test_mse}")

        # Feature importance
        feature_importance = get_feature_importance(model, model_name, X_train.columns.tolist())
        if feature_importance is not None:
            # Normalize feature importance for Linear Regression/SVR
            if model_name in ['Linear Regression', 'SVR']:
                feature_importance = feature_importance / np.sum(feature_importance)
            # Print all features
            importance_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': np.round(feature_importance, 4)
            }).sort_values('Importance', ascending=False)
            print(f"\n  All feature importance (sorted by importance):")
            print(importance_df.to_string(index=False))  # Print all features without index


        # Save performance
        model_performance[model_name] = {
            'model': model,
            'cv_mean_r2': full_cv_mean,
            'cv_std_r2': full_cv_std,
            'test_r2': test_r2,
            'test_mse': test_mse,
            'feature_importance': feature_importance
        }

    # -------------------------- Step 10: Save models (sorted by R²) --------------------------
    sorted_models = sorted(
        model_performance.items(), key=lambda x: x[1]['test_r2'], reverse=True
    )

    print(f"\n{'='*70}")
    print(f" Model performance ranking (by test set R²)")
    print(f"{'='*70}")

    for rank, (model_name, info) in enumerate(sorted_models, 1):
        model_path = f"{model_save_dir}/{rank}_{model_name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(info['model'], model_path)

        print(f"  Rank {rank}: {model_name}")
        print(f"    - Test set R²: {info['test_r2']} | MSE: {info['test_mse']}")
        print(f"    - Cross-validation R²: {info['cv_mean_r2']} (±{info['cv_std_r2']})")
        print(f"    - Model saved to: {model_path}")
        print()

    # -------------------------- Step 11: Save feature information --------------------------
    feature_info = {
        'feature_names': X_train.columns.tolist(),
        'scaler': scaler
    }
    joblib.dump(feature_info, f"{model_save_dir}/feature_info.pkl")
    print(f" Feature information saved to: {model_save_dir}/feature_info.pkl")

    # -------------------------- Step 12: Print summary table --------------------------
    print(f"\n{'='*90}")
    print(f" Model Performance Summary Table")
    print(f"{'='*90}")
    summary_data = []
    for rank, (model_name, info) in enumerate(sorted_models, 1):
        summary_data.append({
            'Rank': rank,
            'Model': model_name,
            'CV Mean R²': info['cv_mean_r2'],
            'CV Std Dev': info['cv_std_r2'],
            'Test Set R²': info['test_r2'],
            'Test Set MSE': info['test_mse']
        })
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print(f"{'='*90}")

    print(f"\n All models trained successfully!")