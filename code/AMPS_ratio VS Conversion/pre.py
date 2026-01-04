import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
from glob import glob


# ========================= 基本配置 =========================
TARGET_COLS = ['AMPS ratio (mol%)', 'conversion（%）']


def read_pred_data(file_path, expected_columns):
    try:
        df = pd.read_excel(
            file_path,
            sheet_name='AMPS_ratio VS Conversion',
            skiprows=[0],
            names=expected_columns
        )
        print(f" Prediction data loaded successfully: {df.shape}")
        print(df.head(3))
        return df
    except Exception as e:
        print(f" Failed to read prediction data: {str(e)}")
        return None


def load_training_assets(stats_path, feature_info_path):
    try:
        training_stats = joblib.load(stats_path)
        feature_names = training_stats['feature_names']

        feature_info = joblib.load(feature_info_path)
        scaler = feature_info['scaler']

        print("\n Training assets loaded:")
        print(" Features:", feature_names)
        print(" Targets:", training_stats.get('target_names', TARGET_COLS))

        return scaler, feature_names
    except Exception as e:
        print(f" Failed to load training assets: {str(e)}")
        return None, None


def preprocess_pred_data(df, feature_names, scaler):
    X = df[feature_names].copy()

    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mean())

    X_scaled = scaler.transform(X)
    return pd.DataFrame(X_scaled, columns=feature_names, index=df.index)


def get_all_model_paths(model_dir):
    paths = glob(os.path.join(model_dir, "*_model.pkl"))
    if not paths:
        print(" No models found.")
        return None

    print("\n Models found:")
    for p in paths:
        print(" ", os.path.basename(p))
    return paths


def predict_with_model(model_path, X_scaled):
    try:
        model = joblib.load(model_path)
        model_name = os.path.basename(model_path).split("_model.pkl")[0].split("_", 1)[1]

        y_pred = model.predict(X_scaled)
        y_pred = np.maximum(y_pred, 0)  # 非负约束

        print(f" {model_name} prediction done, shape={y_pred.shape}")
        return model_name, y_pred
    except Exception as e:
        print(f" Prediction failed for {model_path}: {str(e)}")
        return None, None


def evaluate_and_save_all(df, all_predictions, output_path):
    result_df = df.copy()

    metrics_summary = []

    for model_name, y_pred in all_predictions.items():
        for i, target in enumerate(TARGET_COLS):
            pred_col = f'predicted_{target}_{model_name}'
            result_df[pred_col] = np.round(y_pred[:, i], 3)

            if target in result_df.columns:
                abs_err = result_df[target] - result_df[pred_col]
                result_df[f'abs_error_{target}_{model_name}'] = np.round(abs_err, 3)

    print(f"\n{'=' * 80}")
    print(" Evaluation metrics (per target)")
    print(f"{'=' * 80}")

    for model_name, y_pred in all_predictions.items():
        for i, target in enumerate(TARGET_COLS):
            if target not in df.columns:
                continue

            y_true = df[target]
            y_hat = y_pred[:, i]

            mask = y_true.notna()
            r2 = r2_score(y_true[mask], y_hat[mask])
            mae = mean_absolute_error(y_true[mask], y_hat[mask])
            rmse = np.sqrt(mean_squared_error(y_true[mask], y_hat[mask]))

            metrics_summary.append({
                'Model': model_name,
                'Target': target,
                'R2': round(r2, 4),
                'MAE': round(mae, 3),
                'RMSE': round(rmse, 3)
            })

            print(f"{model_name} | {target}")
            print(f"  R²={r2:.4f} | MAE={mae:.3f} | RMSE={rmse:.3f}")

    summary_df = pd.DataFrame(metrics_summary)
    print(f"\n{'=' * 50}")
    print("Summary table:")
    print(summary_df.to_string(index=False))
    print(f"{'=' * 50}")

    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n Results saved to: {output_path}")

    return result_df


def main():
    config = {
        'pred_data_path': '../../data/Data for Machine Learning.xlsx',
        'model_dir': './model',
        'stats_path': './model/training_stats.pkl',
        'feature_info_path': './model/feature_info.pkl',
        'output_path': 'Predicted results (AMPS_ratio_and_conversion).csv'
    }

    expected_columns = [
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

    df_pred = read_pred_data(config['pred_data_path'], expected_columns)
    if df_pred is None:
        return

    scaler, feature_names = load_training_assets(
        config['stats_path'], config['feature_info_path']
    )
    if scaler is None:
        return

    X_scaled = preprocess_pred_data(df_pred, feature_names, scaler)

    model_paths = get_all_model_paths(config['model_dir'])
    if not model_paths:
        return

    all_predictions = {}
    for path in model_paths:
        model_name, y_pred = predict_with_model(path, X_scaled)
        if model_name:
            all_predictions[model_name] = y_pred

    evaluate_and_save_all(df_pred, all_predictions, config['output_path'])

    print("\n All predictions completed successfully!")


if __name__ == "__main__":
    main()
