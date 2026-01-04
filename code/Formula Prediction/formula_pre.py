import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import os

# Data file paths
input_filepath = '../../data/'
input_filename = 'Data for Machine Learning.xlsx'
output_filepath = './pre_data'
output_filename = 'Recommonded recipes for designed polymers.xlsx'

# 1. Read data
try:
    # Skip the first row, use the second row as column names
    data = pd.read_excel(
        os.path.join(input_filepath, input_filename),
        skiprows=[0],
        header=0
    )
    print(f"Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns")

    # Rename column 8 to [η] (mL/g)
    new_columns = list(data.columns)
    new_columns[7] = '[η] (mL/g)'
    data.columns = new_columns

    # Print corrected column names
    print("\nCorrected column names:")
    for i, col in enumerate(data.columns):
        print(f"  Column {i + 1}: {col}")

except Exception as e:
    print(f"Error reading data: {e}")
    exit(1)

# 2. Data preprocessing: column mapping
col_mapping = {
    # Target columns
    'target1': 'C_i/C_m((g/L)/M)',      # Column 1
    'target2': 'C_ci/C_m((g/L)/M)',     # Column 2
    'target3': 'C_t/C_m((g/L)/M)',      # Column 3
    # Feature columns
    'feat1': 'C_m (M)',                 # Column 4
    'feat2': 'T (°C)',                  # Column 5
    'feat3': 'AMPS feed ratio (mol%)',  # Column 6
    'feat4': 'C_c (mg/L)',              # Column 7
    'feat5': '[η] (mL/g)'               # Column 8
}

# Define features and targets
features = [
    col_mapping['feat1'],
    col_mapping['feat2'],
    col_mapping['feat3'],
    col_mapping['feat4'],
    col_mapping['feat5']
]

targets = [
    col_mapping['target1'],
    col_mapping['target2'],
    col_mapping['target3']
]

# 3. Build prediction models
missing_cols = [col for col in features + targets if col not in data.columns]
if missing_cols:
    print(f"Error: The following columns are missing. Please check the column names in the second row of Excel: {missing_cols}")
    exit(1)

models = {}
for target in targets:
    X = data[features]
    y = data[target]

    # Split training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=52
    )

    # Initialize and train Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nModel for {target} - Mean Squared Error (MSE): {mse:.6f}, R²: {r2:.4f}")
    models[target] = model

# 4. Generate prediction data for specified AMPS ratio values
amps_ratio_values = [30, 80, 100]
pred_data = []

amps_col = col_mapping['feat3']   # AMPS feed ratio (mol%) column
eta_col = col_mapping['feat5']    # [η] (mL/g) column

for amps in amps_ratio_values:
    temp_data = data.copy()
    temp_data[eta_col] = 1000       # Fix intrinsic viscosity [η] to 1000
    temp_data[amps_col] = amps      # Fix AMPS ratio

    # Predict target columns
    for target in targets:
        temp_data[target] = models[target].predict(temp_data[features])

    pred_data.append(temp_data)

# Merge all prediction results
result_df = pd.concat(pred_data, ignore_index=True)

# 5. Reorder result columns
result_cols = [
    eta_col,
    amps_col,
    col_mapping['target1'],
    col_mapping['target2'],
    col_mapping['target3'],
    col_mapping['feat1'],
    col_mapping['feat2'],
    col_mapping['feat4']
]
result_df = result_df[result_cols]

# 6. Export results to Excel
try:
    # Ensure output directory exists
    if not os.path.exists(output_filepath):
        os.makedirs(output_filepath)

    excel_path = os.path.join(output_filepath, output_filename)

    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        from openpyxl.styles import Font
        normal_font = Font(bold=False)

        # Write all prediction data
        result_df.to_excel(writer, sheet_name='All_Predicted_Data', index=False)

        ws_all = writer.sheets['All_Predicted_Data']
        for cell in ws_all[1]:
            cell.font = normal_font

        # Create separate sheets for each AMPS ratio
        for amps in amps_ratio_values:
            sheet_name = f'AMPS_ratio_{amps}'
            amps_df = result_df[result_df[amps_col] == amps]
            amps_df.to_excel(writer, sheet_name=sheet_name, index=False)

            ws_amps = writer.sheets[sheet_name]
            for cell in ws_amps[1]:
                cell.font = normal_font

    print(f"\nPrediction results successfully exported to: {excel_path}")
    print(f"Total rows exported: {len(result_df)}")
    print(f"Each AMPS ratio value {amps_ratio_values} corresponds to {len(data)} rows")

except Exception as e:
    print(f"Error exporting Excel file: {e}")
