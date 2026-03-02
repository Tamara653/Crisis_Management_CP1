import pandas as pd
import numpy as np
import torch
import os
from chronos import Chronos2Pipeline 
from tqdm import tqdm


# --------------------------------================----------------------------
# 1. Configuration Parameters
# --------------------------------================================------------
CONFIG = {
    "csv_file_path": "data/mydata.csv", 
    "col_name": "GASA-04_315",
    
    "prediction_length": 40,   # Prediction horizon: 40 time steps (2 hours) into the future
    "context_len": 480,        # Historical context length (1 day) for the model (keep manageable for speed)
    "id_col": "series_01",   
    
    # Test set time range
    "test_start_date": "2025-08-21 00:00:00",
    "test_end_date":   "2025-08-25 00:00:00", 
    
    "model_name": "amazon/chronos-2", 
    "device_map": "cuda" if torch.cuda.is_available() else "cpu",
    
    "step_stride": 1,  # Sliding stride: 1 = predict every step
    "output_file": "chronos_forecast_results_GASA-04_315.csv" # Path to save results
}

# --------------------------------================----------------------------
# 2. Data Preparation
# ----------------------------------------------------------------------------
def prepare_data(config):
    print("📋 [1/3] Loading data...")
    df = pd.read_csv(config["csv_file_path"])
    
    # Time standardization
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    try:
        df['timestamp'] = df['timestamp'].dt.tz_convert('Europe/Paris')
    except:
        pass
    df['timestamp'] = df['timestamp'].dt.tz_localize(None)
    
    df_clean = df[['timestamp', config["col_name"]]].rename(columns={config["col_name"]: "target"})
    df_clean['id'] = config["id_col"]

    test_start_time = pd.to_datetime(config["test_start_date"])
    test_end_time = pd.to_datetime(config["test_end_date"])
    
    # Split data into context (history) and test set (future for evaluation)
    context_df = df_clean[df_clean['timestamp'] < test_start_time].copy()
    
    test_df_full = df_clean[
        (df_clean['timestamp'] >= test_start_time) & 
        (df_clean['timestamp'] <= test_end_time)
    ].copy().reset_index(drop=True)

    return context_df, test_df_full

# --------------------------------================----------------------------
# 3. Dense Sliding Window Inference & Logging (Core Modification)
# ----------------------------------------------------------------------------
def run_sliding_inference_and_record(context_df, test_df_full, config):
    print(f"📦 [2/3] Loading model {config['model_name']}...")
    pipeline = Chronos2Pipeline.from_pretrained(
        config['model_name'], 
        device_map=config['device_map'],
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    
    # List to collect all forecast records
    all_records = []
    
    horizon = config["prediction_length"]
    stride = config["step_stride"]
    num_steps = len(test_df_full) - 1 
    
    print(f"🔮 [3/3] Starting sliding forecast and recording (stride={stride})...")
    
    # Iterate through the test set
    loop_indices = range(0, num_steps, stride)
    
    for i in tqdm(loop_indices):
        # 1. Construct Context
        # Context includes: original history + data up to the current step in the test set
        current_history = pd.concat([context_df, test_df_full.iloc[:i+1]], ignore_index=True)
        
        # Record the "Anchor Time" (reference time) for this specific forecast
        anchor_timestamp = current_history['timestamp'].iloc[-1]
        
        # Trim context if it exceeds the limit
        if len(current_history) > config["context_len"]:
            current_history = current_history.iloc[-config["context_len"]:]
            
        # 2. Construct future_df (Input for prediction)
        freq = pd.infer_freq(current_history['timestamp'][-10:]) or '15min' 
        future_dates = pd.date_range(start=anchor_timestamp, periods=horizon+1, freq=freq)[1:]
        
        future_dummy = pd.DataFrame({
            "id": [config["id_col"]] * horizon,
            "timestamp": future_dates
        })

        # 3. Inference
        forecast = pipeline.predict_df(
            current_history,
            future_df=future_dummy,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column="id",
            timestamp_column="timestamp",
            target="target"
        )
        
        # 4. 📝 [Core Mod] Record detailed data
        # Deconstruct the prediction into individual rows
        pred_p10 = forecast["0.1"].values
        pred_p50 = forecast["0.5"].values
        pred_p90 = forecast["0.9"].values

        # --- Negative Value Clipping ---
        # Force all prediction values < 0 to 0 (Post-processing)
        pred_p10 = np.maximum(pred_p10, 0.0)
        pred_p50 = np.maximum(pred_p50, 0.0)
        pred_p90 = np.maximum(pred_p90, 0.0)
        
        for step in range(horizon):
            record = {
                "ref_timestamp": anchor_timestamp,    # The time prediction was made
                "pred_timestamp": future_dates[step], # The target time being predicted
                "step_ahead": step + 1,               # Step number (1 to 40)
                "pred_p10": pred_p10[step],
                "pred_p50": pred_p50[step],
                "pred_p90": pred_p90[step]
            }
            all_records.append(record)
        
    # Convert list to DataFrame
    forecast_df = pd.DataFrame(all_records)
    return forecast_df

# --------------------------------================----------------------------
# Main Program
# --------------------------------================================------------
if __name__ == "__main__":
    # 1. Preparation
    context_df, test_df_full = prepare_data(CONFIG)
    
    # 2. Inference
    if len(test_df_full) < CONFIG["prediction_length"] + 2:
        print("❌ Error: Test set is too short.")
    else:
        # Get pure forecast results
        forecast_df = run_sliding_inference_and_record(context_df, test_df_full, CONFIG)
        
        print("\n💾 Merging ground truth and saving...")
        
        # 3. Merge Ground Truth
        # We want to see: Predicted Value vs. True Value in the result table
        # Join on 'pred_timestamp' and test_df_full's 'timestamp'
        result_df = pd.merge(
            forecast_df,
            test_df_full[['timestamp', 'target']].rename(columns={'timestamp': 'pred_timestamp', 'target': 'true_value'}),
            on='pred_timestamp',
            how='left' # Use left join to keep forecast records even if ground truth is missing
        )
        
        # 4. Calculate Error (Optional)
        # Calculate absolute error if needed
        # result_df['abs_error'] = (result_df['pred_p50'] - result_df['true_value']).abs()
        
        # 5. Save File
        save_path = CONFIG["output_file"]
        result_df.to_csv(save_path, index=False)
        print(f"✅ All forecast records saved to: {save_path}")
        print(f"   Columns included: {list(result_df.columns)}")

        # 6. (Optional) Simple Visual Check - Plot the last forecast
        # Extract data for the last 'ref_timestamp'
        last_ref = result_df['ref_timestamp'].max()
        last_forecast = result_df[result_df['ref_timestamp'] == last_ref]