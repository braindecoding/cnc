#!/usr/bin/env python3
# best_model.py - Script to train the best model for digit classification



import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from scipy.signal import welch



def load_digits_simple(file_path, target_digits=[6, 9], max_per_digit=500):
    """
    Load MindBigData format (TXT/TSV) based on specific column structure:
    Column 1: ID (67635)
    Column 2: Event (67635)
    Column 3: Device (EP)
    Column 4: Channel (AF3, F7, F3, etc.)
    Column 5: Digit (6 or 9) ← Target untuk klasifikasi
    Column 6: Length (260) ← Jumlah data points
    Column 7: Data (comma-separated EEG values)
    """
    print(f"📂 Loading data for digits {target_digits}...")
    if file_path is None or not os.path.exists(file_path):
        print("❌ Dataset file not found!")
        return None, None
    print(f"📖 Reading file: {file_path}")
    # Initialize data containers
    data_6 = []
    data_9 = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f):
            if not line.strip():
                continue
            # Split by TAB
            parts = line.split('\t')
            # Need at least 7 columns
            if len(parts) < 7:
                continue
            try:
                # Column 5 (index 4) = digit
                digit = int(parts[4])
                # Only process if it's in target digits
                if digit in target_digits:
                    # Column 7 (index 6) = data
                    data_string = parts[6]
                    # Parse comma-separated values
                    values = [np.float64(x.strip()) for x in data_string.split(',') if x.strip()]
                    # Store based on digit
                    if digit == 6 and len(data_6) < max_per_digit:
                        data_6.append(values)
                    elif digit == 9 and len(data_9) < max_per_digit:
                        data_9.append(values)
                    # Progress
                    if (len(data_6) + len(data_9)) % 100 == 0:
                        print(f" Found: {len(data_6)} digit-6, {len(data_9)} digit-9")
                    # Stop when we have enough
                    if len(data_6) >= max_per_digit and len(data_9) >= max_per_digit:
                        break
            except (ValueError, IndexError):
                continue
    print(f"✅ Final count: {len(data_6)} digit-6, {len(data_9)} digit-9")
    if len(data_6) == 0 or len(data_9) == 0:
        print("❌ Missing data for one or both digits!")
        return None, None
    # Combine data and labels
    all_data = data_6 + data_9
    all_labels = [6] * len(data_6) + [9] * len(data_9)
    # Normalize lengths (simple padding/truncating)
    normalized_data = []
    target_length = 1792 # 14 channels * 128 timepoints
    for trial in all_data:
        if len(trial) >= target_length:
            # Truncate if too long
            normalized_data.append(trial[:target_length])
        else:
            # Pad with repetition if too short
            trial_copy = trial.copy()
            while len(trial_copy) < target_length:
                trial_copy.extend(trial[:min(len(trial), target_length - len(trial_copy))])
            normalized_data.append(trial_copy[:target_length])
    data = np.array(normalized_data, dtype=np.float64)
    labels = np.array(all_labels, dtype=np.int32)
    # Check for NaN or infinity values
    if np.isnan(data).any() or np.isinf(data).any():
        print(" ⚠️ Warning: NaN or Infinity values detected in data, replacing with zeros")
        data = np.nan_to_num(data)
    print(f" 📊 Data shape: {data.shape}")
    return data, labels



def extract_features(data):
    """Extract spatial and frequency features from the data"""
    print("\n🧩 Extracting features...")
    # Define channel groups
    frontal_channels = [0, 1, 2, 3, 11, 12, 13] # AF3, F7, F3, FC5, F4, F8, AF4
    temporal_channels = [4, 5, 8, 9] # T7, P7, P8, T8
    occipital_channels = [6, 7] # O1, O2
    left_channels = [0, 1, 2, 3, 4, 5, 6] # Left hemisphere
    right_channels = [7, 8, 9, 10, 11, 12, 13] # Right hemisphere
    # Reshape data to 14 channels x 128 timepoints
    reshaped_data = []
    for trial in data:
        try:
            # Reshape to 14 x 128
            reshaped = trial.reshape(14, 128)
            reshaped_data.append(reshaped)
        except ValueError:
            print(f" ⚠️ Reshape failed for trial with length {len(trial)}")
            continue
    # Define frequency bands
    fs = 128 # Sampling frequency (Hz)
    features = []
    for trial in reshaped_data:
        trial_features = []
        # Basic statistical features
        channel_means = np.mean(trial, axis=1)
        channel_stds = np.std(trial, axis=1)
        # Hemispheric features
        left_mean = np.mean(channel_means[left_channels])
        right_mean = np.mean(channel_means[right_channels])
        # Regional features
        frontal_mean = np.mean(channel_means[frontal_channels])
        temporal_mean = np.mean(channel_means[temporal_channels])
        occipital_mean = np.mean(channel_means[occipital_channels])
        # Synchronization features
        o1_o2_corr = np.corrcoef(trial[6], trial[7])[0, 1] # O1-O2 correlation
        f3_f4_corr = np.corrcoef(trial[2], trial[11])[0, 1] # F3-F4 correlation
        # Frequency features
        try:
            # Calculate power in different frequency bands
            frontal_alpha_power = 0
            frontal_beta_power = 0
            occipital_alpha_power = 0
            # Calculate power for frontal channels
            for ch_idx in frontal_channels:
                freqs, psd = welch(trial[ch_idx], fs=fs, nperseg=min(64, len(trial[ch_idx])))
                alpha_idx = np.logical_and(freqs >= 8, freqs <= 13)
                beta_idx = np.logical_and(freqs >= 13, freqs <= 30)
                frontal_alpha_power += np.sum(psd[alpha_idx])
                frontal_beta_power += np.sum(psd[beta_idx])
            # Normalize by number of channels
            frontal_alpha_power /= len(frontal_channels)
            frontal_beta_power /= len(frontal_channels)
            # Calculate power for occipital channels
            for ch_idx in occipital_channels:
                freqs, psd = welch(trial[ch_idx], fs=fs, nperseg=min(64, len(trial[ch_idx])))
                alpha_idx = np.logical_and(freqs >= 8, freqs <= 13)
                occipital_alpha_power += np.sum(psd[alpha_idx])
            # Normalize by number of channels
            occipital_alpha_power /= len(occipital_channels)
            # Calculate alpha/beta ratio
            frontal_alpha_beta_ratio = frontal_alpha_power / frontal_beta_power if frontal_beta_power != 0 else 0
        except Exception:
            frontal_alpha_power = 0
            frontal_beta_power = 0
            occipital_alpha_power = 0
            frontal_alpha_beta_ratio = 0
        # Combine all features
        trial_features.extend([
            left_mean - right_mean, # Hemispheric asymmetry
            frontal_mean / occipital_mean if occipital_mean != 0 else 0, # Front-back ratio
            o1_o2_corr, # Occipital synchronization
            f3_f4_corr, # Frontal synchronization
            frontal_alpha_power, # Frontal alpha power
            frontal_beta_power, # Frontal beta power
            occipital_alpha_power, # Occipital alpha power
            frontal_alpha_beta_ratio # Alpha/beta ratio
        ])
        features.append(trial_features)
    features = np.array(features, dtype=np.float64)
    # Check for NaN or infinity values
    if np.isnan(features).any() or np.isinf(features).any():
        print(" ⚠️ Warning: NaN or Infinity values in features, replacing with zeros")
        features = np.nan_to_num(features)
    print(f" ✅ Features extracted: {features.shape}")
    return features



def train_best_model(X, y):
    """Train the best model based on previous tuning"""
    print("\n🤖 Training best model...")
    # Ensure consistent data types
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.int32)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Feature selection
    selector = SelectKBest(f_classif, k=min(20, X_train_scaled.shape[1]))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    # Train best SVM model (based on previous tuning)
    svm = SVC(C=20.0, kernel='rbf', gamma=0.01, random_state=42)
    svm.fit(X_train_selected, y_train)
    # Evaluate
    y_pred = svm.predict(X_test_selected)
    accuracy = accuracy_score(y_test, y_pred)
    print(f" ✅ Model accuracy: {accuracy:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Digit 6', 'Digit 9']))
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f" Confusion Matrix:")
    print(f" {cm[0][0]:4d} {cm[0][1]:4d} | Digit 6")
    print(f" {cm[1][0]:4d} {cm[1][1]:4d} | Digit 9")
    print(f" 6 9 <- Predicted")
    # Calculate sensitivity and specificity
    sensitivity = cm[0][0] / (cm[0][0] + cm[0][1]) # True positive rate for digit 6
    specificity = cm[1][1] / (cm[1][0] + cm[1][1]) # True positive rate for digit 9
    print(f" Sensitivity (Digit 6): {sensitivity:.4f}")
    print(f" Specificity (Digit 9): {specificity:.4f}")
    return svm, scaler, selector



def main():
    """Main function"""
    print("🚀 Best Model for EEG Digit Classification")
    print("=" * 50)
    # Load data
    file_path = "Data/EP1.01.txt"
    data, labels = load_digits_simple(file_path, max_per_digit=500)
    if data is None:
        print("❌ Failed to load data")
        return
    # Extract features
    features = extract_features(data)
    # Train best model
    model, scaler, selector = train_best_model(features, labels)
    print("\n✅ Analysis completed!")
    print("🎯 Note: Accuracy above 0.5 indicates that spatial patterns")
    print(" can be detected in the EEG data to differentiate digits 6 and 9.")



if __name__ == "__main__":
    main()
