def new_validation_method(data, target, timesteps=100, test_size=0.1, val_size=0.1):
    # 1. Create supervised learning samples
    X, y = [], []
    for i in range(len(data) - timesteps - 5):  # 5-day prediction horizon
        X.append(data[i:i+timesteps])
        y.append(target[i+timesteps+5])  # Target is 5 days ahead
    
    X = np.array(X)
    y = np.array(y)
    
    # 2. Shuffle samples (key difference from traditional time series validation)
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # 3. Split into train/val/test using standard holdout validation
    # First split: train + temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_shuffled, y_shuffled, test_size=(test_size + val_size), random_state=42
    )
    
    # Second split: val and test from temp
    test_val_ratio = test_size / (test_size + val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_val_ratio, random_state=42
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test
