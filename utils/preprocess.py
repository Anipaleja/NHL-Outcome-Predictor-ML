import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(df, target_col='target'):
    """
    Preprocess the input dataframe for machine learning:
    - Fill missing values
    - Convert categorical columns to strings, label encode them
    - Scale numerical features
    - Separate features and target

    Parameters:
        df (pd.DataFrame): Raw input dataframe
        target_col (str): Name of the target column

    Returns:
        X_scaled (np.array): Scaled feature matrix
        y (np.array): Target array
    """

    # 1. Fill missing values for all columns
    for col in df.columns:
        if df[col].dtype == 'O':  # Object type, likely categorical
            df[col] = df[col].fillna('missing')
        else:
            # For numeric columns, fill with median
            df[col] = df[col].fillna(df[col].median())

    # 2. Identify categorical columns (object dtype)
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()

    # 3. Label encode categorical columns
    for col in cat_cols:
        # Convert all values explicitly to strings to avoid mixed types
        df[col] = df[col].apply(str)
        df[col] = LabelEncoder().fit_transform(df[col])

    # 4. Separate features and target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 5. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.values
