import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA


def clean_dataframe(df):
    """Remove null values and duplicates, then reset the index."""
    df = df.dropna()
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)
    return df


def encode_labels(series):
    """Encode categorical labels to integers using LabelEncoder.

    Returns (encoded_series, encoder).
    """
    le = LabelEncoder()
    encoded = le.fit_transform(series)
    return encoded, le


def scale_features(X):
    """Standardize features using StandardScaler.

    Returns (scaled_array, scaler).
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def apply_pca(X_scaled, n_components=18, random_state=42):
    """Apply PCA dimensionality reduction.

    Returns (transformed_array, pca_object).
    """
    pca = PCA(n_components=n_components, svd_solver="full", random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca


def get_top_features_per_component(pca, feature_names, top_n=5):
    """Return a dict mapping each PC to its top contributing features."""
    result = {}
    for i in range(pca.n_components_):
        loadings = pd.Series(pca.components_[i], index=feature_names)
        top = loadings.abs().sort_values(ascending=False).head(top_n).index.tolist()
        result[f"PC{i + 1}"] = top
    return result


def get_overall_feature_importance(pca, feature_names):
    """Return a Series of overall feature importance across all PCs."""
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
        index=feature_names,
    )
    return loadings.abs().sum(axis=1).sort_values(ascending=False)


def replace_infinities(df):
    """Replace +/-inf with NaN, then drop resulting NaN rows."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    return df


def cast_numeric_columns(df, exclude=None):
    """Cast all columns except *exclude* to float64."""
    exclude = exclude or []
    for col in df.columns:
        if col not in exclude:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
