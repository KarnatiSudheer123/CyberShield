import numpy as np
import pandas as pd
import pytest

from cybershield.data_preprocessing import (
    clean_dataframe,
    encode_labels,
    scale_features,
    apply_pca,
    get_top_features_per_component,
    get_overall_feature_importance,
    replace_infinities,
    cast_numeric_columns,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df():
    """A small DataFrame with nulls, duplicates, and mixed types."""
    return pd.DataFrame(
        {
            "A": [1.0, 2.0, np.nan, 2.0, 5.0],
            "B": [10.0, 20.0, 30.0, 20.0, 50.0],
            "C": ["x", "y", "z", "y", "w"],
        }
    )


@pytest.fixture
def numeric_df():
    """A purely numeric DataFrame suitable for scaling/PCA."""
    rng = np.random.RandomState(0)
    return pd.DataFrame(rng.randn(100, 5), columns=[f"f{i}" for i in range(5)])


# ---------------------------------------------------------------------------
# clean_dataframe
# ---------------------------------------------------------------------------

class TestCleanDataframe:
    def test_removes_nulls(self, sample_df):
        result = clean_dataframe(sample_df)
        assert result.isnull().sum().sum() == 0

    def test_removes_duplicates(self, sample_df):
        result = clean_dataframe(sample_df)
        assert result.duplicated().sum() == 0

    def test_resets_index(self, sample_df):
        result = clean_dataframe(sample_df)
        assert list(result.index) == list(range(len(result)))

    def test_no_change_on_clean_data(self):
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        result = clean_dataframe(df)
        assert len(result) == 3

    def test_all_null_rows_dropped(self):
        df = pd.DataFrame({"a": [np.nan, np.nan], "b": [np.nan, np.nan]})
        result = clean_dataframe(df)
        assert len(result) == 0

    def test_preserves_columns(self, sample_df):
        result = clean_dataframe(sample_df)
        assert list(result.columns) == list(sample_df.columns)


# ---------------------------------------------------------------------------
# encode_labels
# ---------------------------------------------------------------------------

class TestEncodeLabels:
    def test_encodes_strings(self):
        series = pd.Series(["BENIGN", "DDoS", "BENIGN", "Bot"])
        encoded, le = encode_labels(series)
        assert len(encoded) == 4
        assert set(encoded) == {0, 1, 2} or len(set(encoded)) == 3

    def test_inverse_transform(self):
        series = pd.Series(["A", "B", "C", "A"])
        encoded, le = encode_labels(series)
        decoded = le.inverse_transform(encoded)
        np.testing.assert_array_equal(decoded, series.values)

    def test_encoder_classes(self):
        series = pd.Series(["cat", "dog", "cat", "bird"])
        _, le = encode_labels(series)
        assert sorted(le.classes_) == ["bird", "cat", "dog"]

    def test_single_class(self):
        series = pd.Series(["same", "same", "same"])
        encoded, le = encode_labels(series)
        assert np.all(encoded == 0)

    def test_returns_numpy_array(self):
        series = pd.Series(["a", "b"])
        encoded, _ = encode_labels(series)
        assert isinstance(encoded, np.ndarray)


# ---------------------------------------------------------------------------
# scale_features
# ---------------------------------------------------------------------------

class TestScaleFeatures:
    def test_output_shape(self, numeric_df):
        scaled, _ = scale_features(numeric_df)
        assert scaled.shape == numeric_df.shape

    def test_zero_mean(self, numeric_df):
        scaled, _ = scale_features(numeric_df)
        np.testing.assert_array_almost_equal(scaled.mean(axis=0), 0, decimal=10)

    def test_unit_variance(self, numeric_df):
        scaled, _ = scale_features(numeric_df)
        np.testing.assert_array_almost_equal(scaled.std(axis=0, ddof=0), 1, decimal=10)

    def test_scaler_transform(self, numeric_df):
        scaled, scaler = scale_features(numeric_df)
        retransformed = scaler.transform(numeric_df)
        np.testing.assert_array_almost_equal(scaled, retransformed)

    def test_inverse_transform(self, numeric_df):
        scaled, scaler = scale_features(numeric_df)
        original = scaler.inverse_transform(scaled)
        np.testing.assert_array_almost_equal(original, numeric_df.values, decimal=10)


# ---------------------------------------------------------------------------
# apply_pca
# ---------------------------------------------------------------------------

class TestApplyPca:
    def test_output_shape(self, numeric_df):
        scaled, _ = scale_features(numeric_df)
        X_pca, _ = apply_pca(scaled, n_components=3)
        assert X_pca.shape == (100, 3)

    def test_explained_variance_sums_to_le_one(self, numeric_df):
        scaled, _ = scale_features(numeric_df)
        _, pca = apply_pca(scaled, n_components=3)
        assert pca.explained_variance_ratio_.sum() <= 1.0 + 1e-10

    def test_n_components_attribute(self, numeric_df):
        scaled, _ = scale_features(numeric_df)
        _, pca = apply_pca(scaled, n_components=2)
        assert pca.n_components_ == 2

    def test_deterministic_with_seed(self, numeric_df):
        scaled, _ = scale_features(numeric_df)
        X1, _ = apply_pca(scaled, n_components=3, random_state=0)
        X2, _ = apply_pca(scaled, n_components=3, random_state=0)
        np.testing.assert_array_equal(X1, X2)

    def test_max_components(self, numeric_df):
        scaled, _ = scale_features(numeric_df)
        X_pca, pca = apply_pca(scaled, n_components=5)
        assert X_pca.shape[1] == 5
        np.testing.assert_almost_equal(
            pca.explained_variance_ratio_.sum(), 1.0, decimal=10
        )


# ---------------------------------------------------------------------------
# get_top_features_per_component
# ---------------------------------------------------------------------------

class TestGetTopFeaturesPerComponent:
    def test_returns_correct_keys(self, numeric_df):
        scaled, _ = scale_features(numeric_df)
        _, pca = apply_pca(scaled, n_components=3)
        result = get_top_features_per_component(pca, numeric_df.columns, top_n=2)
        assert set(result.keys()) == {"PC1", "PC2", "PC3"}

    def test_top_n_features_length(self, numeric_df):
        scaled, _ = scale_features(numeric_df)
        _, pca = apply_pca(scaled, n_components=2)
        result = get_top_features_per_component(pca, numeric_df.columns, top_n=3)
        for features in result.values():
            assert len(features) == 3

    def test_features_are_from_input(self, numeric_df):
        scaled, _ = scale_features(numeric_df)
        _, pca = apply_pca(scaled, n_components=2)
        result = get_top_features_per_component(pca, numeric_df.columns, top_n=5)
        for features in result.values():
            for f in features:
                assert f in numeric_df.columns.tolist()


# ---------------------------------------------------------------------------
# get_overall_feature_importance
# ---------------------------------------------------------------------------

class TestGetOverallFeatureImportance:
    def test_returns_series(self, numeric_df):
        scaled, _ = scale_features(numeric_df)
        _, pca = apply_pca(scaled, n_components=3)
        result = get_overall_feature_importance(pca, numeric_df.columns)
        assert isinstance(result, pd.Series)

    def test_sorted_descending(self, numeric_df):
        scaled, _ = scale_features(numeric_df)
        _, pca = apply_pca(scaled, n_components=3)
        result = get_overall_feature_importance(pca, numeric_df.columns)
        assert list(result.values) == sorted(result.values, reverse=True)

    def test_all_features_present(self, numeric_df):
        scaled, _ = scale_features(numeric_df)
        _, pca = apply_pca(scaled, n_components=3)
        result = get_overall_feature_importance(pca, numeric_df.columns)
        assert set(result.index) == set(numeric_df.columns)


# ---------------------------------------------------------------------------
# replace_infinities
# ---------------------------------------------------------------------------

class TestReplaceInfinities:
    def test_removes_pos_inf(self):
        df = pd.DataFrame({"a": [1.0, np.inf, 3.0]})
        result = replace_infinities(df)
        assert not np.isinf(result.values).any()
        assert len(result) == 2

    def test_removes_neg_inf(self):
        df = pd.DataFrame({"a": [1.0, -np.inf, 3.0]})
        result = replace_infinities(df)
        assert len(result) == 2

    def test_no_inf_unchanged(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = replace_infinities(df)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# cast_numeric_columns
# ---------------------------------------------------------------------------

class TestCastNumericColumns:
    def test_casts_to_numeric(self):
        df = pd.DataFrame({"a": ["1", "2", "3"], "b": ["4", "5", "6"]})
        result = cast_numeric_columns(df)
        assert np.issubdtype(result["a"].dtype, np.number)
        assert np.issubdtype(result["b"].dtype, np.number)

    def test_excludes_columns(self):
        df = pd.DataFrame({"a": ["1", "2"], "label": ["x", "y"]})
        result = cast_numeric_columns(df, exclude=["label"])
        assert np.issubdtype(result["a"].dtype, np.number)
        # label column should remain non-numeric (string/object)
        assert not pd.api.types.is_numeric_dtype(result["label"])

    def test_non_numeric_becomes_nan(self):
        df = pd.DataFrame({"a": ["1", "hello", "3"]})
        result = cast_numeric_columns(df)
        assert result["a"].isna().sum() == 1
