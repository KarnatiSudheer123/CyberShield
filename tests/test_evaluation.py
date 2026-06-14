import numpy as np
import pytest

from cybershield.evaluation import (
    ResultsTracker,
    compute_metrics,
    generate_classification_report,
    compute_confusion_matrix,
    compute_roc_data,
    compute_reconstruction_error,
    binary_anomaly_labels,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_predictions():
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    return y_true, y_pred


@pytest.fixture
def multiclass_predictions():
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 2, 1, 0, 1, 2])
    return y_true, y_pred


# ---------------------------------------------------------------------------
# ResultsTracker
# ---------------------------------------------------------------------------

class TestResultsTracker:
    def test_initial_empty(self):
        tracker = ResultsTracker()
        assert len(tracker) == 0

    def test_store_single(self):
        tracker = ResultsTracker()
        tracker.store("RF", 0.95, 0.94, 0.93, 0.935)
        assert len(tracker) == 1
        assert tracker.models[0] == "RF"

    def test_store_multiple(self):
        tracker = ResultsTracker()
        tracker.store("RF", 0.95, 0.94, 0.93, 0.935)
        tracker.store("DT", 0.88, 0.87, 0.86, 0.865)
        assert len(tracker) == 2

    def test_rounding(self):
        tracker = ResultsTracker()
        tracker.store("M", 0.95678, 0.94321, 0.93111, 0.93555)
        assert tracker.accuracy[0] == 0.957
        assert tracker.precision[0] == 0.943
        assert tracker.recall[0] == 0.931
        assert tracker.f1score[0] == 0.936

    def test_to_dict_keys(self):
        tracker = ResultsTracker()
        tracker.store("M", 0.9, 0.9, 0.9, 0.9)
        d = tracker.to_dict()
        assert set(d.keys()) == {"ML Model", "Accuracy", "Precision", "Recall", "F1 Score"}

    def test_to_dict_values(self):
        tracker = ResultsTracker()
        tracker.store("A", 0.1, 0.2, 0.3, 0.4)
        tracker.store("B", 0.5, 0.6, 0.7, 0.8)
        d = tracker.to_dict()
        assert d["ML Model"] == ["A", "B"]
        assert d["Accuracy"] == [0.1, 0.5]


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_predictions(self):
        y = np.array([0, 1, 2, 0, 1])
        acc, prec, rec, f1 = compute_metrics(y, y)
        assert acc == 1.0
        assert prec == 1.0
        assert rec == 1.0
        assert f1 == 1.0

    def test_binary_metrics(self, binary_predictions):
        y_true, y_pred = binary_predictions
        acc, prec, rec, f1 = compute_metrics(y_true, y_pred)
        assert 0.0 <= acc <= 1.0
        assert 0.0 <= prec <= 1.0
        assert 0.0 <= rec <= 1.0
        assert 0.0 <= f1 <= 1.0

    def test_multiclass_metrics(self, multiclass_predictions):
        y_true, y_pred = multiclass_predictions
        acc, prec, rec, f1 = compute_metrics(y_true, y_pred)
        assert acc == pytest.approx(7 / 9)

    def test_all_wrong(self):
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        acc, _, _, _ = compute_metrics(y_true, y_pred)
        assert acc == 0.0

    def test_returns_four_values(self, binary_predictions):
        result = compute_metrics(*binary_predictions)
        assert len(result) == 4


# ---------------------------------------------------------------------------
# generate_classification_report
# ---------------------------------------------------------------------------

class TestGenerateClassificationReport:
    def test_returns_dict(self, binary_predictions):
        y_true, y_pred = binary_predictions
        report = generate_classification_report(y_true, y_pred)
        assert isinstance(report, dict)

    def test_contains_accuracy(self, binary_predictions):
        y_true, y_pred = binary_predictions
        report = generate_classification_report(y_true, y_pred)
        assert "accuracy" in report

    def test_target_names(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])
        report = generate_classification_report(
            y_true, y_pred, target_names=["BENIGN", "DDoS"]
        )
        assert "BENIGN" in report
        assert "DDoS" in report

    def test_per_class_keys(self, multiclass_predictions):
        y_true, y_pred = multiclass_predictions
        report = generate_classification_report(y_true, y_pred)
        for key in ["0", "1", "2"]:
            assert key in report
            assert "precision" in report[key]
            assert "recall" in report[key]
            assert "f1-score" in report[key]


# ---------------------------------------------------------------------------
# compute_confusion_matrix
# ---------------------------------------------------------------------------

class TestComputeConfusionMatrix:
    def test_shape_binary(self, binary_predictions):
        y_true, y_pred = binary_predictions
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)

    def test_shape_multiclass(self, multiclass_predictions):
        y_true, y_pred = multiclass_predictions
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.shape == (3, 3)

    def test_diagonal_perfect(self):
        y = np.array([0, 1, 2, 0])
        cm = compute_confusion_matrix(y, y)
        np.testing.assert_array_equal(np.diag(cm), [2, 1, 1])
        assert cm.sum() - np.diag(cm).sum() == 0

    def test_sum_equals_total(self, binary_predictions):
        y_true, y_pred = binary_predictions
        cm = compute_confusion_matrix(y_true, y_pred)
        assert cm.sum() == len(y_true)


# ---------------------------------------------------------------------------
# compute_roc_data
# ---------------------------------------------------------------------------

class TestComputeRocData:
    def test_binary_roc(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.1, 0.9]])
        results = compute_roc_data(y_true, y_proba, n_classes=2)
        assert len(results) == 2
        for r in results:
            assert "fpr" in r
            assert "tpr" in r
            assert "auc" in r

    def test_perfect_roc_auc(self):
        y_true = np.array([0, 0, 1, 1])
        y_proba = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
        results = compute_roc_data(y_true, y_proba, n_classes=2)
        assert results[0]["auc"] == pytest.approx(1.0)
        assert results[1]["auc"] == pytest.approx(1.0)

    def test_multiclass_roc(self):
        rng = np.random.RandomState(42)
        y_true = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        y_proba = rng.dirichlet([1, 1, 1], size=9)
        results = compute_roc_data(y_true, y_proba, n_classes=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# compute_reconstruction_error
# ---------------------------------------------------------------------------

class TestComputeReconstructionError:
    def test_perfect_reconstruction(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        mse = compute_reconstruction_error(X, X)
        np.testing.assert_array_almost_equal(mse, [0.0, 0.0])

    def test_known_error(self):
        original = np.array([[1.0, 0.0]])
        reconstructed = np.array([[0.0, 0.0]])
        mse = compute_reconstruction_error(original, reconstructed)
        assert mse[0] == pytest.approx(0.5)

    def test_shape(self):
        rng = np.random.RandomState(0)
        X = rng.randn(50, 10)
        mse = compute_reconstruction_error(X, X + 0.1)
        assert mse.shape == (50,)

    def test_non_negative(self):
        rng = np.random.RandomState(1)
        X = rng.randn(20, 5)
        R = rng.randn(20, 5)
        mse = compute_reconstruction_error(X, R)
        assert np.all(mse >= 0)


# ---------------------------------------------------------------------------
# binary_anomaly_labels
# ---------------------------------------------------------------------------

class TestBinaryAnomalyLabels:
    def test_basic(self):
        y = np.array([0, 1, 2, 0, 3])
        result = binary_anomaly_labels(y, benign_label=0)
        np.testing.assert_array_equal(result, [0, 1, 1, 0, 1])

    def test_all_benign(self):
        y = np.array([5, 5, 5])
        result = binary_anomaly_labels(y, benign_label=5)
        np.testing.assert_array_equal(result, [0, 0, 0])

    def test_no_benign(self):
        y = np.array([1, 2, 3])
        result = binary_anomaly_labels(y, benign_label=0)
        np.testing.assert_array_equal(result, [1, 1, 1])

    def test_dtype(self):
        y = np.array([0, 1])
        result = binary_anomaly_labels(y, benign_label=0)
        assert result.dtype == int
