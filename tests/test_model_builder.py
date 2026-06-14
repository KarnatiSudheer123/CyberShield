import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

from cybershield.model_builder import (
    build_decision_tree,
    build_random_forest,
    build_extra_trees,
    build_logistic_regression,
    build_naive_bayes,
    build_voting_classifier,
    get_model_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_dataset():
    """Small synthetic binary classification dataset."""
    rng = np.random.RandomState(42)
    X = rng.randn(80, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def multiclass_dataset():
    """Small synthetic multiclass dataset."""
    rng = np.random.RandomState(42)
    X = rng.randn(120, 4)
    y = np.digitize(X[:, 0], bins=[-0.5, 0.5])  # 3 classes
    return X, y


# ---------------------------------------------------------------------------
# build_decision_tree
# ---------------------------------------------------------------------------

class TestBuildDecisionTree:
    def test_returns_correct_type(self):
        model = build_decision_tree()
        assert isinstance(model, DecisionTreeClassifier)

    def test_default_params(self):
        model = build_decision_tree()
        assert model.criterion == "gini"
        assert model.max_depth is None
        assert model.random_state == 42

    def test_custom_random_state(self):
        model = build_decision_tree(random_state=99)
        assert model.random_state == 99

    def test_fit_predict(self, toy_dataset):
        X, y = toy_dataset
        model = build_decision_tree()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape
        assert set(preds).issubset({0, 1})

    def test_predict_proba(self, toy_dataset):
        X, y = toy_dataset
        model = build_decision_tree()
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (len(X), 2)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# build_random_forest
# ---------------------------------------------------------------------------

class TestBuildRandomForest:
    def test_returns_correct_type(self):
        model = build_random_forest()
        assert isinstance(model, RandomForestClassifier)

    def test_default_params(self):
        model = build_random_forest()
        assert model.n_estimators == 100
        assert model.criterion == "gini"
        assert model.random_state == 42

    def test_custom_estimators(self):
        model = build_random_forest(n_estimators=50)
        assert model.n_estimators == 50

    def test_fit_predict(self, toy_dataset):
        X, y = toy_dataset
        model = build_random_forest(n_estimators=10)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape

    def test_multiclass(self, multiclass_dataset):
        X, y = multiclass_dataset
        model = build_random_forest(n_estimators=10)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(preds).issubset(set(y))


# ---------------------------------------------------------------------------
# build_extra_trees
# ---------------------------------------------------------------------------

class TestBuildExtraTrees:
    def test_returns_correct_type(self):
        model = build_extra_trees()
        assert isinstance(model, ExtraTreesClassifier)

    def test_default_params(self):
        model = build_extra_trees()
        assert model.n_estimators == 10
        assert model.random_state == 42

    def test_custom_estimators(self):
        model = build_extra_trees(n_estimators=20)
        assert model.n_estimators == 20

    def test_fit_predict(self, toy_dataset):
        X, y = toy_dataset
        model = build_extra_trees()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape


# ---------------------------------------------------------------------------
# build_logistic_regression
# ---------------------------------------------------------------------------

class TestBuildLogisticRegression:
    def test_returns_correct_type(self):
        model = build_logistic_regression()
        assert isinstance(model, LogisticRegression)

    def test_default_params(self):
        model = build_logistic_regression()
        assert model.solver == "lbfgs"
        assert model.max_iter == 100
        assert model.random_state == 42
        assert model.n_jobs == -1

    def test_custom_max_iter(self):
        model = build_logistic_regression(max_iter=500)
        assert model.max_iter == 500

    def test_fit_predict(self, toy_dataset):
        X, y = toy_dataset
        model = build_logistic_regression()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape

    def test_predict_proba_sums_to_one(self, toy_dataset):
        X, y = toy_dataset
        model = build_logistic_regression()
        model.fit(X, y)
        proba = model.predict_proba(X)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# build_naive_bayes
# ---------------------------------------------------------------------------

class TestBuildNaiveBayes:
    def test_returns_correct_type(self):
        model = build_naive_bayes()
        assert isinstance(model, GaussianNB)

    def test_fit_predict(self, toy_dataset):
        X, y = toy_dataset
        model = build_naive_bayes()
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape

    def test_predict_proba(self, toy_dataset):
        X, y = toy_dataset
        model = build_naive_bayes()
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape[1] == 2
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)


# ---------------------------------------------------------------------------
# build_voting_classifier
# ---------------------------------------------------------------------------

class TestBuildVotingClassifier:
    def test_returns_correct_type(self):
        model = build_voting_classifier()
        assert isinstance(model, VotingClassifier)

    def test_default_voting(self):
        model = build_voting_classifier()
        assert model.voting == "soft"

    def test_custom_estimators(self):
        estimators = [
            ("dt", DecisionTreeClassifier(random_state=0)),
            ("nb", GaussianNB()),
        ]
        model = build_voting_classifier(estimators=estimators)
        assert len(model.estimators) == 2

    def test_hard_voting(self):
        model = build_voting_classifier(voting="hard")
        assert model.voting == "hard"

    def test_fit_predict(self, toy_dataset):
        X, y = toy_dataset
        estimators = [
            ("dt", DecisionTreeClassifier(random_state=0)),
            ("nb", GaussianNB()),
        ]
        model = build_voting_classifier(estimators=estimators)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == y.shape


# ---------------------------------------------------------------------------
# get_model_config
# ---------------------------------------------------------------------------

class TestGetModelConfig:
    def test_returns_dict(self):
        config = get_model_config()
        assert isinstance(config, dict)

    def test_expected_keys(self):
        config = get_model_config()
        expected = {
            "decision_tree",
            "random_forest",
            "extra_trees",
            "logistic_regression",
            "naive_bayes",
            "voting_classifier",
        }
        assert set(config.keys()) == expected

    def test_each_has_builder(self):
        config = get_model_config()
        for name, entry in config.items():
            assert "builder" in entry
            assert callable(entry["builder"])

    def test_each_has_params(self):
        config = get_model_config()
        for name, entry in config.items():
            assert "params" in entry
            assert isinstance(entry["params"], dict)

    def test_builders_produce_instances(self):
        config = get_model_config()
        for name, entry in config.items():
            model = entry["builder"]()
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")
