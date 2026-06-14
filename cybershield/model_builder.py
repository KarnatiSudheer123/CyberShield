from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


def build_decision_tree(random_state=42):
    """Build a Decision Tree classifier with default hyperparameters."""
    return DecisionTreeClassifier(
        criterion="gini",
        max_depth=None,
        random_state=random_state,
    )


def build_random_forest(n_estimators=100, random_state=42):
    """Build a Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="gini",
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )


def build_extra_trees(n_estimators=10, random_state=42):
    """Build an Extra Trees classifier."""
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        criterion="gini",
        max_depth=None,
        random_state=random_state,
        n_jobs=-1,
    )


def build_logistic_regression(max_iter=100, random_state=42):
    """Build a Logistic Regression classifier."""
    return LogisticRegression(
        solver="lbfgs",
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
    )


def build_naive_bayes():
    """Build a Gaussian Naive Bayes classifier."""
    return GaussianNB()


def build_voting_classifier(estimators=None, voting="soft"):
    """Build a Voting Classifier from a list of (name, estimator) pairs.

    If *estimators* is None a default ensemble of RF + GradientBoosting is used.
    """
    if estimators is None:
        estimators = [
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
            ("gb", GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, random_state=42)),
        ]
    return VotingClassifier(estimators=estimators, voting=voting, n_jobs=-1)


def get_model_config():
    """Return a summary dict of all available model builders and their defaults."""
    return {
        "decision_tree": {
            "builder": build_decision_tree,
            "params": {"criterion": "gini", "max_depth": None, "random_state": 42},
        },
        "random_forest": {
            "builder": build_random_forest,
            "params": {"n_estimators": 100, "criterion": "gini", "random_state": 42},
        },
        "extra_trees": {
            "builder": build_extra_trees,
            "params": {"n_estimators": 10, "criterion": "gini", "random_state": 42},
        },
        "logistic_regression": {
            "builder": build_logistic_regression,
            "params": {"solver": "lbfgs", "max_iter": 100, "random_state": 42, "n_jobs": -1},
        },
        "naive_bayes": {
            "builder": build_naive_bayes,
            "params": {},
        },
        "voting_classifier": {
            "builder": build_voting_classifier,
            "params": {"voting": "soft"},
        },
    }
