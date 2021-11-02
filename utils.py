"""Utilities to use for model learning and selection."""


# =============================================================================
# Imports
# =============================================================================

# Third party
from sklearn.metrics import check_scoring
from sklearn.model_selection import GridSearchCV
import pandas as pd


# =============================================================================
# Functions
# =============================================================================

def optimize_params(estimator, X, y, cv, scoring=None, refit=True, **param_grid):
    """Exhaustive search over specified parameter values for an estimator."""
    grid_search_cv = GridSearchCV(estimator,
                                  param_grid,
                                  scoring=scoring,
                                  refit=refit,
                                  cv=cv,
                                  return_train_score=True).fit(X, y)

    cv_results = pd.DataFrame(grid_search_cv.cv_results_)

    # Drop the results for each validation split and sort by the refit metric
    labels = cv_results.filter(regex="split")
    by = cv_results.filter(regex="rank_test").columns[0]
    cv_results = cv_results.drop(labels, axis=1).sort_values(by)

    display(cv_results)

    return grid_search_cv


def evaluate_estimators(estimators, X, y, *metrics):
    """Evaluate the estimators using the specified metrics."""
    results = pd.DataFrame(columns=metrics)

    for metric in metrics:
        for estimator in estimators:
            # Set the index of the results to the estimator class name
            name = estimator.estimator.__class__.__name__

            # Determine the scorer for evaluating the estimator
            scorer = check_scoring(estimator, metric)

            results.loc[name, metric] = scorer(estimator, X, y)

    return results