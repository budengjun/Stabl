import os
from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from .utils import compute_CI, permutation_test_between_clfs
from .visualization import scatterplot_regression_predictions, boxplot_binary_predictions, plot_roc, \
    plot_prc
from .metrics import jaccard_matrix
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_squared_error, mean_absolute_error

from scipy import stats
from scipy.stats import mannwhitneyu
from sklearn.feature_selection._base import _get_feature_importances
from sklearn.feature_selection._from_model import _calculate_threshold
from sklearn.model_selection import GridSearchCV


def save_plots(predictions_dict, y, task_type, save_path):
    """
    Function to save the plots when performing a stabl against lasso benchmark.

    Modified version:
    - Saves raw prediction table as before.
    - For plotting, converts CV prediction DataFrame to row-wise median prediction.
    - Drops samples with all-NaN predictions before plotting.
    """

    def clean_predictions_for_plots(predictions, y, model_name=None):
        """
        Convert prediction object into clean Series for plotting.
        """

        if isinstance(predictions, pd.DataFrame):
            predictions_clean = predictions.median(axis=1, skipna=True)

        elif isinstance(predictions, pd.Series):
            predictions_clean = predictions

        else:
            predictions_clean = pd.Series(np.asarray(predictions).ravel())

        if isinstance(y, pd.Series):
            y_clean = y.loc[predictions_clean.index]
        else:
            y_clean = pd.Series(np.asarray(y).ravel(), index=predictions_clean.index)

        valid_mask = ~predictions_clean.isna()

        if valid_mask.sum() < len(valid_mask):
            print("\n========== Warning: NaN predictions detected before plotting ==========")
            print("Model:", model_name)
            print("Number of samples:", len(valid_mask))
            print("Samples dropped from plots:", len(valid_mask) - valid_mask.sum())

            if isinstance(predictions, pd.DataFrame):
                print("Rows with all-NaN predictions:")
                print(predictions[predictions.isna().all(axis=1)].head(20))

        predictions_clean = predictions_clean.loc[valid_mask]
        y_clean = y_clean.loc[valid_mask]

        return predictions_clean, y_clean

    for name, predictions in predictions_dict.items():
        print(name, predictions)

        saving_path = Path(save_path, name)
        os.makedirs(saving_path, exist_ok=True)

        # Save raw predictions as before
        if isinstance(predictions, pd.DataFrame):
            pred_to_save = predictions.copy()

        elif isinstance(predictions, pd.Series):
            pred_to_save = predictions.to_frame(name="prediction")

        else:
            pred_to_save = pd.DataFrame(
                np.asarray(predictions).ravel(),
                columns=["prediction"]
            )

        if isinstance(y, pd.Series) and pred_to_save.index.isin(y.index).all():
            y_to_save = y.loc[pred_to_save.index]
        else:
            y_to_save = pd.Series(
                np.asarray(y).ravel()[:len(pred_to_save)],
                index=pred_to_save.index,
                name="y"
            )

        pd.concat(
            [pred_to_save, y_to_save],
            axis=1
        ).to_csv(
            os.path.join(saving_path, f"{name} predictions.csv")
        )

        # Clean predictions only for plotting
        predictions_clean, y_clean = clean_predictions_for_plots(
            predictions=predictions,
            y=y,
            model_name=name
        )

        if task_type == "binary":
            plot_roc(
                y_true=y_clean,
                y_preds=predictions_clean,
                show_CI=True,
                export_file=True,
                show_fig=False,
                paths=os.path.join(saving_path, f"{name} ROC.pdf")
            )

            plot_prc(
                y_true=y_clean,
                y_preds=predictions_clean,
                show_CI=True,
                export_file=True,
                show_fig=False,
                path=os.path.join(saving_path, f"{name} PR Curve.pdf")
            )

            boxplot_binary_predictions(
                y_true=y_clean,
                y_preds=predictions_clean,
                export_file=True,
                show_fig=False,
                paths=os.path.join(saving_path, f"{name} Boxplot of median predictions.pdf")
            )

        elif task_type == "regression":
            scatterplot_regression_predictions(
                y_true=y_clean,
                y_preds=predictions_clean,
                export_file=True,
                show_fig=False,
                paths=os.path.join(saving_path, f"{name} Scatter-plot of median predictions.pdf")
            )


def compute_scores_table(
        predictions_dict,
        y,
        task_type="binary",
        selected_features_dict=None
):
    """Function to output the table of scores for benchmarking.

    Modified version:
    - Handles prediction DataFrames from repeated CV by taking row-wise median.
    - Aligns y with prediction index.
    - Drops samples with all-NaN predictions before computing scores.
    """

    def clean_predictions_for_scores(preds, y, model_name=None):
        """
        Convert prediction object into clean 1D arrays for scoring.

        If preds is a DataFrame, each row may correspond to one sample and each
        column to one CV split. Some entries can be NaN because a sample was not
        in the test set for that split. We take the row-wise median over available
        predictions.

        If a row is still NaN after median, that sample had no prediction in any
        split and is dropped.
        """

        if isinstance(preds, pd.DataFrame):
            preds_clean = preds.median(axis=1, skipna=True)

        elif isinstance(preds, pd.Series):
            preds_clean = preds

        else:
            preds_clean = pd.Series(np.asarray(preds).ravel())

        if isinstance(y, pd.Series):
            if isinstance(preds_clean, pd.Series):
                y_clean = y.loc[preds_clean.index]
            else:
                y_clean = y
        else:
            y_clean = pd.Series(np.asarray(y).ravel(), index=preds_clean.index)

        preds_array = np.asarray(preds_clean, dtype=float)
        y_array = np.asarray(y_clean, dtype=float)

        valid_mask = ~np.isnan(preds_array)

        if valid_mask.sum() < len(valid_mask):
            print("\n========== Warning: NaN predictions detected ==========")
            print("Model:", model_name)
            print("Original preds type:", type(preds))
            print("Original preds shape:", np.asarray(preds).shape)
            print("Number of samples:", len(valid_mask))
            print("Samples dropped because prediction is NaN:", len(valid_mask) - valid_mask.sum())

            if isinstance(preds, pd.DataFrame):
                print("Rows with all-NaN predictions:")
                print(preds[preds.isna().all(axis=1)].head(20))

                print("NaN count per prediction column:")
                print(preds.isna().sum())

        y_array = y_array[valid_mask]
        preds_array = preds_array[valid_mask]

        return y_array, preds_array

    scores_columns = []
    if selected_features_dict is not None:
        if task_type == "binary":
            scores_columns = ["ROC AUC", "Average Precision", "N features", "CVS"]

        elif task_type == "regression":
            scores_columns = ["R2", "RMSE", "MAE", "N features", "CVS"]

    else:
        if task_type == "binary":
            scores_columns = ["ROC AUC", "Average Precision"]

        elif task_type == "regression":
            scores_columns = ["R2", "RMSE", "MAE"]

    table_of_scores = pd.DataFrame(data=None, columns=scores_columns)

    for model, preds in predictions_dict.items():

        y_array, preds_array = clean_predictions_for_scores(
            preds=preds,
            y=y,
            model_name=model
        )

        for metric in scores_columns:

            if metric == "ROC AUC":
                model_roc = roc_auc_score(y_array, preds_array)
                model_roc_CI = compute_CI(y_array, preds_array, scoring="roc_auc")
                cell_value = f"{model_roc:.3f} [{model_roc_CI[0]:.3f}, {model_roc_CI[1]:.3f}]"

            elif metric == "Average Precision":
                model_ap = average_precision_score(y_array, preds_array)
                model_ap_CI = compute_CI(y_array, preds_array, scoring="average_precision")
                cell_value = f"{model_ap:.3f} [{model_ap_CI[0]:.3f}, {model_ap_CI[1]:.3f}]"

            elif metric == "N features":
                sel_features = selected_features_dict[model]["Fold nb of features"]
                median_features = np.median(sel_features)
                iqr_features = np.quantile(sel_features, [.25, .75])
                cell_value = f"{median_features:.3f} [{iqr_features[0]:.3f}, {iqr_features[1]:.3f}]"

            elif metric == "CVS":
                jaccard_mat = jaccard_matrix(
                    selected_features_dict[model]["Fold selected features"],
                    remove_diag=False
                )
                jaccard_val = jaccard_mat[np.triu_indices_from(jaccard_mat, k=1)]
                jaccard_median = np.median(jaccard_val)
                jaccard_iqr = np.quantile(jaccard_val, [.25, .75])
                cell_value = f"{jaccard_median:.3f} [{jaccard_iqr[0]:.3f}, {jaccard_iqr[1]:.3f}]"

            elif metric == "R2":
                model_r2 = r2_score(y_array, preds_array)
                model_r2_CI = compute_CI(y_array, preds_array, scoring="r2")
                cell_value = f"{model_r2:.3f} [{model_r2_CI[0]:.3f}, {model_r2_CI[1]:.3f}]"

            elif metric == "RMSE":
                model_rmse = np.sqrt(mean_squared_error(y_array, preds_array))
                model_rmse_CI = compute_CI(y_array, preds_array, scoring="rmse")
                cell_value = f"{model_rmse:.3f} [{model_rmse_CI[0]:.3f}, {model_rmse_CI[1]:.3f}]"

            elif metric == "MAE":
                model_mae = mean_absolute_error(y_array, preds_array)
                model_mae_CI = compute_CI(y_array, preds_array, scoring="mae")
                cell_value = f"{model_mae:.3f} [{model_mae_CI[0]:.3f}, {model_mae_CI[1]:.3f}]"

            table_of_scores.loc[model, metric] = cell_value

    return table_of_scores


def compute_pvalues_table(
        predictions_dict,
        y,
        task_type="binary",
        selected_features_dict=None
):
    """Function to output the p-values table for benchmarking.

    Modified version:
    - Handles prediction DataFrames by taking row-wise median.
    - Aligns y, preds, and preds2 by shared sample index.
    - Drops samples where either model has NaN prediction.
    """

    def clean_pair_predictions_for_pvalues(preds, preds2, y, model_name=None, model2_name=None):
        """
        Convert two prediction objects into clean aligned 1D arrays.

        This is needed because CV prediction DataFrames may contain NaN values
        for samples that were not predicted in some folds.
        """

        if isinstance(preds, pd.DataFrame):
            preds_clean = preds.median(axis=1, skipna=True)
        elif isinstance(preds, pd.Series):
            preds_clean = preds
        else:
            preds_clean = pd.Series(np.asarray(preds).ravel())

        if isinstance(preds2, pd.DataFrame):
            preds2_clean = preds2.median(axis=1, skipna=True)
        elif isinstance(preds2, pd.Series):
            preds2_clean = preds2
        else:
            preds2_clean = pd.Series(np.asarray(preds2).ravel())

        if isinstance(y, pd.Series):
            common_index = preds_clean.index.intersection(preds2_clean.index).intersection(y.index)
            y_clean = y.loc[common_index]
            preds_clean = preds_clean.loc[common_index]
            preds2_clean = preds2_clean.loc[common_index]
        else:
            y_clean = pd.Series(np.asarray(y).ravel())
            common_index = preds_clean.index.intersection(preds2_clean.index)
            y_clean = y_clean.loc[common_index]
            preds_clean = preds_clean.loc[common_index]
            preds2_clean = preds2_clean.loc[common_index]

        y_array = np.asarray(y_clean, dtype=float)
        preds_array = np.asarray(preds_clean, dtype=float)
        preds2_array = np.asarray(preds2_clean, dtype=float)

        valid_mask = (
            ~np.isnan(y_array)
            & ~np.isnan(preds_array)
            & ~np.isnan(preds2_array)
        )

        if valid_mask.sum() < len(valid_mask):
            print("\n========== Warning: NaN predictions detected in p-value computation ==========")
            print("Model 1:", model_name)
            print("Model 2:", model2_name)
            print("Number of samples:", len(valid_mask))
            print("Samples dropped:", len(valid_mask) - valid_mask.sum())

        return y_array[valid_mask], preds_array[valid_mask], preds2_array[valid_mask]

    scores_columns = []
    if selected_features_dict is not None:
        if task_type == "binary":
            scores_columns = ["ROC AUC", "Average Precision", "N features", "CVS"]

        elif task_type == "regression":
            scores_columns = ["Prediction", "N features", "CVS"]

    else:
        if task_type == "binary":
            scores_columns = ["ROC AUC", "Average Precision"]

        elif task_type == "regression":
            scores_columns = ["Prediction"]

    p_values_dict = {
        s: pd.DataFrame(
            columns=predictions_dict.keys(),
            index=predictions_dict.keys()
        )
        for s in scores_columns
    }

    for metric in scores_columns:
        p_values_df = p_values_dict[metric]

        for model, preds in predictions_dict.items():
            for model2, preds2 in predictions_dict.items():

                if metric == "ROC AUC":
                    y_array, preds_array, preds2_array = clean_pair_predictions_for_pvalues(
                        preds=preds,
                        preds2=preds2,
                        y=y,
                        model_name=model,
                        model2_name=model2
                    )

                    p_value = permutation_test_between_clfs(
                        y_array,
                        preds_array,
                        preds2_array,
                        scoring="roc_auc"
                    )[1]

                elif metric == "Average Precision":
                    y_array, preds_array, preds2_array = clean_pair_predictions_for_pvalues(
                        preds=preds,
                        preds2=preds2,
                        y=y,
                        model_name=model,
                        model2_name=model2
                    )

                    p_value = permutation_test_between_clfs(
                        y_array,
                        preds_array,
                        preds2_array,
                        scoring="average_precision"
                    )[1]

                elif metric == "N features":
                    sel_features = selected_features_dict[model]["Fold nb of features"]
                    sel_features2 = selected_features_dict[model2]["Fold nb of features"]

                    p_value = mannwhitneyu(
                        x=sel_features,
                        y=sel_features2
                    ).pvalue

                elif metric == "CVS":
                    jaccard_mat = jaccard_matrix(
                        selected_features_dict[model]["Fold selected features"],
                        remove_diag=False
                    )
                    jaccard_val = jaccard_mat[np.triu_indices_from(
                        jaccard_mat,
                        k=1
                    )]

                    jaccard_mat2 = jaccard_matrix(
                        selected_features_dict[model2]["Fold selected features"],
                        remove_diag=False
                    )
                    jaccard_val2 = jaccard_mat2[np.triu_indices_from(
                        jaccard_mat2,
                        k=1
                    )]

                    p_value = mannwhitneyu(
                        x=jaccard_val,
                        y=jaccard_val2
                    ).pvalue

                else:
                    y_array, preds_array, preds2_array = clean_pair_predictions_for_pvalues(
                        preds=preds,
                        preds2=preds2,
                        y=y,
                        model_name=model,
                        model2_name=model2
                    )

                    p_value = mannwhitneyu(
                        x=preds_array,
                        y=preds2_array
                    ).pvalue

                p_values_df.loc[model, model2] = p_value

    return p_values_dict


def compute_features_table(
        selected_features_dict,
        X_train,
        y_train,
        X_test=None,
        y_test=None,
        task_type="binary"
):
    """

    Parameters
    ----------
    selected_features_dict

    X_train: pd.DataFrame
        Training input dataframe

    y_train: pd.Series
        Training outcome pandas Series.

    X_test: pd.DataFrame, default=None
        Testing input dataframe

    y_test: pd.Series, default=None
        Testing outcome pandas Series

    task_type: str, default="binary"
        task type "binary" for binary classification, "regression" for regression.

    Returns
    -------

    """
    all_features = []
    for model, el in selected_features_dict.items():
        all_features += list(selected_features_dict[model])

    all_features = np.unique(all_features)

    df_out = pd.DataFrame(
        data=False,
        index=all_features,
        columns=[f"Selected by {model}" for model in selected_features_dict.keys()])

    for model in selected_features_dict.keys():
        df_out.loc[selected_features_dict[model], f"Selected by {model}"] = True

    if task_type == "binary":
        df_out["Train Mannwithney pvalues"] = [mannwhitneyu(
            X_train.loc[y_train == 0, i],
            X_train.loc[y_train == 1, i],
            nan_policy="omit")[1] for i in all_features]

        df_out["Train T-test pvalues"] = [stats.ttest_ind(
            X_train.loc[y_train == 1, i],
            X_train.loc[y_train == 0, i],
            nan_policy="omit")[1] for i in all_features]

    elif task_type == "regression":
        df_out["Train Pearson-r pvalues"] = [stats.pearsonr(
            X_train.loc[:, i].dropna(),
            y_train.loc[X_train.loc[:, i].dropna().index])[1] for i in all_features]

        df_out["Train Spearman-r pvalues"] = [stats.spearmanr(
            X_train.loc[:, i].dropna(),
            y_train.loc[X_train.loc[:, i].dropna().index])[1] for i in all_features]

    if X_test is not None:
        if task_type == "binary":
            df_out["Test Mannwithney pvalues"] = [mannwhitneyu(X_test.loc[y_test == 0, i],
                                                               X_test.loc[y_test == 1, i],
                                                               nan_policy="omit")[1]
                                                  for i in all_features]

            df_out["Test T-test pvalues"] = [stats.ttest_ind(X_test.loc[y_test == 1, i],
                                                             X_test.loc[y_test == 0, i],
                                                             nan_policy="omit")[1]
                                             for i in all_features]
        elif task_type == "regression":

            df_out["Test Pearson-R pvalues"] = [stats.pearsonr(X_test.loc[:, i].dropna(),
                                                               y_test.loc[X_test.loc[:, i].dropna().index]
                                                               )[1]
                                                for i in all_features]

            df_out["Test Spearman-R pvalues"] = [stats.spearmanr(X_test.loc[:, i].dropna(),
                                                                 y_test.loc[X_test.loc[:, i].dropna().index]
                                                                 )[1]
                                                 for i in all_features]

    return df_out


class BenchmarkWrapper():
    """Wrapper for benchmarking models with basic implement of necessary methods.
    """

    def __init__(self, model, fit=None, predict=None, use_predict_proba=True, get_support=None, get_importances=None, threshold=1e-5) -> None:
        """Initiate the wrapper.

        Parameters
        ----------
        model : sklearn estimator
            Model to wrap.
        fit : function, optional
            If provided, it is the function used as the fit function of the wrapper.
            It is directly the called function, so you can access to the model by
            'self.model' or with the fit function of the model directly.
            If None, the wrapper will automatically take the 'fit' function of the model,
            or throws an error. By default, it is set to None.

        predict : function, optional
            If provided, it is the function used as the predict function of the wrapper.
            It is directly the called function, so you can access to the model by
            'self.model' or with the predict function of the model directly.
            If None, the wrapper will automatically take the 'predict' function of the model,
            or throws an error. By default, it is set to None.

        get_support : function, optional
            If provided, it is the function used as the get_support function of the wrapper. 
            It is directly the called function, so you can access to the model by
            'self.model' or with the get_support function of the model directly. 
            If None, the wrapper will automatically take the 'get_support' function of the model, 
            or determine the support with the 'get_importances' function like in 
            SelectFromModel class of sklearn, or throws an error. 
            By default, it is set to None.

        get_importances : function, optional
            If provided, it is the function used as the 'get_importances' function of the wrapper. 
            It is directly the called function, so you can access to the model by
            'self.model' or with the 'get_importances' function of the model directly. 
            If None, the wrapper will automatically take the 'get_importances' function of the model, 
            or determine the feature importances thanks to the '_get_feature_importances' function of 
            sklearn (get with coef_ or feature_importances_ attribute of model) and take the absolute value, 
            or throws an error. By default, it is set to None.

        threshold : str or float, optional
            The threshold value to use for feature selection if 'get_support' is not provided or implemented in the model. 
            Features whose absolute importance value is greater or equal are kept while the others are discarded.
            If “median” (resp. “mean”), then the threshold value is the median (resp. the mean) of the feature importances.
            A scaling factor (e.g., “1.25*mean”) may also be used. 
            If None and if the estimator has a parameter penalty set to l1, 
            either explicitly or implicitly (e.g, Lasso), the threshold used is 1e-5. 
            Otherwise, “mean” is used by default., by default None

        use_predict_proba : bool, optional
            If True, the predict_proba function is used for the predict function.
        """
        self.model = model
        self.threshold = threshold
        if fit is None:
            fit = getattr(self.model, "fit", None)
        if fit is None:
            raise NotImplemented(
                "The model does not have the method 'fit'. Please provide it.")
        self._fit = fit
        if predict is None:
            if use_predict_proba:
                if hasattr(self.model, "predict_proba"):
                    def predict(x): return getattr(self.model, "predict_proba")(x)[:, 1].flatten()
            else:
                predict = getattr(self.model, "predict", None)
        self._predict = predict
        self._set_attr("get_importances", get_importances)
        self._set_attr("get_support", get_support)

    def _set_attr(self, attr, value):
        """Set the attribute of the wrapper with the value provided or the attribute of the model.
        """
        if value is None:
            value = getattr(self.model, attr, None)
        if value is None:
            value = getattr(self, f"_{attr}", None)
        if value is not None:
            setattr(self, attr, value)
        else:
            raise NotImplemented(
                f"The model does not have the method '{attr}'. Please provide it.")

    def fit(self, X, y, **kwargs):
        """Fit the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.
        y : array-like of shape (n_samples,)
            The target values.
        """
        self._fit(X, y, **kwargs)
        return self

    def predict(self, X, **kwargs):
        """Predict using the model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        """
        if self._predict is None:
            raise NotImplemented(
                "The model does not have the method 'predict'. Please provide it.")
        res = self._predict(X, **kwargs)
        return res

    def _get_importances(self):
        """ Get the feature importances of the model and return the absolute value. """
        try:
            scores = _get_feature_importances(
                estimator=self.model.best_estimator_ if isinstance(self.model, GridSearchCV) else self.model, getter="auto", transform_func=None).flatten()
            res = np.abs(scores)
            return res
        except ValueError:
            raise NotImplemented(
                f"The model does not have the method 'get_importances' \
                    and there is no way to retreive the feature importances. \
                        Please provide it.")

    def _get_support(self, indices=False):
        """ Get the support of the model. """
        scores = self.get_importances()
        threshold = _calculate_threshold(self.model, scores, self.threshold)
        support = scores > threshold
        if indices:
            return np.where(support)[0]
        return support.astype(int)
