import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings("ignore")


def load_data(data_path):
    data = pd.read_csv(data_path)
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)

    data_without_C = data.drop(columns=["SOC (%)","Capacity (mAh)","Time"])
    data_without_C_T = data.drop(columns=["SOC (%)","Capacity (mAh)","Time","Temp (C)"])

    return  data_without_C, data_without_C_T, data


def model_error_analysis(X_train, X_test, y_train, y_test):
    # Use a pipeline with StandardScaler for all models
    models = {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", RandomForestRegressor(n_estimators=100, random_state=42))
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", GradientBoostingRegressor(n_estimators=100, random_state=42))
        ]),
        "AdaBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", AdaBoostRegressor(n_estimators=100, random_state=42))
        ]),
        "Linear Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", LinearRegression())
        ]),
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Ridge())
        ]),
        "Lasso Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", Lasso())
        ]),
        "Support Vector Regressor": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", SVR())
        ]),
        "K-Neighbors Regressor": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", KNeighborsRegressor())
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", DecisionTreeRegressor(random_state=42))
        ]),
        "Dummy Regressor": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", DummyRegressor(strategy="mean"))
        ])
    }

    error_analysis = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        error_analysis.append({
            "Model": name,
            "MSE": mse,
            "MAE": mae,
            "R2 Score": r2
        })
    return pd.DataFrame(error_analysis), fitted_models


def visualize_model_performance(error_df, X_test, y_test, models, file_name):
    os.makedirs("outputs", exist_ok=True)

    # Select best model
    least_error_model_name = error_df.loc[error_df['MSE'].idxmin(), 'Model']
    least_error_model = models[least_error_model_name]
    y_pred = least_error_model.predict(X_test)
    residuals = y_test - y_pred

    # Create a combined figure with 3 rows
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))

    # --- Error Metrics ---
    sns.barplot(x="MSE", y="Model", data=error_df, ax=axes[0], palette="viridis")
    for i, v in enumerate(error_df["MSE"]):
        axes[0].text(v, i, f'{v:.2e}', color='black', va='center')
    axes[0].set_title('Mean Squared Error (MSE)')

    sns.barplot(x="MAE", y="Model", data=error_df, ax=axes[0].twinx(), palette="mako", alpha=0.5)
    axes[0].set_title('Error Metrics (MSE & MAE)')

    # --- Actual vs Predicted ---
    axes[1].scatter(y_test, y_pred, alpha=0.5)
    axes[1].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                 'r--', lw=2, label='Ideal')
    axes[1].set_xlabel('Actual')
    axes[1].set_ylabel('Predicted')
    axes[1].set_title(f'Actual vs Predicted Values for {least_error_model_name}')
    axes[1].legend()

    # --- Residual Plot ---
    sns.scatterplot(x=y_pred, y=residuals, ax=axes[2])
    axes[2].axhline(y=0, color='r', linestyle='--', label='Zero Residuals')
    axes[2].set_xlabel('Predicted Values')
    axes[2].set_ylabel('Residuals')
    axes[2].set_title(f'Residual Plot for {least_error_model_name}')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(f"outputs/{file_name}_combined_plot.png")
    plt.close()

def plot_feature_importance(model, feature_names, file_name, title_suffix=""):
    """
    Plot and save feature importances for tree-based models or coefficients for linear models.
    """
    import numpy as np

    # Try to get the regressor from the pipeline
    reg = model
    if hasattr(model, "named_steps") and "reg" in model.named_steps:
        reg = model.named_steps["reg"]

    # Try to get feature importances or coefficients
    if hasattr(reg, "feature_importances_"):
        importances = reg.feature_importances_
        method = "Feature Importance"
    elif hasattr(reg, "coef_"):
        importances = np.abs(reg.coef_)
        method = "Absolute Coefficient"
    else:
        print(f"Feature importance not available for model: {type(reg).__name__}")
        return

    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=sorted_importances, y=sorted_features, palette="crest")
    plt.title(f"{method} for {type(reg).__name__} {title_suffix}")
    plt.xlabel(method)
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"outputs/{file_name}_feature_importance.png")
    plt.close()
    print(f"Saved feature importance plot to outputs/{file_name}_feature_importance.png")
    # Also print the importances
    print("Feature importances:")
    for feat, imp in zip(sorted_features, sorted_importances):
        print(f"  {feat}: {imp:.4f}")

if __name__ == "__main__":
    import re
    file_names = ["full2empty.csv","full2half.csv","full2low_bad.csv"]
    os.makedirs("outputs", exist_ok=True)
    for file_name in file_names:
        # Remove .csv extension for naming
        name = re.sub(r"\.csv$", "", file_name)
        data_without_C, data_without_C_T, data = load_data(file_name)
        print("="*100)
        print(f"Model performance on {name} using temperature :\n")
        # With temperature
        X_train, X_test, y_train, y_test = train_test_split(data_without_C, data["SOC (%)"], test_size=0.2, random_state=42)
        error_analysis, models = model_error_analysis(X_train, X_test, y_train, y_test)
        visualize_model_performance(error_analysis, X_test, y_test, models, name+"_with_temp")
        print(error_analysis)
        # Save best model (with temp)
        best_model_name = error_analysis.loc[error_analysis['MSE'].idxmin(), 'Model']
        best_model = models[best_model_name]
        joblib.dump(best_model, f"outputs/{name}_with_temp_best_model.joblib")
        print(f"Saved best model with temp: {best_model_name} to outputs/{name}_with_temp_best_model.joblib")
        # Feature importance (with temp)
        print(f"Feature importance for best model ({best_model_name}) with temperature:")
        plot_feature_importance(
            best_model,
            feature_names=list(data_without_C.columns),
            file_name=f"{name}_with_temp",
            title_suffix="(with temp)"
        )
        print("="*100)
        print(f"Model performance on {name} without temperature :\n")
        # Without temperature
        X_train, X_test, y_train, y_test = train_test_split(data_without_C_T, data["SOC (%)"], test_size=0.2, random_state=42)
        error_analysis, models = model_error_analysis(X_train, X_test, y_train, y_test)
        print(error_analysis)
        visualize_model_performance(error_analysis, X_test, y_test, models, name+"_without_temp")
        # Save best model (without temp)
        best_model_name_wo = error_analysis.loc[error_analysis['MSE'].idxmin(), 'Model']
        best_model_wo = models[best_model_name_wo]
        joblib.dump(best_model_wo, f"outputs/{name}_without_temp_best_model.joblib")
        print(f"Saved best model without temp: {best_model_name_wo} to outputs/{name}_without_temp_best_model.joblib")
        # Feature importance (without temp)
        print(f"Feature importance for best model ({best_model_name_wo}) without temperature:")
        plot_feature_importance(
            best_model_wo,
            feature_names=list(data_without_C_T.columns),
            file_name=f"{name}_without_temp",
            title_suffix="(without temp)"
        )