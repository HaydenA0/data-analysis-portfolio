import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os


def plot(data, save_dir="Project/figures", filename="all_diabetes_plots.png"):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(5, 3, figsize=(20, 30))
    axes = axes.flatten()

    # --- Plotting Code ( 그대로 유지 ) ---
    sns.countplot(
        x="diabetes",
        data=data,
        ax=axes[0],
        palette="viridis",
        hue="diabetes",
        legend=False,
    )
    axes[0].set_title("Distribution of Diabetes Status (0=No, 1=Yes)")
    axes[0].set_xlabel("Diabetes Status")

    sns.histplot(data=data, x="age", ax=axes[1], color="skyblue", bins=30)
    axes[1].set_title("Age Distribution")

    # Handle potential infinite values in 'bmi' before plotting
    # Make a copy to avoid potential SettingWithCopyWarning if 'data' is a slice
    data_copy = data.copy()
    data_copy["bmi"] = data_copy["bmi"].replace([np.inf, -np.inf], np.nan)
    # Filter out NaNs for the histplot specifically
    bmi_finite_data = data_copy.dropna(subset=["bmi"])
    sns.histplot(data=bmi_finite_data, x="bmi", ax=axes[2], color="lightcoral", bins=30)
    axes[2].set_title("BMI Distribution")
    axes[2].set_xlabel("BMI")

    sns.histplot(data=data, x="HbA1c_level", kde=True, ax=axes[3], color="lightgreen")
    axes[3].set_title("HbA1c Level Distribution")

    sns.histplot(data=data, x="blood_glucose_level", kde=True, ax=axes[4], color="gold")
    axes[4].set_title("Blood Glucose Level Distribution")

    sns.countplot(
        y="smoking_history",
        data=data,
        ax=axes[5],
        palette="Spectral",
        order=data["smoking_history"].value_counts().index,
        # hue="smoking_history", # Redundant for countplot on y, removed
        # legend=False # Removed as hue was removed
    )
    axes[5].set_title("Smoking History Distribution")
    axes[5].tick_params(axis="y", rotation=0)

    sns.boxplot(
        x="diabetes",
        y="age",
        data=data,
        ax=axes[6],
        palette="coolwarm",
        hue="diabetes",
        legend=False,
    )
    axes[6].set_title("Age Distribution by Diabetes Status")
    axes[6].set_xlabel("Diabetes Status (0=No, 1=Yes)")

    # Use the cleaned data_copy for BMI boxplot as well
    sns.boxplot(
        x="diabetes",
        y="bmi",
        data=data_copy,  # Use data_copy where inf/-inf are replaced by NaN
        ax=axes[7],
        palette="coolwarm",
        hue="diabetes",
        legend=False,
    )
    axes[7].set_title("BMI Distribution by Diabetes Status")
    axes[7].set_xlabel("Diabetes Status (0=No, 1=Yes)")

    sns.boxplot(
        x="diabetes",
        y="HbA1c_level",
        data=data,
        ax=axes[8],
        palette="coolwarm",
        hue="diabetes",
        legend=False,
    )
    axes[8].set_title("HbA1c Level by Diabetes Status")
    axes[8].set_xlabel("Diabetes Status (0=No, 1=Yes)")

    sns.boxplot(
        x="diabetes",
        y="blood_glucose_level",
        data=data,
        ax=axes[9],
        palette="coolwarm",
        hue="diabetes",
        legend=False,
    )
    axes[9].set_title("Blood Glucose Level by Diabetes Status")
    axes[9].set_xlabel("Diabetes Status (0=No, 1=Yes)")

    sns.countplot(
        x="smoking_history",
        hue="diabetes",
        data=data,
        ax=axes[10],
        palette="pastel",
        order=data["smoking_history"].value_counts().index,
    )
    axes[10].set_title("Smoking History by Diabetes Status")
    axes[10].tick_params(axis="x", rotation=45)
    axes[10].set_xlabel("Smoking History")
    # More robust legend mapping
    handles, labels = axes[10].get_legend_handles_labels()
    new_labels = [
        "No" if label == "0" else "Yes" for label in labels
    ]  # Assuming diabetes is 0/1
    axes[10].legend(handles=handles, labels=new_labels, title="Diabetes")

    numerical_cols = [
        "age",
        "bmi",
        "HbA1c_level",
        "blood_glucose_level",
        "hypertension",
        "heart_disease",
        "diabetes",
    ]
    # Use data_copy for correlation matrix to handle potential inf/-inf in bmi
    corr_data = data_copy[numerical_cols].dropna()
    correlation_matrix = corr_data.corr()
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="vlag",
        fmt=".2f",
        linewidths=0.5,
        ax=axes[11],
    )
    axes[11].set_title("Correlation Matrix (Numerical Features)")
    axes[11].tick_params(axis="x", rotation=45)
    axes[11].tick_params(axis="y", rotation=0)

    gender_diabetes_pct = data.groupby("gender")["diabetes"].mean().reset_index()
    gender_diabetes_pct["diabetes"] *= 100
    sns.barplot(
        x="gender",
        y="diabetes",
        data=gender_diabetes_pct,
        ax=axes[12],
        palette="Blues",
        hue="gender",
        legend=False,
    )
    axes[12].set_title("Diabetes Percentage by Gender")
    axes[12].set_ylabel("Diabetes (%)")
    axes[12].set_xlabel("Gender")
    for container in axes[12].containers:
        axes[12].bar_label(container, fmt="%.1f%%")

    hypertension_diabetes_pct = (
        data.groupby("hypertension")["diabetes"].mean().reset_index()
    )
    hypertension_diabetes_pct["diabetes"] *= 100
    hypertension_diabetes_pct["hypertension_label"] = hypertension_diabetes_pct[
        "hypertension"
    ].map({0: "No", 1: "Yes"})  # Create label column
    sns.barplot(
        x="hypertension_label",  # Plot using the label column
        y="diabetes",
        data=hypertension_diabetes_pct,
        ax=axes[13],
        palette="Reds",
        hue="hypertension_label",  # Hue using the label column
        legend=False,
    )
    axes[13].set_title("Diabetes Percentage by Hypertension")
    axes[13].set_ylabel("Diabetes (%)")
    axes[13].set_xlabel("Hypertension Status")
    for container in axes[13].containers:
        axes[13].bar_label(container, fmt="%.1f%%")

    heart_disease_diabetes_pct = (
        data.groupby("heart_disease")["diabetes"].mean().reset_index()
    )
    heart_disease_diabetes_pct["diabetes"] *= 100
    heart_disease_diabetes_pct["heart_disease_label"] = heart_disease_diabetes_pct[
        "heart_disease"
    ].map({0: "No", 1: "Yes"})  # Create label column
    sns.barplot(
        x="heart_disease_label",  # Plot using the label column
        y="diabetes",
        data=heart_disease_diabetes_pct,
        ax=axes[14],
        palette="Greens",
        hue="heart_disease_label",  # Hue using the label column
        legend=False,
    )
    axes[14].set_title("Diabetes Percentage by Heart Disease")
    axes[14].set_ylabel("Diabetes (%)")
    axes[14].set_xlabel("Heart Disease Status")
    for container in axes[14].containers:
        axes[14].bar_label(container, fmt="%.1f%%")

    # --- Saving Part ---
    plt.tight_layout()  # Adjust layout first

    # 1. Ensure the target directory exists
    # Using os:
    os.makedirs(save_dir, exist_ok=True)
    # Using pathlib (alternative):
    # Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 2. Construct the full path to the file
    # Using os:
    full_save_path = os.path.join(save_dir, filename)
    # Using pathlib (alternative):
    # full_save_path = Path(save_dir) / filename

    # 3. Save the figure
    # Use bbox_inches='tight' to prevent labels/titles from being cut off
    # Use dpi for controlling resolution (dots per inch)
    try:
        fig.savefig(full_save_path, bbox_inches="tight", dpi=300)
        print(f"Successfully saved plot to: {full_save_path}")
    except Exception as e:
        print(f"Error saving plot to {full_save_path}: {e}")

    # Optional: Close the figure after saving and showing to free up memory
    plt.close(fig)


def plot_xgboost_tree(
    model,
    feature_names=None,
    num_trees=0,
    figsize=(20, 15),
    rankdir="LR",
    save_path=None,
    **kwargs,
):
    """
    Plots a specified tree from a trained XGBoost model.

    Handles importing xgboost.plot_tree, creating a figure, plotting the tree,
    optionally saving the plot, and displaying it. Includes error handling
    for missing library, incorrect model type, plotting errors, and saving errors.

    Args:
        model: Trained XGBoost model object (e.g., XGBRegressor, XGBClassifier).
               Must have a `get_booster` method.
        feature_names (list or similar, optional): List of feature names corresponding
            to the columns in the training data. Used to label nodes in the tree plot.
            Defaults to None, in which case generic feature names (f0, f1, ...) might be used.
        num_trees (int, optional): The index of the specific tree within the XGBoost
            ensemble to plot. Defaults to 0 (the first tree).
        figsize (tuple, optional): The size (width, height) of the matplotlib figure
            in inches. Defaults to (20, 15).
        rankdir (str, optional): The orientation of the tree plot. 'LR' for left-to-right,
            'TB' for top-to-bottom. Defaults to "LR".
        save_path (str, optional): The full file path (including directory and filename,
            e.g., 'plots/xgb_tree_0.png') where the plot image should be saved.
            If None, the plot is not saved to a file. Defaults to None.
            The directory will be created if it doesn't exist.
        **kwargs: Additional keyword arguments passed directly to the
            `xgboost.plot_tree` function.

    Returns:
        tuple: A tuple containing:
            - fig (matplotlib.figure.Figure or None): The matplotlib Figure object
              containing the plot, or None if plotting fails.
            - ax (matplotlib.axes.Axes or None): The matplotlib Axes object
              containing the plot, or None if plotting fails.
        Returns (None, None) if the xgboost library is not installed, if the model
        is not a valid XGBoost model, or if any error occurs during plotting.
    """
    try:
        from xgboost import plot_tree
    except ImportError:
        print("Error: xgboost library is required to use plot_xgboost_tree.")
        print("Install it using: pip install xgboost")
        return None, None

    if not hasattr(model, "get_booster"):
        print(
            "Error: The provided model object does not appear to be a standard XGBoost model."
        )
        return None, None

    fig, ax = plt.subplots(figsize=figsize)

    try:
        plot_tree(
            model,
            num_trees=num_trees,
            rankdir=rankdir,
            ax=ax,
            feature_names=feature_names,
            **kwargs,
        )
        plt.title(f"XGBoost Tree #{num_trees} (Layout: {rankdir})")

        if save_path:
            try:
                save_dir = os.path.dirname(save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                fig.savefig(save_path, bbox_inches="tight", dpi=300)
                print(f"Successfully saved XGBoost tree plot to: {save_path}")
            except Exception as e:
                print(f"Error saving XGBoost tree plot to {save_path}: {e}")

        return fig, ax

    except Exception as e:
        print(f"Error plotting XGBoost tree: {e}")
        plt.close(fig)
        return None, None


def analyze_logistic_regression(model, feature_names=None):
    """
    Performs basic analysis of a trained scikit-learn Logistic Regression model.

    Prints the intercept(s) and coefficients (representing change in log-odds)
    for the model. Handles both binary and multi-class classification cases.
    If feature names are provided and match the number of coefficients,
    it displays the coefficients alongside their corresponding feature names.

    Args:
        model: A trained scikit-learn LogisticRegression model object. It should
               have `coef_` and `intercept_` attributes.
        feature_names (list or pd.Index, optional): A list or pandas Index of
            feature names corresponding to the columns of the data used to train
            the model. The order should match the order of features in the
            training data. If provided and the length matches the number of
            coefficients, the output will include feature names. Defaults to None.

    Returns:
        None: This function prints the analysis results directly to the console.
              It does not return any value.
    """
    if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
        print(
            "Error: Model does not have 'coef_' or 'intercept_' attributes. "
            "Expected a trained scikit-learn Logistic Regression model."
        )
        return
    if not hasattr(model, "classes_"):
        print(
            "Warning: Model does not have 'classes_' attribute. Output interpretation might be limited."
        )

    print("--- Logistic Regression Analysis ---")

    if model.coef_.shape[0] > 1:
        print(f"Model appears to be multi-class (found {model.coef_.shape[0]} classes)")
        if hasattr(model, "classes_"):
            print(f"Classes: {model.classes_}")

        for i, class_intercept in enumerate(model.intercept_):
            class_label = (
                model.classes_[i]
                if hasattr(model, "classes_")
                and len(model.classes_) == model.coef_.shape[0]
                else f"Class {i}"
            )
            print(f"\nIntercept for {class_label}: {class_intercept:.4f}")
            class_coef = model.coef_[i]

            if feature_names is not None:
                if len(feature_names) == len(class_coef):
                    print(f"Coefficients (Log-Odds) for {class_label}:")
                    coef_df = pd.DataFrame(
                        {"Feature": feature_names, "Coefficient": class_coef}
                    )
                    print(coef_df.round(4).to_string(index=False))
                else:
                    # Warning if mismatch in number of names and coefficients
                    print(
                        f"Warning: Number of feature names ({len(feature_names)}) does not "
                        f"match number of coefficients ({len(class_coef)}) for {class_label}. "
                        "Printing coefficients without names."
                    )
                    print(f"Coefficients for {class_label}: {np.round(class_coef, 4)}")
            else:
                # Print coefficients without names if feature_names is None
                print(f"Coefficients for {class_label}: {np.round(class_coef, 4)}")
    else:  # Binary classification case (or only one set of coefficients)
        print(f"Intercept: {model.intercept_[0]:.4f}")
        coefficients = model.coef_[0]  # Get the single array of coefficients

        if feature_names is not None:
            # Check if number of names matches coefficients
            if len(feature_names) == len(coefficients):
                print("Coefficients (Log-Odds):")
                # Create DataFrame for better readability
                coef_df = pd.DataFrame(
                    {"Feature": feature_names, "Coefficient": coefficients}
                )
                print(coef_df.round(4).to_string(index=False))
                # Optional: Calculate and display Odds Ratios for binary case
                # coef_df['Odds_Ratio'] = np.exp(coefficients)
                # print("\nOdds Ratios:")
                # print(coef_df[['Feature', 'Odds_Ratio']].round(4).to_string(index=False))
            else:
                # Warning if mismatch
                print(
                    f"Warning: Number of feature names ({len(feature_names)}) does not "
                    f"match number of coefficients ({len(coefficients)}). Printing coefficients without names."
                )
                print(f"Coefficients (Log-Odds): {np.round(coefficients, 4)}")
        else:
            # Print coefficients without names
            print(f"Coefficients (Log-Odds): {np.round(coefficients, 4)}")

    print("------------------------------------\n")


def plot_analyze_decision_tree(
    model,
    feature_names=None,
    class_names=None,
    figsize=(25, 15),
    save_dir="Project/figures",
    filename="decision_tree_plot.png",
    **kwargs,
):
    """
    Plots a trained scikit-learn Decision Tree model and saves the plot.
    Also prints basic tree statistics (maximum depth, number of leaves).

    Uses `sklearn.tree.plot_tree` for visualization. Handles importing
    scikit-learn, checking model type, creating the plot, saving the plot,
    and displaying it. Includes error handling for missing library,
    incorrect model type, plotting errors, and saving errors.

    Args:
        model: Trained scikit-learn DecisionTreeClassifier or DecisionTreeRegressor
               object. Must have a `tree_` attribute (indicating it's fitted).
        feature_names (list or pd.Index, optional): List or Index of feature names
            corresponding to the columns in the training data. Used to label
            split nodes in the tree plot. Defaults to None.
        class_names (list, optional): List of strings representing the names of
            the target classes in a classification task. The order should match
            `model.classes_`. Used to label the majority class in leaf nodes.
            Ignored for regression trees. Defaults to None.
        figsize (tuple, optional): The size (width, height) of the matplotlib
            figure in inches. Defaults to (25, 15).
        save_dir (str, optional): The directory path where the plot image will
            be saved. If the directory does not exist, it will be created.
            Defaults to "Project/figures". If None or empty, the plot won't be saved.
        filename (str, optional): The name of the file (including extension, e.g.,
            '.png', '.pdf', '.jpg') to save the plot as within the `save_dir`.
            Defaults to "decision_tree_plot.png". If None or empty, the plot won't be saved.
        **kwargs: Additional keyword arguments passed directly to the
            `sklearn.tree.plot_tree` function (e.g., `max_depth` to limit
            plotting depth, `filled=True`, `rounded=True`, `fontsize`).

    Returns:
        tuple: A tuple containing:
            - fig (matplotlib.figure.Figure or None): The matplotlib Figure object
              containing the plot, or None if plotting fails or prerequisites are not met.
            - ax (matplotlib.axes.Axes or None): The matplotlib Axes object
              containing the plot, or None if plotting fails or prerequisites are not met.
        Returns (None, None) if scikit-learn is not installed, if the model is not
        a valid Decision Tree, if the model is not trained, or if any error occurs
        during plotting or saving.
    """
    try:
        from sklearn.tree import plot_tree as sklearn_plot_tree
        # Import base classes for type checking
        # from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor # Already imported above
    except ImportError:
        print(
            "Error: scikit-learn library is required to use plot_analyze_decision_tree."
        )
        print("Install it using: pip install scikit-learn")
        return None, None
    # Check if the model has been fitted (presence of the 'tree_' attribute)
    if not hasattr(model, "tree_"):
        print("Error: Model does not have 'tree_' attribute. Is it trained?")
        return None, None

    print("--- Decision Tree Analysis & Plot ---")
    # Attempt to get tree statistics
    try:
        depth = model.get_depth()
        leaves = model.get_n_leaves()
        print(f"Maximum Depth: {depth}")
        print(f"Number of Leaves: {leaves}")
    except Exception as e:
        # Catch potential errors if methods are unavailable (shouldn't happen with checked types)
        print(f"Could not retrieve tree depth/leaves: {e}")
        depth = "N/A"  # Set defaults for title if stats fail
        leaves = "N/A"

    fig, ax = plt.subplots(figsize=figsize)
    try:
        # Set up default plot arguments, allowing overrides via kwargs
        plot_kwargs = {
            "filled": True,  # Color nodes by majority class/value
            "rounded": True,  # Use rounded boxes for nodes
            "feature_names": feature_names,  # Use provided feature names
            "class_names": class_names,  # Use provided class names (for classification)
            "ax": ax,  # Plot on the created axes
            "fontsize": 10,  # Default font size (can be overridden by kwargs)
        }
        # Update defaults with any user-provided kwargs
        plot_kwargs.update(kwargs)

        # Plot the tree using scikit-learn's function
        sklearn_plot_tree(model, **plot_kwargs)
        plt.title(f"Decision Tree Visualization (Depth: {depth}, Leaves: {leaves})")

        # Save the figure if path and filename are provided
        if save_dir and filename:
            full_save_path = os.path.join(save_dir, filename)
            try:
                # Create the directory if it doesn't exist
                os.makedirs(save_dir, exist_ok=True)
                # Save the figure; bbox_inches='tight' avoids cutting off labels
                fig.savefig(full_save_path, bbox_inches="tight", dpi=300)
                print(f"Successfully saved Decision Tree plot to: {full_save_path}")
            except Exception as e:
                print(f"Error saving Decision Tree plot to {full_save_path}: {e}")
        else:
            print("Plot not saved (save_dir or filename not provided).")

        return fig, ax

    except Exception as e:
        # Catch errors during the plotting process itself
        print(f"Error plotting Decision Tree: {e}")
        plt.close(fig)  # Close the figure if plotting failed
        return None, None


def plot_model_performance_comparison():
    """
    Generates bar plots comparing different models based on performance metrics.

    Takes a DataFrame containing model names and various performance metrics
    (like Accuracy, Training Time, Prediction Time, ROC AUC, Size) and creates
    a figure with multiple subplots, each showing a bar chart comparing the
    models for one metric.

    Args:
        performance_df (pd.DataFrame): A DataFrame where each row represents a
            model and columns represent different performance metrics. Expected
            columns include 'Model' (as string identifier) and numerical metrics
            like 'Accuracy Score %', 'Training Time (ms)', 'Predicting Time (ms)',
            'ROC Auc Score', 'Size (kb)'.

    Returns:
        None: Displays the matplotlib figure with the comparison plots.
    """
    performance_df = pd.DataFrame(
        {
            "Model": [
                "Naive always no",
                "Logistical Regression",
                "XGBoost",
                "Decision Tree",
            ],
            "Training Time (ms)": [0, 5210, 38.8, 22.3],
            "Accuracy Score %": [90.00, 96.01, 97.19, 97.18],
            "Predicting Time (ms)": [0.005, 4.81, 2.9, 6.46],
            "True Positives": [0, 45295, 45727, 45727],
            "False Negatives": [4264, 1562, 1403, 1403],
            "ROC Auc Score": [np.nan, 0.91, 0.98, 0.98],
            "Size (kb)": [0, 1.19, 7.80, 2.00],
        }
    )

    # Define the metrics to plot from the DataFrame columns
    metrics_to_plot = [
        "Accuracy Score %",
        "Training Time (ms)",
        "Predicting Time (ms)",
        "ROC Auc Score",
        "Size (kb)",
        # Add other numerical columns you want to plot here
    ]

    # Filter out metrics that are not present in the DataFrame
    metrics_present = [m for m in metrics_to_plot if m in performance_df.columns]

    if not metrics_present:
        print("No plottable metrics found in the DataFrame columns.")
        return

    n_metrics = len(metrics_present)
    # Adjust layout based on number of metrics (e.g., 2 columns)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols  # Calculate required rows

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8 * n_cols, 5 * n_rows))
    # Flatten axes array for easy iteration, handles cases where n_rows=1 or n_cols=1
    axes = axes.flatten()

    # Sort DataFrame by Accuracy for potentially better visualization in that plot
    df_sorted_acc = performance_df.sort_values("Accuracy Score %", ascending=False)

    for i, metric in enumerate(metrics_present):
        ax = axes[i]
        # Use the accuracy-sorted df for the accuracy plot, others use original order
        df_to_plot = df_sorted_acc if metric == "Accuracy Score %" else performance_df

        # Create bar plot
        sns.barplot(x="Model", y=metric, data=df_to_plot, ax=ax, palette="viridis")

        ax.set_title(f"Comparison by {metric}")
        ax.set_xlabel("Model")
        ax.set_ylabel(metric)
        ax.tick_params(
            axis="x", rotation=45, labelsize=10
        )  # Rotate x-labels for readability

        # Add value labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt="%.2f", fontsize=9, padding=3)

    # Hide any unused subplots if the grid is larger than needed
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.savefig("Project/figures/models_comparaison.png")
