import numpy as np
import pandas as pd


# Function to process the DataFrame and generate LaTeX output with 3 decimal places
def generate_latex_with_bolding(df, drug, return_df=False):
    latex_df = df.copy()

    if drug:
        min_columns = ["L^1_{1}", "L^1_{2}", "DTW"]
    else:
        min_columns = [
            "L^1_{1}",
            "L^1_{2}",
            "\Delta t_{1}",
            "\Delta t_{2}",
            "DTW",
        ]

    for col in min_columns:
        min_value = df[col].min()
        latex_df[col] = df[col].apply(
            lambda x: f"\\textbf{{{x:.3f}}}" if x == min_value else f"{x:.3f}"
        )

    for col in ["R^2_{1}", "R^2_{2}"]:
        max_value = df[col].max()
        latex_df[col] = df[col].apply(
            lambda x: f"\\textbf{{{x:.3f}}}" if x == max_value else f"{x:.3f}"
        )

    if return_df:
        return latex_df
    else:
        return latex_df.to_latex(index=True, escape=False)


# # Function to process DataFrame and include uncertainties
# def generate_latex_with_bolding_and_uncertainty(df, df_std, drug, return_df=False):
#     latex_df = df.copy()

#     if drug:
#         min_columns = ["L^1_{1}", "L^1_{2}", "DTW"]
#     else:
#         min_columns = [
#             "L^1_{1}",
#             "L^1_{2}",
#             "\Delta t_{1}",
#             "\Delta t_{2}",
#             "DTW_{1}",
#             "DTW_{2}",
#         ]

#     # Process minimum values
#     for col in min_columns:
#         if col in df.columns:
#             min_value = df[col].min()
#             latex_df[col] = df[col].apply(
#                 lambda x: f"\\textbf{{{x:.3f}}}" if x == min_value else f"{x:.3f}"
#             )

#     # Process maximum values
#     for col in ["R^2_{1}", "R^2_{2}"]:
#         if col in df.columns:
#             max_value = df[col].max()
#             latex_df[col] = df[col].apply(
#                 lambda x: f"\\textbf{{{x:.3f}}}" if x == max_value else f"{x:.3f}"
#             )

#     # Add uncertainty from df_std
#     for col in df.columns:
#         if col in df_std.columns:  # Ensure matching columns exist in df_std
#             latex_df[col] = (
#                 latex_df[col] + " ± " + df_std[col].apply(lambda x: f"{x:.3f}")
#             )

#     if return_df:
#         return latex_df
#     else:
#         return latex_df.to_latex(index=True, escape=False)


# def generate_latex_with_bolding_v2(df, return_df=False):
#     latex_df = pd.DataFrame()

#     # Process each modality independently
#     for modality in df["modality"].unique():
#         # Filter DataFrame for the current modality
#         sub_df = df[df["modality"] == modality].copy()

#         # Columns to ignore
#         ignore_columns = ["modality"]

#         # Columns where the max value should be bolded
#         max_columns = ["R^2_{1}", "R^2_{2}"]

#         # Apply bolding logic for this modality
#         for col in sub_df.columns:
#             if col in ignore_columns:
#                 continue  # Skip ignored columns

#             if col in max_columns:
#                 max_value = sub_df[col].max()
#                 sub_df[col] = sub_df[col].apply(
#                     lambda x: f"\\textbf{{{x}}}" if x == max_value else f"{x}"
#                 )
#             else:
#                 min_value = sub_df[col].min()
#                 sub_df[col] = sub_df[col].apply(
#                     lambda x: f"\\textbf{{{x}}}" if x == min_value else f"{x}"
#                 )

#         # Append processed subtable
#         latex_df = pd.concat([latex_df, sub_df])

#     if return_df:
#         return latex_df
#     else:
#         return latex_df.to_latex(index=True, escape=False)


def generate_latex_with_bolding_3f(df, return_df=False):
    latex_df = pd.DataFrame()

    # Process each modality independently
    for modality in df["modality"].unique():
        # Filter DataFrame for the current modality
        sub_df = df[df["modality"] == modality].copy()

        # Columns to ignore
        ignore_columns = ["modality"]

        # Columns where the max value should be bolded
        max_columns = ["R^2_{1}", "R^2_{2}"]

        # Apply bolding logic for this modality
        for col in sub_df.columns:
            if col in ignore_columns:
                continue  # Skip ignored columns

            if col in max_columns:
                max_value = sub_df[col].max()
                sub_df[col] = sub_df[col].apply(
                    lambda x: f"\\textbf{{{x:.3f}}}" if x == max_value else f"{x:.3f}"
                )
            else:
                min_value = sub_df[col].min()
                sub_df[col] = sub_df[col].apply(
                    lambda x: f"\\textbf{{{x:.3f}}}" if x == min_value else f"{x:.3f}"
                )

        # Append processed subtable
        latex_df = pd.concat([latex_df, sub_df])

    if return_df:
        return latex_df
    else:
        return latex_df.to_latex(index=True, escape=False)


def generate_latex_with_bolding_1f(df, return_df=False):
    latex_df = pd.DataFrame()

    # Process each modality independently
    for modality in df["modality"].unique():
        # Filter DataFrame for the current modality
        sub_df = df[df["modality"] == modality].copy()

        # Columns to ignore
        ignore_columns = ["modality"]

        # Columns where the max value should be bolded
        max_columns = ["R^2_{1}", "R^2_{2}"]

        # Apply bolding logic for this modality
        for col in sub_df.columns:
            if col in ignore_columns:
                continue  # Skip ignored columns

            if col in max_columns:
                max_value = sub_df[col].max()
                sub_df[col] = sub_df[col].apply(
                    lambda x: f"\\textbf{{{x:.1f}}}" if x == max_value else f"{x:.1f}"
                )
            else:
                min_value = sub_df[col].min()
                sub_df[col] = sub_df[col].apply(
                    lambda x: f"\\textbf{{{x:.1f}}}" if x == min_value else f"{x:.1f}"
                )

        # Append processed subtable
        latex_df = pd.concat([latex_df, sub_df])

    if return_df:
        return latex_df
    else:
        return latex_df.to_latex(index=True, escape=False)
