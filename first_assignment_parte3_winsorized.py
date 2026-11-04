"""First Assignment - Part 3: three cross-sectional OLS regressions.

Run from the First_Assignment folder:
    python first_assignment_parte3.py

Requires: pandas numpy statsmodels matplotlib openpyxl
Reads:    dati/First_Assignment_dataset.xlsx
Writes:   output/parte_3_cross_section/
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Save charts without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


# ============================================================
# 1. PATHS - SAME STRUCTURE AS PARTS 1 AND 2
# ============================================================
ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "dati" / "First_Assignment_dataset.xlsx"
OUT = ROOT / "output" / "winsorized" / "parte_3"

if not SOURCE.is_file():
    raise SystemExit(
        "Excel originale non trovato!\n"
        f"Il file deve essere qui: {SOURCE}\n"
        "Rinomina la copia eventualmente chiamata "
        "'First_Assignment_dataset(3).xlsx' in 'First_Assignment_dataset.xlsx'."
    )
OUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. READ THE ORIGINAL MONTHLY DATA AND VALIDATE
# ============================================================
df = pd.read_excel(SOURCE, sheet_name=0, engine="openpyxl")
df.columns = df.columns.astype(str).str.strip()
portfolio_names = [f"XS{i}" for i in range(1, 11)]
factor_names = ["rmrf", "smb", "hml", "umd"]
needed = ["month", "rf"] + factor_names + [f"S{i}" for i in range(1, 11)]
missing = [column for column in needed if column not in df.columns]
if missing:
    raise SystemExit(f"Colonne mancanti nel file Excel: {missing}")

numeric = df[needed[1:]].apply(pd.to_numeric, errors="raise")
if not np.isfinite(numeric.to_numpy(dtype=float)).all():
    raise SystemExit("Il file contiene numeri mancanti o non validi.")
if len(df) != 363:
    print(f"ATTENZIONE: il dataset contiene {len(df)} mesi, non 363.")

# ============================================================
# 3. RECREATE XS1-XS10 (PART 1)
# XS_i(t) = S_i(t) - RF(t)
# ============================================================
xs = pd.DataFrame({
    f"XS{i}": numeric[f"S{i}"] - numeric["rf"]
    for i in range(1, 11)
})

# ============================================================
# 1% / 99% WINSORIZATION OF EXCESS RETURNS ONLY
# Keep the original series unchanged for comparison.
# RMRF, SMB, HML and UMD are NOT winsorized.
# ============================================================
xs_original = xs.copy()
xs = xs_original.clip(
    lower=xs_original.quantile(0.01),
    upper=xs_original.quantile(0.99),
    axis=1,
)

monthly = pd.concat([
    df[["month"]].reset_index(drop=True),
    numeric[factor_names].rename(columns={
        "rmrf": "RMRF", "smb": "SMB", "hml": "HML", "umd": "UMD"
    }).reset_index(drop=True),
    xs.reset_index(drop=True)
], axis=1)

# ============================================================
# 4. RECREATE THE FOUR-FACTOR BETAS (PART 2)
# Each regression has 363 monthly observations.
# ============================================================
time_X = sm.add_constant(numeric[factor_names], has_constant="add")
inputs = []
for i in range(1, 11):
    portfolio = f"XS{i}"
    time_fit = sm.OLS(xs[portfolio], time_X).fit()
    inputs.append({
        "Portfolio": portfolio,
        "Mean_XS_decimal": float(xs[portfolio].mean()),
        "Mean_XS_percent": float(xs[portfolio].mean() * 100),
        "Beta_MKT": float(time_fit.params["rmrf"]),
        "Beta_SMB": float(time_fit.params["smb"]),
        "Beta_HML": float(time_fit.params["hml"]),
        "Beta_UMD": float(time_fit.params["umd"]),
        "TimeSeries_R2": float(time_fit.rsquared),
    })
cross = pd.DataFrame(inputs)

# ============================================================
# 5. RUN THREE CROSS-SECTIONAL OLS REGRESSIONS (PART 3)
# Each regression has 10 portfolios, 2 betas + intercept.
# ============================================================
models = {"MKT + SMB": "Beta_SMB", "MKT + HML": "Beta_HML", "MKT + UMD": "Beta_UMD"}
comparison_rows, detail_rows = [], []
fitted = cross[["Portfolio", "Mean_XS_percent"]].copy()
results = {}

for model_label, second_beta in models.items():
    predictors = sm.add_constant(cross[["Beta_MKT", second_beta]], has_constant="add")
    target = cross["Mean_XS_decimal"]  # monthly decimal returns
    fit = sm.OLS(target, predictors).fit()
    results[model_label] = fit

    if int(fit.nobs) != 10 or int(fit.df_resid) != 7:
        raise RuntimeError("Cross-sectional sample or degrees of freedom are wrong.")

    # The lambda coefficients and their standard errors are shown in
    # percentage points per unit of beta, not in decimal return units.
    comparison_rows.append({
        "Model": model_label,
        "N_portfolios": int(fit.nobs),
        "Residual_df": int(fit.df_resid),
        "Intercept_lambda0_pp": float(fit.params["const"] * 100),
        "Lambda_MKT_pp": float(fit.params["Beta_MKT"] * 100),
        "Lambda_other_pp": float(fit.params[second_beta] * 100),
        "p_MKT": float(fit.pvalues["Beta_MKT"]),
        "p_other": float(fit.pvalues[second_beta]),
        "R_squared": float(fit.rsquared),
        "Adjusted_R_squared": float(fit.rsquared_adj),
    })

    for parameter in ["const", "Beta_MKT", second_beta]:
        ci_low, ci_high = fit.conf_int().loc[parameter]
        detail_rows.append({
            "Model": model_label,
            "Parameter": "Intercept" if parameter == "const" else parameter,
            "Coefficient_pp": float(fit.params[parameter] * 100),
            "Std_Error_pp": float(fit.bse[parameter] * 100),
            "t_statistic": float(fit.tvalues[parameter]),
            "p_value": float(fit.pvalues[parameter]),
            "CI_95_lower_pp": float(ci_low * 100),
            "CI_95_upper_pp": float(ci_high * 100),
            "Significant_5pct": "YES" if fit.pvalues[parameter] < 0.05 else "NO",
        })

    factor_short = second_beta.replace("Beta_", "")
    fitted[f"Fitted_{factor_short}_percent"] = fit.predict(predictors) * 100
    fitted[f"Residual_{factor_short}_pp"] = (
        target - fit.predict(predictors)
    ) * 100

comparison = pd.DataFrame(comparison_rows)
details = pd.DataFrame(detail_rows)

# ============================================================
# 6. EXPORT COMPLETE EXCEL AND CSV RESULTS
# ============================================================
notes = pd.DataFrame([
    ("Data source", "First_Assignment_dataset.xlsx, original first worksheet"),
    ("Sample", "October 1980 - December 2010; 363 months"),
    ("Part 1", "XS_i,t = S_i,t - RF_t; then winsorize each XS at its P1/P99"),
    ("Winsorization", "Only XS1-XS10 clipped; RMRF, SMB, HML, UMD unchanged"),
    ("Part 2", "Four-factor OLS with RMRF, SMB, HML, UMD and an intercept"),
    ("Part 3 dependent variable", "10 average monthly EXCESS returns, one per portfolio"),
    ("Part 3 regressors", "Estimated MKT beta and one of SMB, HML, UMD beta"),
    ("OLS sample size", "10 portfolios per cross-sectional regression"),
    ("Residual degrees of freedom", "10 - 2 predictors - 1 intercept = 7"),
    ("Coefficient_pp", "Monthly percentage points for a one-unit change in beta"),
    ("p-value", "Two-sided standard OLS t-test of coefficient equal to zero"),
    ("R_squared", "Share of variation in the 10 portfolio means explained"),
    ("Excel_* tabs", "Live Excel LINEST formulas; open file in Excel to recalculate"),
    ("Statistical caveat", "Only 10 portfolios; betas are estimated and may be correlated"),
], columns=["Item", "Explanation"])

file_xlsx = OUT / "parte3_risultati_cross_section.xlsx"
with pd.ExcelWriter(file_xlsx, engine="openpyxl") as writer:
    monthly.to_excel(writer, sheet_name="Monthly_Data", index=False)
    cross.to_excel(writer, sheet_name="Cross_Section_Inputs", index=False)
    comparison.to_excel(writer, sheet_name="Model_Comparison", index=False)
    details.to_excel(writer, sheet_name="Coefficient_Tests", index=False)
    fitted.to_excel(writer, sheet_name="Fitted_vs_Actual", index=False)
    notes.to_excel(writer, sheet_name="Read_Me", index=False)

    # Excel verification: same three regressions with live formulas.
    # LINEST returns coefficients in reverse predictor order:
    # second beta, market beta, intercept.
    for suffix, beta_column in [("SMB", "E"), ("HML", "F"), ("UMD", "G")]:
        sheet_name = f"Excel_{suffix}"
        sheet = writer.book.create_sheet(sheet_name)
        sheet.append(["Portfolio", "Beta_MKT", f"Beta_{suffix}", "Mean_XS_decimal"])
        for row in range(2, 12):
            sheet.cell(row, 1, f"='Cross_Section_Inputs'!A{row}")
            sheet.cell(row, 2, f"='Cross_Section_Inputs'!D{row}")
            sheet.cell(row, 3, f"='Cross_Section_Inputs'!{beta_column}{row}")
            sheet.cell(row, 4, f"='Cross_Section_Inputs'!B{row}")
        sheet["F2"] = "Parameter"
        sheet["G2"] = "Coefficient (pp/month)"
        sheet["H2"] = "Std. error (pp/month)"
        sheet["I2"] = "t-statistic"
        sheet["J2"] = "p-value"
        L = "LINEST($D$2:$D$11,$B$2:$C$11,TRUE,TRUE)"
        for row, label, idx in [(3, "Intercept", 3), (4, "Beta_MKT", 2), (5, f"Beta_{suffix}", 1)]:
            sheet[f"F{row}"] = label
            sheet[f"G{row}"] = f"=INDEX({L},1,{idx})*100"
            sheet[f"H{row}"] = f"=INDEX({L},2,{idx})*100"
            sheet[f"I{row}"] = f"=G{row}/H{row}"
            sheet[f"J{row}"] = f"=T.DIST.2T(ABS(I{row}),$G$9)"
        sheet["F7"] = "R-squared"
        sheet["G7"] = f"=INDEX({L},3,1)"
        sheet["F8"] = "Observations"
        sheet["G8"] = "=COUNT(D2:D11)"
        sheet["F9"] = "Residual df"
        sheet["G9"] = "=G8-3"
        sheet["F11"] = "NOTE: G3:G5 are percentage points, G7 is a decimal R2."
        sheet["F12"] = "Main calculated results are in Model_Comparison and Coefficient_Tests."

    # A clean presentation with fixed widths and frozen headers.
    from openpyxl.styles import Font, PatternFill, Alignment
    for sh in writer.book.worksheets:
        sh.freeze_panes = "B2" if sh.title in ("Monthly_Data", "Cross_Section_Inputs") else "A2"
        sh.sheet_view.showGridLines = False
        last_col = sh.max_column
        for cell in sh[1][:last_col]:
            if cell.value is not None:
                cell.fill = PatternFill("solid", fgColor="24358D")
                cell.font = Font(bold=True, color="FFFFFF")
                cell.alignment = Alignment(wrap_text=True)
        sh.row_dimensions[1].height = 29
        for col in sh.columns:
            letter = col[0].column_letter
            max_len = min(43, max((len(str(cell.value)) for cell in col if cell.value is not None), default=10))
            sh.column_dimensions[letter].width = max(15, max_len + 2)
        for row in sh.iter_rows(min_row=2):
            for cell in row:
                if cell.data_type == "f":
                    cell.font = Font(color="008000") if "'Cross_Section_Inputs'!" in str(cell.value) else Font(color="000000")
                elif isinstance(cell.value, (float, int)):
                    cell.font = Font(color="0000FF")
                    cell.number_format = '0.000000;[Red](0.000000);-'
        if sh.title in ("Model_Comparison", "Coefficient_Tests"):
            sh.column_dimensions["A"].width = 20
        if sh.title == "Read_Me":
            sh.column_dimensions["A"].width = 31
            sh.column_dimensions["B"].width = 75
            for row in sh.iter_rows():
                for cell in row:
                    cell.alignment = Alignment(wrap_text=True, vertical="center")

comparison.to_csv(OUT / "confronto_modelli.csv", index=False)
details.to_csv(OUT / "test_coefficienti.csv", index=False)
cross.to_csv(OUT / "input_cross_section.csv", index=False)

# ============================================================
# 7. CHARTS - ACTUAL VS FITTED AND R-SQUARED COMPARISON
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, (model_label, second_beta) in zip(axes, models.items()):
    name = second_beta.replace("Beta_", "")
    predicted = fitted[f"Fitted_{name}_percent"]
    actual = fitted["Mean_XS_percent"]
    low = min(actual.min(), predicted.min()) - 0.08
    high = max(actual.max(), predicted.max()) + 0.08
    ax.scatter(actual, predicted)
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1)
    for p, x, y in zip(cross["Portfolio"], actual, predicted):
        ax.annotate(p, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=8)
    ax.set(xlim=(low, high), ylim=(low, high), title=model_label,
           xlabel="Actual mean excess return (%)", ylabel="Fitted mean excess return (%)")
    ax.grid(alpha=0.25)
fig.suptitle("Cross-sectional regressions: actual vs fitted monthly means")
fig.tight_layout()
fig.savefig(OUT / "grafico_7_actual_vs_fitted.png", dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(comparison["Model"], comparison["R_squared"] * 100)
ax.set(title="Cross-sectional explanatory power", ylabel="R-squared (%)", ylim=(0, 100))
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(OUT / "grafico_8_R_squared_cross_section.png", dpi=160)
plt.close(fig)

# ============================================================
# 8. PRINT RESULTS AND FINAL CHECKS
# ============================================================
print("\nPART 3 - CROSS-SECTIONAL OLS (10 PORTFOLIOS)")
print(comparison.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
print("\nCOEFFICIENT TESTS")
print(details[["Model", "Parameter", "Coefficient_pp", "t_statistic", "p_value"]].to_string(
    index=False, float_format=lambda x: f"{x:.6f}"))
print("\nANALISI COMPLETATA. File salvati in:", OUT)
