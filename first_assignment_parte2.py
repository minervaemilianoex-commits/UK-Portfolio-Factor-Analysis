
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. CARTELLE DEL PROGETTO
# ============================================================

RADICE = Path(__file__).resolve().parent

EXCEL_ORIGINALE = (
    RADICE / "dati" / "First_Assignment_dataset.xlsx"
)

OUTPUT = RADICE / "output" / "parte_2_regressioni"

if not EXCEL_ORIGINALE.exists():
    raise SystemExit(
        f"File Excel non trovato: {EXCEL_ORIGINALE}"
    )

OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. IMPORTAZIONE E CONTROLLO DEI DATI
# ============================================================

raw = pd.read_excel(
    EXCEL_ORIGINALE,
    sheet_name=0,
    engine="openpyxl"
)

raw.columns = raw.columns.astype(str).str.strip()

fattori = ["rmrf", "smb", "hml", "umd"]

portafogli = [
    f"S{i}" for i in range(1, 11)
]

richieste = ["month", "rf"] + fattori + portafogli

mancanti = [
    c for c in richieste if c not in raw.columns
]

if mancanti:
    raise SystemExit(
        f"Colonne mancanti: {mancanti}"
    )

numeri = raw[richieste[1:]].apply(
    pd.to_numeric,
    errors="raise"
)

if not np.isfinite(
    numeri.to_numpy(dtype=float)
).all():
    raise SystemExit(
        "Sono presenti valori mancanti o non validi."
    )

n = len(raw)

print("Osservazioni mensili:", n)


# ============================================================
# 3. CALCOLO DEI DIECI EXCESS RETURNS
# ============================================================

xs = pd.DataFrame({
    f"XS{i}": numeri[f"S{i}"] - numeri["rf"]
    for i in range(1, 11)
})


# ============================================================
# 4. CREAZIONE DELLA TABELLA PER LE REGRESSIONI
# ============================================================

reg_data = pd.DataFrame({

    "month": raw["month"],

    "RMRF": numeri["rmrf"],

    "SMB": numeri["smb"],

    "HML": numeri["hml"],

    "UMD": numeri["umd"]

})

reg_data = pd.concat(
    [reg_data, xs],
    axis=1
)


# ============================================================
# 5. COSTRUZIONE DELLA MATRICE X
# ============================================================

X = sm.add_constant(
    numeri[fattori],
    has_constant="add"
)

X = X.rename(columns={

    "const": "Alpha",

    "rmrf": "MKT",

    "smb": "SMB",

    "hml": "HML",

    "umd": "UMD"

})

ordine = [
    "Alpha", "MKT", "SMB", "HML", "UMD"
]


# ============================================================
# 6. DIECI REGRESSIONI OLS
# ============================================================

righe_coefficienti = []

righe_test = []

righe_dettaglio = []

righe_crosssection = []

riepiloghi = []


for i in range(1, 11):

    nome = f"XS{i}"

    # Variabile dipendente
    y = xs[nome]

    # Stima della regressione
    risultato = sm.OLS(y, X).fit()

    # Estrazione delle statistiche
    b = risultato.params
    t = risultato.tvalues
    p = risultato.pvalues
    se = risultato.bse

    # Coefficienti e R-quadrato
    righe_coefficienti.append({

        "Portfolio": nome,

        "Mean_XS_%_monthly": y.mean() * 100,

        "Alpha_%_monthly": b["Alpha"] * 100,

        "Beta_MKT": b["MKT"],

        "Beta_SMB": b["SMB"],

        "Beta_HML": b["HML"],

        "Beta_UMD": b["UMD"],

        "R_squared": risultato.rsquared,

        "R_squared_%": risultato.rsquared * 100,

        "Adjusted_R_squared": risultato.rsquared_adj,

        "Observations": int(risultato.nobs)

    })

    # t-statistics e p-values
    riga_test = {"Portfolio": nome}

    for factor in ordine:

        riga_test[f"t_{factor}"] = t[factor]

        riga_test[f"p_{factor}"] = p[factor]

        righe_dettaglio.append({

            "Portfolio": nome,

            "Factor": factor,

            "Coefficient_decimal": b[factor],

            "Standard_Error": se[factor],

            "t_stat": t[factor],

            "p_value": p[factor],

            "Significant_at_5%": (
                "YES" if p[factor] < 0.05 else "NO"
            )

        })

    righe_test.append(riga_test)

    # Dati che serviranno nella Parte 3
    righe_crosssection.append({

        "Portfolio": nome,

        "Mean_XS_decimal": y.mean(),

        "Beta_MKT": b["MKT"],

        "Beta_SMB": b["SMB"],

        "Beta_HML": b["HML"],

        "Beta_UMD": b["UMD"]

    })

    riepiloghi.append(
        "=" * 75
        + f"\n{nome}\n"
        + risultato.summary().as_text()
    )


# ============================================================
# 7. ORGANIZZAZIONE DEI RISULTATI
# ============================================================

coeff = pd.DataFrame(righe_coefficienti)

tests = pd.DataFrame(righe_test)

details = pd.DataFrame(righe_dettaglio)

cross = pd.DataFrame(righe_crosssection)


# ============================================================
# 8. ESPORTAZIONE DEI RISULTATI
# ============================================================

coeff.to_csv(
    OUTPUT / "coefficienti_regressioni.csv",
    index=False
)

tests.to_csv(
    OUTPUT / "t_stat_e_p_value.csv",
    index=False
)

(OUTPUT / "regressioni_ols_complete.txt").write_text(
    "\n\n".join(riepiloghi),
    encoding="utf-8"
)

info = pd.DataFrame([

    ("Sample", "October 1980 - December 2010"),

    ("Observations per portfolio", str(n)),

    ("Model", "Four-factor OLS with intercept"),

    ("Returns", "Decimal: 0.01 means 1%"),

    ("Alpha", "Monthly alpha in percent units"),

    ("p-value", "Two-sided test: coefficient = 0"),

    ("R-squared", "Proportion of variation explained"),

    ("Part 3", "Cross-section inputs only")

], columns=["Item", "Explanation"])


with pd.ExcelWriter(
    OUTPUT / "parte2_risultati_regressioni.xlsx",
    engine="openpyxl"
) as writer:

    reg_data.to_excel(
        writer,
        sheet_name="Regression_Data",
        index=False
    )

    coeff.to_excel(
        writer,
        sheet_name="Coefficients",
        index=False
    )

    tests.to_excel(
        writer,
        sheet_name="T_and_P_values",
        index=False
    )

    details.to_excel(
        writer,
        sheet_name="Factor_Details",
        index=False
    )

    cross.to_excel(
        writer,
        sheet_name="Cross_Section_Ready",
        index=False
    )

    info.to_excel(
        writer,
        sheet_name="Read_Me",
        index=False
    )


# ============================================================
# 9. GRAFICO DELLE ESPOSIZIONI AI FATTORI
# ============================================================

fig, assi = plt.subplots(
    2, 2,
    figsize=(12, 8),
    sharex=True
)

for ax, factor in zip(
    assi.flat,
    ["MKT", "SMB", "HML", "UMD"]
):

    ax.plot(
        coeff["Portfolio"],
        coeff[f"Beta_{factor}"],
        marker="o"
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=0.8
    )

    ax.set_title(f"Exposure to {factor}")

    ax.set_ylabel("Estimated beta")

    ax.grid(alpha=0.25)

for ax in assi[-1, :]:
    ax.set_xlabel("UK size-sorted portfolio")

fig.suptitle(
    "Four-factor risk exposures across UK portfolios"
)

fig.tight_layout(rect=[0, 0, 1, 0.95])

fig.savefig(
    OUTPUT / "grafico_5_esposizioni_ai_fattori.png",
    dpi=160
)

plt.close(fig)


# ============================================================
# 10. GRAFICO DEL COEFFICIENTE R-SQUARED
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    coeff["Portfolio"],
    coeff["R_squared_%"]
)

ax.set_ylim(0, 100)

ax.set_title(
    "Explanatory power of the four-factor model"
)

ax.set_ylabel("R-squared (%)")

ax.set_xlabel("UK size-sorted portfolio")

ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

fig.savefig(
    OUTPUT / "grafico_6_r_squared.png",
    dpi=160
)

plt.close(fig)


# ============================================================
# 11. VISUALIZZAZIONE DEI RISULTATI
# ============================================================

print("\n=== COEFFICIENTI DELLE REGRESSIONI ===")

colonne = [
    "Portfolio",
    "Alpha_%_monthly",
    "Beta_MKT",
    "Beta_SMB",
    "Beta_HML",
    "Beta_UMD",
    "R_squared_%"
]

print(
    coeff[colonne].to_string(
        index=False,
        float_format=lambda v: f"{v:.4f}"
    )
)

print("\n=== T-STATISTICS E P-VALUES ===")

print(
    tests.to_string(
        index=False,
        float_format=lambda v: f"{v:.5g}"
    )
)

print("\nANALISI COMPLETATA!")

print("Risultati salvati in:", OUTPUT)
