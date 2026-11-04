
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis


# =============================================================
# 1. PERCORSI DEL PROGETTO
# =============================================================

CARTELLA_PROGETTO = Path(__file__).resolve().parent

FILE_EXCEL = (
    CARTELLA_PROGETTO
    / "dati"
    / "First_Assignment_dataset.xlsx"
)

CARTELLA_OUTPUT = CARTELLA_PROGETTO / "output" / "winsorized" / "parte_1"

if not FILE_EXCEL.exists():
    raise SystemExit(
        "ERRORE: file Excel non trovato.\n"
        f"Metti 'First_Assignment_dataset.xlsx' in: "
        f"{CARTELLA_PROGETTO / 'dati'}\n"
        "Controlla il nome e l'estensione .xlsx."
    )

CARTELLA_OUTPUT.mkdir(parents=True, exist_ok=True)


# =============================================================
# 2. LETTURA DEL FILE EXCEL E CONTROLLI
# =============================================================

df = pd.read_excel(
    FILE_EXCEL,
    sheet_name=0,
    engine="openpyxl"
)

df.columns = df.columns.astype(str).str.strip()

portafogli = [f"S{i}" for i in range(1, 11)]
nomi_excess = [f"XS{i}" for i in range(1, 11)]

colonne_richieste = [
    "month", "smb", "hml", "umd", "rf", "rm", "rmrf"
] + portafogli

mancanti = [
    col for col in colonne_richieste
    if col not in df.columns
]

if mancanti:
    raise SystemExit(
        f"ERRORE: mancano queste colonne: {mancanti}"
    )

# Trasformazione delle date: 1980m10 = ottobre 1980

mesi_originali = df["month"].astype(str).str.strip()

parti_data = mesi_originali.str.extract(
    r"^(\d{4})m(\d{1,2})$"
)

if parti_data.isna().any().any():
    raise SystemExit(
        "ERRORE: formato delle date non valido."
    )

date = pd.to_datetime({
    "year": parti_data[0].astype(int),
    "month": parti_data[1].astype(int),
    "day": 1
})

if date.duplicated().any() or not date.is_monotonic_increasing:
    raise SystemExit(
        "ERRORE: mesi duplicati o fuori ordine."
    )

# Verifica che i mesi siano consecutivi

mesi_attesi = pd.date_range(
    start=date.iloc[0],
    periods=len(date),
    freq="MS"
)

if not date.reset_index(drop=True).equals(
    pd.Series(mesi_attesi)
):
    raise SystemExit(
        "ERRORE: mancano alcuni mesi nel dataset."
    )

# Conversione in numeri

numeriche = (
    df[colonne_richieste]
    .drop(columns="month")
    .apply(pd.to_numeric, errors="raise")
)

if not np.isfinite(
    numeriche.to_numpy(dtype=float)
).all():
    raise SystemExit(
        "ERRORE: valori numerici mancanti o non validi."
    )

# Controllo: RMRF dovrebbe essere RM - RF

if not np.allclose(
    numeriche["rm"] - numeriche["rf"],
    numeriche["rmrf"],
    rtol=0,
    atol=2e-7
):
    print("ATTENZIONE: verificare la colonna rmrf.")

print(f"\nOsservazioni mensili: {len(df)}")
print(
    f"Periodo: {mesi_originali.iloc[0]}"
    f" - {mesi_originali.iloc[-1]}"
)


# =============================================================
# 3. CALCOLO DEGLI EXCESS RETURNS
# FORMULA: XSi = Si - RF
# =============================================================

excess = pd.DataFrame({
    f"XS{i}": numeriche[f"S{i}"] - numeriche["rf"]
    for i in range(1, 11)
})

# ============================================================
# 1% / 99% WINSORIZATION OF EXCESS RETURNS ONLY
# Keep the original series unchanged for comparison.
# RMRF, SMB, HML and UMD are NOT winsorized.
# ============================================================
excess_original = excess.copy()
excess = excess_original.clip(
    lower=excess_original.quantile(0.01),
    upper=excess_original.quantile(0.99),
    axis=1,
)


# Salviamo i dati con la colonna delle date

excess_da_salvare = pd.concat([
    df[["month"]].reset_index(drop=True),
    excess.reset_index(drop=True)
], axis=1)

excess_da_salvare.to_csv(
    CARTELLA_OUTPUT / "excess_returns.csv",
    index=False,
    float_format="%.8f"
)

print("\nESEMPIO DI EXCESS RETURN (PRIMO MESE)")

print(f"S1  = {numeriche['S1'].iloc[0]:.7f}")
print(f"RF  = {numeriche['rf'].iloc[0]:.7f}")
print(f"XS1 = {excess['XS1'].iloc[0]:.7f}")


# =============================================================
# 4. CALCOLO DELLE SUMMARY STATISTICS
# =============================================================

righe_statistiche = []

for nome in nomi_excess:

    x = excess[nome]

    idx_min = int(x.idxmin())
    idx_max = int(x.idxmax())

    righe_statistiche.append({

        "Portfolio": nome,

        "Observations": len(x),

        "Mean (%)": x.mean() * 100,

        "Median (%)": x.median() * 100,

        "Std. Dev. (%)": x.std(ddof=1) * 100,

        "Min (%)": x.min() * 100,

        "Max (%)": x.max() * 100,

        "Skewness": skew(x, bias=False),

        "Excess Kurtosis": kurtosis(
            x,
            fisher=True,
            bias=False
        ),

        "Negative Months (%)": (x < 0).mean() * 100,

        "Worst Month": mesi_originali.iloc[idx_min],

        "Best Month": mesi_originali.iloc[idx_max]

    })

statistiche = pd.DataFrame(righe_statistiche)

statistiche.to_csv(
    CARTELLA_OUTPUT / "statistiche_excess_returns.csv",
    index=False,
    float_format="%.6f"
)


# =============================================================
# 5. SALVATAGGIO RISULTATI IN EXCEL
# =============================================================

with pd.ExcelWriter(
    CARTELLA_OUTPUT / "risultati_first_assignment.xlsx",
    engine="openpyxl"
) as writer:

    excess_da_salvare.to_excel(
        writer,
        sheet_name="Excess_Returns",
        index=False
    )

    statistiche.to_excel(
        writer,
        sheet_name="Summary_Statistics",
        index=False
    )


# =============================================================
# 6. STAMPA DELLE STATISTICHE NEL TERMINALE
# =============================================================

print("\nSUMMARY STATISTICS")
print("Rendimenti mensili espressi in percentuale")

colonne_mostrate = [
    "Portfolio",
    "Mean (%)",
    "Median (%)",
    "Std. Dev. (%)",
    "Min (%)",
    "Max (%)",
    "Skewness",
    "Excess Kurtosis"
]

print(
    statistiche[colonne_mostrate].to_string(
        index=False,
        float_format=lambda valore: f"{valore:8.2f}"
    )
)

print("\nMESI CON RENDIMENTI ESTREMI")

for _, riga in statistiche.iterrows():

    print(
        f"{riga['Portfolio']}: "
        f"minimo {riga['Min (%)']:+.2f}% "
        f"({riga['Worst Month']}) | "
        f"massimo {riga['Max (%)']:+.2f}% "
        f"({riga['Best Month']})"
    )


# =============================================================
# 7. GRAFICO 1 - SERIE TEMPORALI
# =============================================================

fig, assi = plt.subplots(
    5,
    2,
    figsize=(16, 17),
    sharex=True
)

for i, ax in enumerate(assi.flat):

    nome = nomi_excess[i]

    ax.plot(
        date,
        excess[nome] * 100,
        linewidth=0.9
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=0.8
    )

    ax.set_title(nome)
    ax.set_ylabel("Excess return (%)")
    ax.grid(alpha=0.25)

    ax.xaxis.set_major_locator(
        mdates.YearLocator(5)
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y")
    )

fig.suptitle(
    "Monthly excess returns - UK portfolios (1980-2010)",
    fontsize=16
)

fig.tight_layout(rect=[0, 0, 1, 0.975])

fig.savefig(
    CARTELLA_OUTPUT / "grafico_1_serie_temporali.png",
    dpi=160
)

plt.close(fig)


# =============================================================
# 8. GRAFICO 2 - RENDIMENTI MEDI
# =============================================================

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    statistiche["Portfolio"],
    statistiche["Mean (%)"]
)

ax.set_title("Average monthly excess return")

ax.set_xlabel(
    "Portfolio (XS1 = smallest; XS10 = largest)"
)

ax.set_ylabel("Mean excess return (%)")

ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

fig.savefig(
    CARTELLA_OUTPUT / "grafico_2_rendimenti_medi.png",
    dpi=160
)

plt.close(fig)


# =============================================================
# 9. GRAFICO 3 - VOLATILITA'
# =============================================================

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(
    statistiche["Portfolio"],
    statistiche["Std. Dev. (%)"]
)

ax.set_title("Monthly volatility across portfolios")

ax.set_xlabel(
    "Portfolio (XS1 = smallest; XS10 = largest)"
)

ax.set_ylabel("Standard deviation (%)")

ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

fig.savefig(
    CARTELLA_OUTPUT / "grafico_3_volatilita.png",
    dpi=160
)

plt.close(fig)


# =============================================================
# 10. GRAFICO 4 - BOXPLOT
# =============================================================

fig, ax = plt.subplots(figsize=(12, 6))

ax.boxplot(
    [
        excess[nome] * 100
        for nome in nomi_excess
    ],
    tick_labels=nomi_excess
)

ax.axhline(
    0,
    linestyle="--",
    linewidth=0.8
)

ax.set_title(
    "Distribution of monthly excess returns"
)

ax.set_xlabel("Portfolio")
ax.set_ylabel("Excess return (%)")
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()

fig.savefig(
    CARTELLA_OUTPUT / "grafico_4_boxplot.png",
    dpi=160
)

plt.close(fig)


# =============================================================
# 11. MESSAGGIO FINALE
# =============================================================

print("\nANALISI COMPLETATA!")

print(
    f"Tutti i file si trovano in: {CARTELLA_OUTPUT}"
)

print("\nFile generati:")

print("- excess_returns.csv")
print("- statistiche_excess_returns.csv")
print("- risultati_first_assignment.xlsx")
print("- grafico_1_serie_temporali.png")
print("- grafico_2_rendimenti_medi.png")
print("- grafico_3_volatilita.png")
print("- grafico_4_boxplot.png")
