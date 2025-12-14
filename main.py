import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

DATA_FILE = "mun0208.csv"        # tedd ide a letöltött KSH fájlt
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

# Alapértelmezett foglalkozás (példa):
# 1322 = "Informatikai és telekommunikációs tevékenységet folytató egység vezetője"
DEFAULT_FEOR = "1322"


def to_number(x):
    """
    KSH-ben gyakori: '1 825 697' -> 1825697
    Hiány: '–' vagy üres -> NaN
    """
    if pd.isna(x):
        return pd.NA
    s = str(x).strip()
    if s in {"", "–", "-", ""}:
        return pd.NA
    s = s.replace(" ", "").replace("\xa0", "")
    s = re.sub(r"[^\d]", "", s)
    return int(s) if s else pd.NA


def load_ksh_csv(path: str) -> pd.DataFrame:
    # A KSH STADAT CSV-k sokszor latin2 / cp1250 környéke
    # A fájl tetején van egy "cím sor", azt átugorjuk: skiprows=1
    return pd.read_csv(path, sep=";", skiprows=1, encoding="latin2")


def build_timeseries(df: pd.DataFrame, feor_code: str) -> pd.DataFrame:
    df["Foglalkozás FEOR'08 kódja"] = df["Foglalkozás FEOR'08 kódja"].astype(str).str.strip()

    row = df[df["Foglalkozás FEOR'08 kódja"] == feor_code]
    if row.empty:
        raise ValueError(f"Nincs ilyen FEOR kód a táblában: {feor_code}")

    row = row.iloc[0]

    year_cols = [c for c in df.columns if c.endswith("Együtt") and c[:4].isdigit()]
    year_cols = sorted(year_cols, key=lambda c: int(c.split()[0]))

    years = [int(c.split()[0]) for c in year_cols]
    values = [to_number(row[c]) for c in year_cols]

    out = pd.DataFrame({"Ev": years, "Brutto_atlagkereset_Ft_ho": values})
    out["Foglalkozas"] = row["Foglalkozás megnevezése"]
    out = out.dropna(subset=["Brutto_atlagkereset_Ft_ho"]).copy()
    out["Brutto_atlagkereset_Ft_ho"] = out["Brutto_atlagkereset_Ft_ho"].astype(int)

    # plusz stat oszlopok
    out["YoY_valtozas_Ft"] = out["Brutto_atlagkereset_Ft_ho"].diff()
    out["YoY_valtozas_szazalek"] = out["Brutto_atlagkereset_Ft_ho"].pct_change() * 100
    return out


def describe_stats(ts: pd.DataFrame) -> dict:
    s = ts["Brutto_atlagkereset_Ft_ho"]
    return {
        "min": int(s.min()),
        "max": int(s.max()),
        "atlag": float(s.mean()),
        "szoras": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "osszes_valtozas_szazalek": (float(s.iloc[-1]) / float(s.iloc[0]) - 1) * 100,
    }


def plot_line(ts: pd.DataFrame, title: str):
    plt.figure()
    plt.plot(ts["Ev"], ts["Brutto_atlagkereset_Ft_ho"], marker="o")
    plt.title(title)
    plt.xlabel("Év")
    plt.ylabel("Bruttó átlagkereset (Ft/fő/hó)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "vonaldiagram.png", dpi=160)


def plot_scatter_with_regression(ts: pd.DataFrame, title: str):
    X = ts["Ev"].values.reshape(-1, 1)
    y = ts["Brutto_atlagkereset_Ft_ho"].values

    model = LinearRegression().fit(X, y)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2 = float(model.score(X, y))

    y_hat = model.predict(X)

    plt.figure()
    plt.scatter(ts["Ev"], ts["Brutto_atlagkereset_Ft_ho"])
    plt.plot(ts["Ev"], y_hat)
    plt.title(title)
    plt.xlabel("Év")
    plt.ylabel("Bruttó átlagkereset (Ft/fő/hó)")
    plt.grid(True, alpha=0.3)

    eq = f"y = {slope:,.0f} * év + {intercept:,.0f}    (R²={r2:.3f})"
    plt.figtext(0.5, 0.01, eq, ha="center")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "pontdiagram_regresszio.png", dpi=160)

    return slope, intercept, r2


def main():
    df = load_ksh_csv(DATA_FILE)

    feor = input(f"Adj meg egy FEOR kódot (pl. {DEFAULT_FEOR}) [Enter=alap]: ").strip() or DEFAULT_FEOR
    ts = build_timeseries(df, feor)

    title_base = f"KSH STADAT mun0208 – {ts['Foglalkozas'].iloc[0]}"
    stats = describe_stats(ts)

    # mentés (strukturált adatok)
    ts.to_csv(OUT_DIR / "tisztitott_idosor.csv", index=False, encoding="utf-8")

    # statisztika kiírás
    print("\n--- Alap statisztikák ---")
    for k, v in stats.items():
        print(f"{k}: {v}")

    print("\n--- Idősor (év, Ft/hó, YoY) ---")
    print(ts[["Ev", "Brutto_atlagkereset_Ft_ho", "YoY_valtozas_Ft", "YoY_valtozas_szazalek"]].to_string(index=False))

    # plotok
    plot_line(ts, title_base + " (vonaldiagram)")
    slope, intercept, r2 = plot_scatter_with_regression(ts, title_base + " (pontdiagram + regresszió)")

    print(f"\nLineáris regresszió: meredekség ~ {slope:,.0f} Ft/év, R²={r2:.3f}")
    print(f"Kimenetek: {OUT_DIR}/vonaldiagram.png, {OUT_DIR}/pontdiagram_regresszio.png, {OUT_DIR}/tisztitott_idosor.csv")


if __name__ == "__main__":
    main()
