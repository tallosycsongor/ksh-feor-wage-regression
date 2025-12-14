import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

DATA_FILE = "mun0208.csv"  # lehet .csv vagy .xlsx
OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

DEFAULT_FEOR = "1322"

_MISSING_TOKENS = {"", "–", "-", ""}
_NON_DIGIT_RE = re.compile(r"[^\d]+")


def _read_csv_with_fallback(path: str) -> pd.DataFrame:
    # KSH STADAT CSV-knél tipikus az eltérő kódolás -> próbálunk párat
    for enc in ("utf-8-sig", "latin2", "cp1250"):
        try:
            return pd.read_csv(path, sep=";", skiprows=1, encoding=enc)
        except UnicodeDecodeError:
            continue
    # ha egyik sem jó, dobjuk a "valódi" hibát
    return pd.read_csv(path, sep=";", skiprows=1, encoding="latin2")


def load_ksh_file(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Nem találom a fájlt: {path}")

    suf = p.suffix.lower()
    if suf == ".csv":
        return _read_csv_with_fallback(path)
    if suf in (".xlsx", ".xls"):
        # STADAT xlsx-ek általában 1 fejlécsorral indulnak -> skiprows=1
        return pd.read_excel(path, skiprows=1)

    raise ValueError("Csak .csv vagy .xlsx fájlt támogat a program.")


def build_timeseries(df: pd.DataFrame, feor_code: str) -> pd.DataFrame:
    df = df.copy()
    df["Foglalkozás FEOR'08 kódja"] = df["Foglalkozás FEOR'08 kódja"].astype(str).str.strip()

    row_df = df.loc[df["Foglalkozás FEOR'08 kódja"] == str(feor_code).strip()]
    if row_df.empty:
        raise ValueError(f"Nincs ilyen FEOR kód a táblában: {feor_code}")

    row = row_df.iloc[0]

    # 2019 Együtt, 2020 Együtt, ...
    year_cols = [c for c in df.columns if isinstance(c, str) and c[:4].isdigit() and c.endswith("Együtt")]
    year_cols.sort(key=lambda c: int(c[:4]))

    years = [int(c[:4]) for c in year_cols]

    # --- OPTI: tisztítás vektorosan (gyorsabb mint elemről elemre) ---
    raw = row[year_cols]

    # egységes string + hiányzó jelölések kezelése
    s = raw.astype(str).str.strip()
    s = s.replace(list(_MISSING_TOKENS), pd.NA)

    # szóközök + minden nem-szám eltávolítása (pl. "1 825 697" -> "1825697")
    s = s.str.replace("\xa0", "", regex=False).str.replace(" ", "", regex=False)
    s = s.str.replace(_NON_DIGIT_RE, "", regex=True)

    values = pd.to_numeric(s, errors="coerce").astype("Int64")

    out = pd.DataFrame(
        {
            "Ev": years,
            "Brutto_atlagkereset_Ft_ho": values,
        }
    )
    out["Foglalkozas"] = row["Foglalkozás megnevezése"]
    out = out.dropna(subset=["Brutto_atlagkereset_Ft_ho"]).copy()
    out["Brutto_atlagkereset_Ft_ho"] = out["Brutto_atlagkereset_Ft_ho"].astype(int)

    out["YoY_valtozas_Ft"] = out["Brutto_atlagkereset_Ft_ho"].diff()
    out["YoY_valtozas_szazalek"] = out["Brutto_atlagkereset_Ft_ho"].pct_change() * 100
    return out


def describe_stats(ts: pd.DataFrame) -> dict:
    s = ts["Brutto_atlagkereset_Ft_ho"]
    first = float(s.iloc[0])
    last = float(s.iloc[-1])
    return {
        "min": int(s.min()),
        "max": int(s.max()),
        "atlag": float(s.mean()),
        "szoras": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "osszes_valtozas_szazalek": (last / first - 1) * 100,
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
    plt.close()


def plot_scatter_with_regression(ts: pd.DataFrame, title: str):
    X = ts["Ev"].to_numpy().reshape(-1, 1)
    y = ts["Brutto_atlagkereset_Ft_ho"].to_numpy()

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
    plt.close()

    return slope, intercept, r2


def main():
    df = load_ksh_file(DATA_FILE)

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
