diff --git a/main.py b/main.py
index 1111111..2222222 100644
--- a/main.py
+++ b/main.py
@@ -1,165 +1,149 @@
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
-    # KSH STADAT CSV-knél tipikus az eltérő kódolás -> próbálunk párat
-    for enc in ("utf-8-sig", "latin2", "cp1250"):
-        try:
-            return pd.read_csv(path, sep=";", skiprows=1, encoding=enc)
-        except UnicodeDecodeError:
-            continue
-    # ha egyik sem jó, dobjuk a "valódi" hibát
-    return pd.read_csv(path, sep=";", skiprows=1, encoding="latin2")
+    # (WIP) egyszerűsítés: próbáljuk csak UTF-8-ként
+    return pd.read_csv(path, sep=";", skiprows=1, encoding="utf-8-sig")
 
 
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
-    year_cols = [c for c in df.columns if isinstance(c, str) and c[:4].isdigit() and c.endswith("Együtt")]
+    # (WIP) elgépelés: "Egyutt" -> ez direkt hiba lesz
+    year_cols = [c for c in df.columns if isinstance(c, str) and c[:4].isdigit() and c.endswith("Egyutt")]
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
