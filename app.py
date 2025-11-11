import pandas as pd
import streamlit as st
import re, json
from pathlib import Path

# ====== Gemini (opsional) ======
try:
    import google.generativeai as genai
    GEMINI_OK = True
except Exception:
    GEMINI_OK = False

st.set_page_config(page_title="Chatbot Kepegawaian — PSEKP", layout="centered")
st.title("Chatbot Kepegawaian — PSEKP")
st.caption("Sumber data: file di repo → data/kepegawaian.xlsx (sheet: DATA)")

# -----------------------------
# Konfigurasi kolom & helper
# -----------------------------
COL = {
    "nip":"NIP",
    "nama":"Nama",
    "jf":"Jabatan Fungsional Tertentu",
    "js":"Jabatan Struktural",
    "gol":"Golongan Pegawai Saat Ini",
    "pang":"Pangkat Pegawai Saat Ini",
    "tmtj":"TMT Jabatan",
    "tmtg":"TMT Golongan Saat Ini",
    "email":"Email",
    "hp":"No HP"
}
REQ = list(COL.values())
MAX_LIST = 20

def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].astype(str).str.replace("\u00A0", " ", regex=False).str.strip()
    return df

def check_columns(df: pd.DataFrame):
    miss = [c for c in REQ if c not in df.columns]
    if miss:
        st.error(f"Kolom berikut belum ada di Excel: {miss}")
        st.stop()

def fmt(r: dict) -> str:
    jab = r.get(COL["jf"]) if r.get(COL["jf"]) and r.get(COL["jf"]) != "-" else r.get(COL["js"], "-")
    return (
        f"**{r.get(COL['nama'],'-')}**\n"
        f"NIP: `{r.get(COL['nip'],'-')}`\n"
        f"Jabatan: {jab}\n"
        f"Gol/Pangkat: {r.get(COL['gol'],'-')} / {r.get(COL['pang'],'-')}\n"
        f"TMT Jabatan: {r.get(COL['tmtj'],'-')}\n"
        f"TMT Gol: {r.get(COL['tmtg'],'-')}\n"
        f"Kontak: {r.get(COL['email'],'-')} | {r.get(COL['hp'],'-')}"
    )

def mask(nip: str) -> str:
    nip = str(nip or "")
    return nip if len(nip) < 8 else nip[:4] + "****" + nip[-4:]

# -----------------------------
# Baca data dari repo saja
# -----------------------------
DATA_PATH = Path("data/kepegawaian.xlsx")
if not DATA_PATH.exists():
    st.error("File **data/kepegawaian.xlsx** tidak ditemukan di repo. Pastikan nama & lokasinya benar.")
    st.stop()

try:
    df = pd.read_excel(DATA_PATH, sheet_name="DATA", dtype=str)
except ValueError:
    st.error("Sheet **'DATA'** tidak ditemukan. Ubah nama sheet di Excel menjadi 'DATA'.")
    st.stop()
except Exception as e:
    st.error(f"Gagal membaca Excel: {e}")
    st.stop()

df = clean_df(df)
check_columns(df)

# -----------------------------
# Toggle: Gemini router (opsional)
# -----------------------------
use_gemini = st.toggle(
    "Gunakan Gemini untuk interpretasi query (opsional)",
    value=False,
    help="Gemini hanya merutekan maksud (nip_exact/nip_prefix/name_contains). Pencarian tetap ke Excel."
)
if use_gemini and not GEMINI_OK:
    st.warning("Paket google-generativeai belum terpasang. Tambahkan di requirements.txt, lalu deploy ulang.")
    use_gemini = False

# -----------------------------
# Pencarian (NIP prefix / Nama)
# -----------------------------
st.success("Ketik angka (NIP penuh/awalan, mis. 1994) atau nama (tidak case sensitive).")
q = st.text_input("Cari NIP (boleh sebagian) atau Nama", placeholder="contoh: 1994 atau restu")

def search_engine(df: pd.DataFrame, mode: str, value: str) -> pd.DataFrame:
    if mode == "nip_exact":
        nip_series = df[COL["nip"]].astype(str).str.replace(r"\s", "", regex=True)
        return df[nip_series.eq(value)]
    if mode == "nip_prefix":
        nip_series = df[COL["nip"]].astype(str).str.replace(r"\s", "", regex=True)
        return df[nip_series.str.startswith(value, na=False)]
    if mode == "name_contains":
        return df[df[COL["nama"]].fillna("").str.contains(value, case=False, na=False)]
    return df.head(0)

# Router via Gemini (opsional)
def gemini_route(query: str):
    """
    Output JSON: {"mode":"nip_exact|nip_prefix|name_contains","value":"..."}
    """
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    except KeyError:
        st.error("GEMINI_API_KEY belum diisi di Streamlit Secrets.")
        return None

    system = (
        "Kamu router intent. Terima teks user, tentukan mode pencarian:\n"
        "- Jika hanya angka dan panjang >= 18 → mode=nip_exact\n"
        "- Jika hanya angka dan panjang < 18 → mode=nip_prefix\n"
        "- Selain itu → mode=name_contains\n"
        "Balas **hanya** JSON valid satu baris. Contoh: {\"mode\":\"nip_prefix\",\"value\":\"1994\"}"
    )
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content(system + "\n\nUser: " + query)
        text = (resp.text or "").strip()
        text = text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except Exception as e:
        st.warning(f"Gagal interpretasi Gemini, pakai aturan biasa. Detail: {e}")
        return None

if q:
    q = q.strip()
    route = None
    if use_gemini:
        route = gemini_route(q)

    if route and isinstance(route, dict) and route.get("mode") and route.get("value") is not None:
        mode_key = route["mode"]
        val = str(route["value"]).strip()
    else:
        # fallback deterministik (tanpa LLM)
        if q.isdigit():
            mode_key, val = ("nip_exact", q) if len(q) >= 18 else ("nip_prefix", q)
        else:
            mode_key, val = "name_contains", q

    hits = search_engine(df, mode_key, val)

    if hits.empty:
        st.warning(f"Tidak ada data cocok (mode: {mode_key}, nilai: {val}).")
    elif len(hits) == 1:
        st.markdown("**Hasil:**")
        st.markdown(fmt(hits.iloc[0].to_dict()))
    else:
        st.markdown(f"Ditemukan {len(hits)} data. Menampilkan maksimal {MAX_LIST}:")
        for _, r in hits.head(MAX_LIST).iterrows():
            jab = (r.get(COL["jf"]) or r.get(COL["js"]) or "-")
            st.markdown("---")
            st.markdown(f"**{r[COL['nama']]}** — NIP: `{mask(r[COL['nip']])}`\nJabatan: {jab}")

st.divider()

# -----------------------------
# Panel Tanya AI (jawab pakai data Excel)
# -----------------------------
st.subheader("Tanya AI (berdasar baris relevan dari Excel)")
if not GEMINI_OK:
    st.info("Tambahkan di requirements.txt: google-generativeai>=0.7.0 untuk mengaktifkan Gemini.")
else:
    ask = st.text_area("Pertanyaan (contoh: 'jabatan restu apa?' atau 'NIP 1976... jabatannya?')", height=100)
    n = st.slider("Konteks baris (maks)", 1, 10, 5)

    def pick_rows(df: pd.DataFrame, question: str, limit: int) -> pd.DataFrame:
        ql = question.lower()
        digits = re.findall(r"\d{8,20}", ql)
        if digits:
            nip_series = df[COL["nip"]].astype(str).str.replace(r"\s", "", regex=True)
            pick = df[nip_series.isin(digits)]
            if not pick.empty:
                return pick.head(limit)
        tokens = [t for t in re.split(r"[^a-zA-Z]+", ql) if len(t) >= 3]
        if not tokens:
            return df.head(0)
        base = df[COL["nama"]].fillna("").str.lower()
        maskname = pd.Series(False, index=df.index)
        for t in tokens:
            maskname = maskname | base.str.contains(t)
        return df[maskname].head(limit)

    if st.button("Tanya Gemini"):
        if not ask.strip():
            st.warning("Tulis pertanyaannya dulu ya.")
        else:
            try:
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
            except KeyError:
                st.error("GEMINI_API_KEY belum diisi di Streamlit Secrets.")
                st.stop()

            ctx = pick_rows(df, ask, n)[
                [COL["nip"], COL["nama"], COL["jf"], COL["js"], COL["gol"], COL["pang"], COL["tmtj"], COL["tmtg"], COL["email"], COL["hp"]]
            ].fillna("-")
            csv_ctx = ctx.to_csv(index=False)

            system = (
                "Kamu asisten kepegawaian PSEKP. Jawab hanya berdasarkan DATA berikut (CSV). "
                "Jika info tidak ada, katakan 'data tidak tersedia'. Jawab ringkas & rapi."
            )
            prompt = f"{system}\n\nDATA (CSV):\n{csv_ctx}\n\nPERTANYAAN:\n{ask}"

            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                with st.spinner("Gemini berpikir..."):
                    resp = model.generate_content(prompt)
                st.success("Jawaban:")
                st.write((resp.text or "").strip())
                with st.expander("Lihat konteks (CSV)"):
                    st.code(csv_ctx, language="csv")
            except Exception as e:
                st.error(f"Gagal memanggil Gemini: {e}")
