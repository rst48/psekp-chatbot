import pandas as pd
import streamlit as st
import re
from pathlib import Path

# ==== LLM (OpenAI) untuk interpretasi query (opsional) ====
try:
    from openai import OpenAI
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False

st.set_page_config(page_title="Chatbot Kepegawaian (PSEKP)", layout="centered")
st.title("Chatbot Kepegawaian — PoC (PSEKP)")
st.caption("Baca Excel dari upload atau dari repo (data/kepegawaian.xlsx). Sheet harus bernama: DATA")

# ======= Konfigurasi kolom =======
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

def format_row(r: dict) -> str:
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

# ======= Pilih sumber data =======
mode = st.radio("Sumber data", ["Upload Excel", "Pakai file di repo (data/kepegawaian.xlsx)"])
if mode == "Upload Excel":
    uploaded = st.file_uploader("Unggah Excel (sheet DATA)", type=["xlsx"])
    if not uploaded:
        st.info("Unggah file Excel kamu dulu (kolom sesuai template).")
        st.stop()
    df = pd.read_excel(uploaded, sheet_name="DATA", dtype=str)
else:
    path = Path("data/kepegawaian.xlsx")
    if not path.exists():
        st.error("File data/kepegawaian.xlsx tidak ditemukan di repo.")
        st.stop()
    try:
        df = pd.read_excel(path, sheet_name="DATA", dtype=str)
    except ValueError:
        st.error("Sheet 'DATA' tidak ditemukan. Ubah nama sheet di Excel menjadi 'DATA'.")
        st.stop()

df = clean_df(df)
check_columns(df)

# ======= Toggle LLM routing (interpret query) =======
use_llm = st.toggle("Gunakan LLM untuk interpretasi query (opsional)", value=False,
                    help="LLM hanya dipakai untuk memahami maksud (nip_prefix atau name_contains). Pencarian tetap deterministik di Excel.")
if use_llm and not LLM_AVAILABLE:
    st.warning("Paket openai belum terpasang. Tambahkan 'openai>=1.0.0' di requirements.txt, lalu deploy ulang.")
    use_llm = False

# ======= Input Query =======
st.success("Ketik: angka (NIP penuh atau awalan NIP seperti 1994) atau nama (tidak case sensitive).")
q = st.text_input("Cari NIP (boleh sebagian) atau Nama", placeholder="contoh: 1994 atau restu")

# ======= LLM router (opsional) =======
def llm_route(query: str):
    """
    Kembalikan dict {mode: 'nip_exact'|'nip_prefix'|'name_contains', value: '...'}
    LLM hanya bantu interpretasi, bukan mencari.
    """
    try:
        client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    except Exception:
        st.error("OPENAI_API_KEY belum diisi di Secrets. Matikan toggle LLM atau isi kuncinya.")
        return None

    system_msg = (
        "Kamu adalah router intent. Terima pertanyaan manusia, putuskan apakah itu pencarian NIP atau Nama.\n"
        "- Jika hanya angka dan panjang >= 18 → mode=nip_exact, value=angka itu.\n"
        "- Jika hanya angka dan panjang < 18 → mode=nip_prefix, value=angka itu.\n"
        "- Selain itu → mode=name_contains, value=frasa nama (lowercase, tanpa kata tidak penting).\n"
        "Jawab KETAT dalam JSON satu baris, tanpa komentar. Contoh: {\"mode\":\"nip_prefix\",\"value\":\"1994\"}"
    )
    user_msg = f"Pertanyaan: {query}"
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role":"system","content":system_msg},
                {"role":"user","content":user_msg}
            ],
            temperature=0
        )
        text = resp.choices[0].message.content.strip()
        import json
        return json.loads(text)
    except Exception as e:
        st.warning(f"Gagal interpretasi LLM, fallback ke aturan biasa. Detail: {e}")
        return None

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

if q:
    q = q.strip()

    # Tentukan mode/value
    route = None
    if use_llm:
        route = llm_route(q)

    if route and isinstance(route, dict) and route.get("mode") and route.get("value") is not None:
        mode_key = route["mode"]
        val = str(route["value"]).strip()
    else:
        # Fallback deterministic (tanpa LLM)
        if q.isdigit():
            if len(q) >= 18:
                mode_key, val = "nip_exact", q
            else:
                mode_key, val = "nip_prefix", q
        else:
            mode_key, val = "name_contains", q

    hits = search_engine(df, mode_key, val)

    # Output
    if hits.empty:
        st.warning(f"Tidak ada data cocok (mode: {mode_key}, nilai: {val}).")
    elif len(hits) == 1:
        st.markdown("**Hasil:**")
        st.markdown(format_row(hits.iloc[0].to_dict()))
    else:
        st.markdown(f"Ditemukan {len(hits)} data. Menampilkan maksimal {MAX_LIST}:")
        for _, r in hits.head(MAX_LIST).iterrows():
            st.markdown("---")
            jab = (r.get(COL["jf"]) or r.get(COL["js"]) or "-")
            st.markdown(f"**{r[COL['nama']]}** — NIP: `{mask(r[COL['nip']])}`\nJabatan: {jab}")
