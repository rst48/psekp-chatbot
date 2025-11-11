import pandas as pd
import streamlit as st
import re, requests
from pathlib import Path

st.set_page_config(page_title="PSEKP AI Chat (OpenRouter)", layout="centered")
st.title("Selamat datang di **PSEKP AI Chat** 🤖")
st.caption("Powered by OpenRouter — Apa yang mau kamu tanya hari ini tentang PSEKP?")

# ================== BACA DATA DARI REPO ==================
DATA_PATH = Path("data/kepegawaian.xlsx")
if not DATA_PATH.exists():
    st.error("File **data/kepegawaian.xlsx** tidak ditemukan di repo. Pastikan nama & lokasinya benar.")
    st.stop()

try:
    df = pd.read_excel(DATA_PATH, sheet_name="DATA", dtype=str)
except ValueError:
    st.error("Sheet **'DATA'** tidak ditemukan. Ubah nama sheet Excel menjadi 'DATA'.")
    st.stop()
except Exception as e:
    st.error(f"Gagal membaca Excel: {e}")
    st.stop()

# Bersihkan spasi tak terlihat
for c in df.columns:
    if df[c].dtype == "object":
        df[c] = df[c].astype(str).str.replace("\u00A0", " ", regex=False).str.strip()

# Kolom penting
COL = {
    "nip":"NIP", "nama":"Nama",
    "jf":"Jabatan Fungsional Tertentu", "js":"Jabatan Struktural",
    "gol":"Golongan Pegawai Saat Ini", "pang":"Pangkat Pegawai Saat Ini",
    "tmtj":"TMT Jabatan", "tmtg":"TMT Golongan Saat Ini",
    "email":"Email", "hp":"No HP",
}
missing = [v for v in COL.values() if v not in df.columns]
if missing:
    st.error(f"Kolom berikut belum ada di Excel: {missing}")
    st.stop()

# ================== PILIH BARIS KONTEKS ==================
def pick_rows(question: str, limit: int = 5) -> pd.DataFrame:
    ql = (question or "").lower().strip()
    if not ql:
        return df.head(0)

    # NIP persis atau prefix
    full_nip = re.findall(r"\d{8,20}", ql)
    if full_nip:
        nip_series = df[COL["nip"]].astype(str).str.replace(r"\s", "", regex=True)
        pick = df[nip_series.isin(full_nip)]
        if not pick.empty:
            return pick.head(limit)
    m = re.search(r"\b(\d{2,17})\b", ql)
    if m:
        prefix = m.group(1)
        nip_series = df[COL["nip"]].astype(str).str.replace(r"\s", "", regex=True)
        pick = df[nip_series.str.startswith(prefix, na=False)]
        if not pick.empty:
            return pick.head(limit)

    # Token nama
    tokens = [t for t in re.split(r"[^a-zA-Z]+", ql) if len(t) >= 3]
    if not tokens:
        return df.head(0)
    base = df[COL["nama"]].fillna("").str.lower()
    mask = pd.Series(False, index=df.index)
    for t in tokens:
        mask = mask | base.str.contains(t)
    return df[mask].head(limit)

# ================== FUNGSI OPENROUTER ==================
def ask_openrouter(prompt: str) -> str:
    """
    Panggil model gratis di OpenRouter.
    Daftar model: https://openrouter.ai/models
    """
    api_key = st.secrets.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise KeyError("OPENROUTER_API_KEY belum diisi di Streamlit Secrets.")

    model = st.secrets.get("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Kamu asisten kepegawaian PSEKP. Jawab HANYA berdasarkan data CSV yang diberikan. "
                    "Jika informasi tidak ada, jawab 'data tidak tersedia'. Jawab sopan dan ringkas."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Gagal: {resp.status_code} - {resp.text}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# ================== UI: SATU FIELD PERTANYAAN ==================
query = st.text_input(
    "Tulis pertanyaan lalu tekan Enter",
    placeholder="contoh: 'jabatan restu apa?' atau '1994 siapa saja?' atau 'NIP 1976... jabatannya?'"
)

if query:
    ctx_df = pick_rows(query, limit=5)[
        [COL["nip"], COL["nama"], COL["jf"], COL["js"], COL["gol"], COL["pang"], COL["tmtj"], COL["tmtg"], COL["email"], COL["hp"]]
    ].fillna("-")
    ctx_csv = ctx_df.to_csv(index=False)

    prompt = f"DATA (CSV):\n{ctx_csv}\n\nPERTANYAAN:\n{query}"

    try:
        with st.spinner("Model sedang berpikir…"):
            answer = ask_openrouter(prompt)
        st.success("Jawaban:")
        st.write(answer if answer else "data tidak tersedia")
        with st.expander("Lihat konteks (CSV)"):
            st.code(ctx_csv, language="csv")
    except Exception as e:
        st.error(f"Gagal memanggil OpenRouter: {e}")
