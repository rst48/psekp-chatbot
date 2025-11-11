import pandas as pd
import streamlit as st
import re, requests
from pathlib import Path

# ================== KONFIGURASI APP ==================
st.set_page_config(page_title="PSEKP AI Chat", layout="centered")
st.title("Selamat datang di **PSEKP AI Chat** 🤖")
st.caption("Tanya apa saja soal data kepegawaian. Jawaban disusun secara alami berdasarkan data Excel di repo.")

# ================== BACA DATA EXCEL ==================
DATA_PATH = Path("data/kepegawaian.xlsx")
if not DATA_PATH.exists():
    st.error("❌ File **data/kepegawaian.xlsx** tidak ditemukan di repo.")
    st.stop()

try:
    df = pd.read_excel(DATA_PATH, sheet_name="DATA", dtype=str)
except ValueError:
    st.error("❌ Sheet **'DATA'** tidak ditemukan. Ubah nama sheet Excel menjadi 'DATA'.")
    st.stop()
except Exception as e:
    st.error(f"❌ Gagal membaca Excel: {e}")
    st.stop()

# Bersihkan spasi tak terlihat
for c in df.columns:
    if df[c].dtype == "object":
        df[c] = df[c].astype(str).str.replace("\u00A0", " ", regex=False).str.strip()

# Kolom utama
COL = {
    "nip": "NIP", "nama": "Nama",
    "jf": "Jabatan Fungsional Tertentu", "js": "Jabatan Struktural",
    "gol": "Golongan Pegawai Saat Ini", "pang": "Pangkat Pegawai Saat Ini",
    "tmtj": "TMT Jabatan", "tmtg": "TMT Golongan Saat Ini",
    "email": "Email", "hp": "No HP",
}
missing = [v for v in COL.values() if v not in df.columns]
if missing:
    st.error(f"Kolom berikut belum ada di Excel: {missing}")
    st.stop()

# ================== PILIH BARIS KONTEKS DARI EXCEL ==================
def pick_rows(question: str, limit: int = 5) -> pd.DataFrame:
    """Ambil baris relevan berdasarkan NIP (persis/prefix) atau token nama."""
    ql = (question or "").lower().strip()
    if not ql:
        return df.head(0)

    # Cari NIP lengkap
    full_nip = re.findall(r"\d{8,20}", ql)
    if full_nip:
        nip_series = df[COL["nip"]].astype(str).str.replace(r"\s", "", regex=True)
        pick = df[nip_series.isin(full_nip)]
        if not pick.empty:
            return pick.head(limit)

    # Cari awalan NIP
    m = re.search(r"\b(\d{2,17})\b", ql)
    if m:
        prefix = m.group(1)
        nip_series = df[COL["nip"]].astype(str).str.replace(r"\s", "", regex=True)
        pick = df[nip_series.str.startswith(prefix, na=False)]
        if not pick.empty:
            return pick.head(limit)

    # Cari berdasarkan nama
    tokens = [t for t in re.split(r"[^a-zA-Z]+", ql) if len(t) >= 3]
    if not tokens:
        return df.head(0)
    base = df[COL["nama"]].fillna("").str.lower()
    mask = pd.Series(False, index=df.index)
    for t in tokens:
        mask = mask | base.str.contains(t)
    return df[mask].head(limit)

# ================== PANGGIL MODEL DARI OPENROUTER ==================
def ask_openrouter(prompt: str, temperature: float = 0.75) -> str:
    """
    Kirim prompt ke model OpenRouter.
    Model dibaca dari Secrets: OPENROUTER_MODEL.
    """
    api_key = st.secrets.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise KeyError("❌ OPENROUTER_API_KEY belum diisi di Streamlit Secrets.")

    model = st.secrets.get("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    system_prompt = (
        "Kamu adalah asisten kepegawaian PSEKP yang ramah dan profesional. "
        "Gunakan gaya bahasa alami seperti mengetik manual. "
        "Gunakan data CSV berikut sebagai referensi utama untuk menjawab pertanyaan. "
        "Jangan tampilkan CSV mentah, tapi tuliskan penjelasan dengan kalimat alami. "
        "Jika data tidak tersedia, sampaikan dengan sopan bahwa data tidak ada."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Gagal: {resp.status_code} - {resp.text}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# ================== UI: SATU FIELD PERTANYAAN ==================
query = st.text_input(
    "Tulis pertanyaan lalu tekan Enter",
    placeholder="contoh: 'jabatan restu apa?' atau 'NIP 1976... jabatannya?'"
)

if query:
    # Ambil baris relevan dari Excel
    ctx_df = pick_rows(query, limit=5)[
        [COL["nip"], COL["nama"], COL["jf"], COL["js"], COL["gol"], COL["pang"], COL["tmtj"], COL["tmtg"], COL["email"], COL["hp"]]
    ].fillna("-")
    ctx_csv = ctx_df.to_csv(index=False)

    # Siapkan prompt untuk LLM
    prompt = (
        "Gunakan gaya bahasa alami seperti mengetik manual.\n"
        "DATA (CSV):\n"
        f"{ctx_csv}\n\n"
        f"PERTANYAAN:\n{query}"
    )

    try:
        with st.spinner("💬 Menyusun jawaban alami..."):
            answer = ask_openrouter(prompt)
        st.success("Jawaban:")
        st.write(answer if answer else "data tidak tersedia")
        with st.expander("Konteks (CSV) yang digunakan"):
            st.code(ctx_csv, language="csv")
    except Exception as e:
        st.error(f"Gagal memanggil OpenRouter: {e}")
