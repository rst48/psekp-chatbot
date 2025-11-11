import pandas as pd
import streamlit as st
import re, requests
from pathlib import Path

st.set_page_config(page_title="PSEKP AI Chat (Ekspresif)", layout="centered")
st.title("Selamat datang di **PSEKP AI Chat** 🤖")
st.caption("Tanya apa saja soal data kepegawaian. Jawaban akan berbasis Excel, namun ditulis dengan bahasa yang lebih natural.")

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

# ================== PILIH BARIS KONTEKS DARI EXCEL ==================
def pick_rows(question: str, limit: int = 5) -> pd.DataFrame:
    ql = (question or "").lower().strip()
    if not ql:
        return df.head(0)

    # 1) NIP persis (8–20 digit)
    full_nip = re.findall(r"\d{8,20}", ql)
    if full_nip:
        nip_series = df[COL["nip"]].astype(str).str.replace(r"\s", "", regex=True)
        pick = df[nip_series.isin(full_nip)]
        if not pick.empty:
            return pick.head(limit)

    # 2) Awalan NIP (2–17 digit)
    m = re.search(r"\b(\d{2,17})\b", ql)
    if m:
        prefix = m.group(1)
        nip_series = df[COL["nip"]].astype(str).str.replace(r"\s", "", regex=True)
        pick = df[nip_series.str.startswith(prefix, na=False)]
        if not pick.empty:
            return pick.head(limit)

    # 3) Token nama (case-insensitive)
    tokens = [t for t in re.split(r"[^a-zA-Z]+", ql) if len(t) >= 3]
    if not tokens:
        return df.head(0)
    base = df[COL["nama"]].fillna("").str.lower()
    mask = pd.Series(False, index=df.index)
    for t in tokens:
        mask = mask | base.str.contains(t)
    return df[mask].head(limit)

# ================== FUNGSI OPENROUTER (EKSPRESIF + GROUNDED) ==================
def ask_openrouter(prompt: str, temperature: float = 0.6, model: str = None) -> str:
    api_key = st.secrets.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise KeyError("OPENROUTER_API_KEY belum diisi di Streamlit Secrets.")

    model = model or st.secrets.get("OPENROUTER_MODEL", "meta-llama/llama-3-8b-instruct")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Prompt sistem yang lebih natural tapi tetap patuh data
    system_prompt = (
        "Kamu adalah asisten kepegawaian PSEKP yang ramah dan profesional.\n"
        "- Dasarkan jawaban pada DATA CSV yang diberikan sebagai sumber fakta.\n"
        "- Ekspresikan jawaban secara alami seperti mengetik manual: kalimat mengalir, tidak kaku, boleh ringkas atau bullet seperlunya.\n"
        "- Bila ada beberapa kandidat, rangkum rapi: sebutkan yang paling relevan lalu alternatif.\n"
        "- Jangan menebak; jika data tak ada, tulis 'data tidak tersedia'.\n"
        "- Jangan menampilkan CSV mentah; gunakan narasi yang enak dibaca.\n"
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        # Temperature disesuaikan agar lebih natural
        "temperature": temperature,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    if resp.status_code != 200:
        raise RuntimeError(f"Gagal: {resp.status_code} - {resp.text}")
    data = resp.json()
    return data["choices"][0]["message"]["content"]

# ================== UI: GAYA & PERTANYAAN ==================
col1, col2 = st.columns([1,1])
with col1:
    style = st.selectbox("Gaya jawaban", ["Standard", "Natural", "Super Natural"], index=1,
                         help="Standard: ringkas & faktual; Natural: seperti ngobrol; Super Natural: paling ekspresif.")
with col2:
    model_choice = st.selectbox("Model", [
        "meta-llama/llama-3-8b-instruct",
        "nousresearch/nous-hermes-2-mixtral-8x7b-dpo",
        "mistralai/mistral-7b-instruct",
    ], index=0)

temperature_map = {"Standard": 0.3, "Natural": 0.6, "Super Natural": 0.85}
temperature = temperature_map[style]

query = st.text_input(
    "Tulis pertanyaan lalu tekan Enter",
    placeholder="contoh: 'jabatan restu apa?' atau '1994 siapa saja?' atau 'NIP 1976... jabatannya?'"
)

if query:
    # Siapkan konteks dari Excel jadi CSV (dipakai model sebagai referensi fakta)
    ctx_df = pick_rows(query, limit=5)[
        [COL["nip"], COL["nama"], COL["jf"], COL["js"], COL["gol"], COL["pang"], COL["tmtj"], COL["tmtg"], COL["email"], COL["hp"]]
    ].fillna("-")
    ctx_csv = ctx_df.to_csv(index=False)

    # Prompt user: minta jawab alami + tetap merujuk data
    user_prompt = (
        "Gunakan gaya bahasa alami dan sopan.\n"
        "Ringkas inti jawaban, lalu jika perlu tambahkan detail penting.\n"
        "DATA (CSV) — jangan tampilkan mentah, cukup jadikan referensi fakta:\n"
        f"{ctx_csv}\n\n"
        f"PERTANYAAN:\n{query}"
    )

    try:
        with st.spinner("Menyusun jawaban…"):
            answer = ask_openrouter(user_prompt, temperature=temperature, model=model_choice)
        st.success("Jawaban:")
        st.write(answer if answer else "data tidak tersedia")
        with st.expander("Konteks (CSV) yang dipakai — untuk audit"):
            st.code(ctx_csv, language="csv")
    except Exception as e:
        st.error(f"Gagal memanggil OpenRouter: {e}")
