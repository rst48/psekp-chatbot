import pandas as pd
import streamlit as st
import re, json
from pathlib import Path

# =============== SETUP APP ===============
st.set_page_config(page_title="PSEKP Chatbot", layout="centered")
st.title("Selamat datang di **PSEKP Chatbot**")
st.caption("Apa yang mau kamu tanya hari ini tentang Kepegawaian PSEKP?")

# =============== BACA DATA DARI REPO ===============
DATA_PATH = Path("data/kepegawaian.xlsx")
if not DATA_PATH.exists():
    st.error("File **data/kepegawaian.xlsx** tidak ditemukan di repo. Pastikan nama & lokasinya benar.")
    st.stop()

try:
    # dtype=str agar NIP tidak berubah jadi angka
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

# Kolom yang dipakai untuk konteks
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

# =============== HELPER: PILIH BARIS KONTEKS ===============
def pick_rows(question: str, limit: int = 5) -> pd.DataFrame:
    """Ambil baris relevan berdasar NIP (persis/prefix) atau token nama (case-insensitive)."""
    ql = (question or "").lower().strip()
    if not ql:
        return df.head(0)

    # 1) NIP persis (8–20 digit) → prioritas utama
    full_nip = re.findall(r"\d{8,20}", ql)
    if full_nip:
        nip_series = df[COL["nip"]].astype(str).str.replace(r"\s", "", regex=True)
        pick = df[nip_series.isin(full_nip)]
        if not pick.empty:
            return pick.head(limit)

    # 2) Prefix NIP (angka < 18 digit)
    m = re.search(r"\b(\d{2,17})\b", ql)
    if m:
        prefix = m.group(1)
        nip_series = df[COL["nip"]].astype(str).str.replace(r"\s", "", regex=True)
        pick = df[nip_series.str.startswith(prefix, na=False)]
        if not pick.empty:
            return pick.head(limit)

    # 3) Token nama
    tokens = [t for t in re.split(r"[^a-zA-Z]+", ql) if len(t) >= 3]
    if not tokens:
        return df.head(0)

    base = df[COL["nama"]].fillna("").str.lower()
    maskname = pd.Series(False, index=df.index)
    for t in tokens:
        maskname = maskname | base.str.contains(t)
    pick = df[maskname]
    return pick.head(limit)

# =============== LLM: GEMINI (AUTO-PICK MODEL) ===============
def get_gemini_model_id():
    """Pilih model Gemini yang tersedia & mendukung generateContent."""
    import google.generativeai as genai
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    prefer = (
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-1.0-pro",
    )
    try:
        models = list(genai.list_models())
        available = set()
        for m in models:
            name = getattr(m, "name", "")
            if name.startswith("models/"):
                name = name.split("/", 1)[1]
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                available.add(name)
        for m in prefer:
            if m in available:
                return m
    except Exception:
        pass
    return "gemini-1.5-flash-latest"  # fallback best-effort

def ask_gemini(question: str, ctx_csv: str) -> str:
    """Tanya Gemini dengan konteks CSV (jawab hanya berdasar data)."""
    import google.generativeai as genai
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model_id = get_gemini_model_id()
    model = genai.GenerativeModel(model_id)

    system = (
        "Kamu asisten kepegawaian PSEKP. Jawab HANYA berdasarkan DATA (CSV) berikut. "
        "Jika info tidak ada di data, jawab: 'data tidak tersedia'. "
        "Jawab ringkas, rapi, dan sebutkan NIP/Nama bila relevan."
    )
    prompt = f"{system}\n\nDATA (CSV):\n{ctx_csv}\n\nPERTANYAAN:\n{question}"

    resp = model.generate_content(prompt)
    return (resp.text or "").strip()

# =============== UI SEDERHANA: SATU KOTAK PERTANYAAN ===============
query = st.text_input("Ketik pertanyaan kamu lalu tekan Enter")

if query:
    # siapkan konteks dari Excel
    ctx_df = pick_rows(query, limit=5)[
        [COL["nip"], COL["nama"], COL["jf"], COL["js"], COL["gol"], COL["pang"], COL["tmtj"], COL["tmtg"], COL["email"], COL["hp"]]
    ].fillna("-")
    ctx_csv = ctx_df.to_csv(index=False)

    try:
        with st.spinner("Gemini sedang berpikir…"):
            answer = ask_gemini(query, ctx_csv)
        st.success("Jawaban:")
        st.write(answer if answer else "data tidak tersedia")
        with st.expander("Lihat konteks (CSV) yang dipakai"):
            st.code(ctx_csv, language="csv")
    except KeyError:
        st.error("GEMINI_API_KEY belum diisi di Streamlit Secrets.")
    except ModuleNotFoundError:
        st.error("Paket `google-generativeai` belum terpasang. Tambahkan di requirements.txt lalu deploy ulang.")
    except Exception as e:
        st.error(f"Gagal memanggil Gemini: {e}")
