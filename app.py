import pandas as pd
import streamlit as st
import re, json
from pathlib import Path

st.set_page_config(page_title="PSEKP AI Chat", layout="centered")
st.title("Selamat datang di **PSEKP AI Chat**")
st.caption("Apa yang mau kamu tanya hari ini tentang PSEKP? (contoh: 'jabatan restu apa?', '1994 siapa saja?')")

# ================== BACA DATA DARI REPO ==================
DATA_PATH = Path("data/kepegawaian.xlsx")
if not DATA_PATH.exists():
    st.error("File **data/kepegawaian.xlsx** tidak ditemukan di repo. Pastikan nama & lokasinya benar.")
    st.stop()

try:
    df = pd.read_excel(DATA_PATH, sheet_name="DATA", dtype=str)  # dtype=str agar NIP aman
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

# Kolom yang dipakai
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

# ================== AMBIL BARIS KONTEKS ==================
def pick_rows(question: str, limit: int = 5) -> pd.DataFrame:
    """Ambil baris relevan berdasarkan NIP (persis/prefix) atau token nama."""
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

    # 2) Prefix NIP (2–17 digit)
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
    return df[maskname].head(limit)

# ================== GEMINI: PICKER ANTI-404 ==================
def pick_gemini_model_id():
    """
    Ambil model yang mendukung generateContent dari API key ini.
    Mengembalikan nama yang valid (bisa 'models/xxx' atau 'xxx').
    """
    import google.generativeai as genai
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

    prefer = (
        "models/gemini-1.5-flash-latest",
        "models/gemini-1.5-pro-latest",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro",
        "models/gemini-1.0-pro",
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro-latest",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
        "gemini-1.0-pro",
    )

    available = set()
    try:
        for m in genai.list_models():
            name = getattr(m, "name", "")
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                available.add(name)  # biasanya "models/xxx"
                if name.startswith("models/"):
                    available.add(name.split("/", 1)[1])  # simpan juga versi tanpa prefix
    except Exception as e:
        st.warning(f"Tidak bisa membaca daftar model: {e}")
        return None

    for want in prefer:
        if want in available:
            return want
    return None

def ask_gemini(question: str, ctx_csv: str) -> str:
    """Tanya Gemini dengan konteks CSV (jawab hanya berdasar data)."""
    import google.generativeai as genai
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

    model_id = pick_gemini_model_id()
    if not model_id:
        raise RuntimeError(
            "Tidak menemukan model Gemini yang mendukung generateContent untuk API key ini."
        )

    model = genai.GenerativeModel(model_id)
    system = (
        "Kamu asisten kepegawaian PSEKP. Jawab HANYA berdasarkan DATA (CSV) berikut. "
        "Jika info tidak ada di data, jawab: 'data tidak tersedia'. "
        "Jawab ringkas dan rapi; sebutkan NIP/Nama jika relevan."
    )
    prompt = f"{system}\n\nDATA (CSV):\n{ctx_csv}\n\nPERTANYAAN:\n{question}"
    resp = model.generate_content(prompt)
    return (resp.text or "").strip()

# ================== UI: SATU FIELD PERTANYAAN ==================
query = st.text_input(
    "Tulis pertanyaan, lalu tekan Enter",
    placeholder="contoh: 'jabatan restu apa?' atau '1994 siapa saja?' atau 'NIP 1976... jabatannya?'"
)

if query:
    ctx_df = pick_rows(query, limit=5)[
        [COL["nip"], COL["nama"], COL["jf"], COL["js"], COL["gol"], COL["pang"], COL["tmtj"], COL["tmtg"], COL["email"], COL["hp"]]
    ].fillna("-")
    ctx_csv = ctx_df.to_csv(index=False)

    try:
        with st.spinner("Gemini sedang berpikir…"):
            answer = ask_gemini(query, ctx_csv)
        st.success("Jawaban:")
        st.write(answer if answer else "data tidak tersedia")
        with st.expander("Lihat konteks (CSV)"):
            st.code(ctx_csv, language="csv")
    except KeyError:
        st.error("GEMINI_API_KEY belum diisi di Streamlit Secrets.")
    except ModuleNotFoundError:
        st.error("Paket `google-generativeai` belum terpasang. Tambahkan di requirements.txt lalu deploy ulang.")
    except Exception as e:
        st.error(f"Gagal memanggil Gemini: {e}")

# ================== DEBUG: LIHAT MODEL YANG TERSEDIA ==================
with st.expander("Debug model Gemini (opsional)"):
    try:
        import google.generativeai as genai
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        rows = []
        for m in genai.list_models():
            rows.append(f"{getattr(m,'name','?')} — {getattr(m,'supported_generation_methods',[])}")
        if rows:
            st.code("\n".join(rows))
        else:
            st.write("Tidak ada model yang terdaftar untuk API key ini.")
    except Exception as e:
        st.write(f"Gagal menampilkan daftar model: {e}")
