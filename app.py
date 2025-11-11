import pandas as pd
import streamlit as st
import re, requests
from pathlib import Path
from bs4 import BeautifulSoup

# ================== KONFIGURASI ==================
st.set_page_config(page_title="PSEKP AI Chat", layout="centered")
st.title("Selamat datang di **PSEKP AI Chat** 🤖")
st.caption("Tanya apa saja seputar kepegawaian atau informasi tentang PSEKP")

DATA_XLSX = Path("data/kepegawaian.xlsx")
MODEL_DEFAULT = "meta-llama/llama-3-8b-instruct"
MAX_ROWS = 5
MAX_WEB_SNIPS = 3

# ================== SUMBER WEBSITE DEFAULT ==================
DEFAULT_URLS = [
    "https://psekp.setjen.pertanian.go.id/web/",
    "https://psekp.setjen.pertanian.go.id/web/?page_id=396",
    "https://psekp.setjen.pertanian.go.id/web/?page_id=594"
]

# ================== BACA DATA EXCEL ==================
if not DATA_XLSX.exists():
    st.error("❌ File **data/kepegawaian.xlsx** tidak ditemukan di repo.")
    st.stop()

try:
    df = pd.read_excel(DATA_XLSX, sheet_name="DATA", dtype=str)
except ValueError:
    st.error("❌ Sheet 'DATA' tidak ditemukan. Ubah nama sheet menjadi 'DATA'.")
    st.stop()
except Exception as e:
    st.error(f"❌ Gagal membaca Excel: {e}")
    st.stop()

# Bersihkan karakter tak terlihat
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

# ================== UTILITAS ==================
def tokens_from_query(q: str):
    return [t for t in re.split(r"[^a-zA-Z0-9]+", (q or '').lower()) if len(t) >= 3]

def score_text(text: str, toks):
    t = (text or '').lower()
    return sum(t.count(tok) for tok in toks)

# ================== PILIH BARIS EXCEL ==================
def pick_rows_excel(question: str, limit=MAX_ROWS) -> pd.DataFrame:
    ql = (question or '').lower().strip()
    if not ql:
        return df.head(0)
    full_nip = re.findall(r'\d{8,20}', ql)
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
    toks = tokens_from_query(ql)
    base = df[COL["nama"]].fillna("").str.lower()
    mask = pd.Series(False, index=df.index)
    for t in toks:
        mask = mask | base.str.contains(t)
    return df[mask].head(limit)

# ================== WEBSITE ==================
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_url_text(url: str) -> str:
    """Ambil teks dari website dan bersihkan HTML, disimpan cache 1 jam."""
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent":"Mozilla/5.0"})
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script","style","nav","footer","header"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)
        lines = [ln for ln in text.splitlines() if len(ln.strip()) > 40]
        return "\n".join(lines[:1000])
    except Exception:
        return ""

def pick_snippets_web(question: str, urls: list, max_snips=MAX_WEB_SNIPS):
    toks = tokens_from_query(question)
    if not toks or not urls:
        return []
    snips = []
    for url in urls:
        raw = fetch_url_text(url.strip())
        if not raw:
            continue
        paras = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
        scored = sorted(paras, key=lambda p: score_text(p, toks), reverse=True)
        for para in scored[:2]:
            if score_text(para, toks) > 0:
                snips.append(para)
    snips = sorted(snips, key=lambda x: score_text(x, toks), reverse=True)[:max_snips]
    return snips

# ================== LLM (OpenRouter) ==================
def ask_openrouter(prompt: str, temperature=0.7) -> str:
    api_key = st.secrets.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise KeyError("❌ OPENROUTER_API_KEY belum diisi di Secrets.")
    model = st.secrets.get("OPENROUTER_MODEL", MODEL_DEFAULT)

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    system_prompt = (
        "Kamu asisten kepegawaian PSEKP yang ramah dan profesional.\n"
        "- Gunakan gaya bahasa alami seperti mengetik manual.\n"
        "- Jawaban harus berdasarkan konteks dari Excel dan Website (tanpa menampilkan URL).\n"
        "- Tambahkan penanda sumber di akhir kalimat fakta: [Excel] atau [Web].\n"
        "- Jika data tidak memadai, jawab 'data tidak tersedia'.\n"
        "- Hindari menyebut URL atau tautan secara eksplisit."
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
    return resp.json()["choices"][0]["message"]["content"]

# ================== UI ==================
query = st.text_input(
    "Tulis pertanyaan lalu tekan Enter",
    placeholder="contoh: 'Apa tugas dan fungsi PSEKP?', atau 'Jabatan Eko Nugroho apa?'"
)

if query:
    # Ambil data Excel relevan
    ctx_df = pick_rows_excel(query, limit=MAX_ROWS)[
        [COL["nip"], COL["nama"], COL["jf"], COL["js"], COL["gol"], COL["pang"], COL["tmtj"], COL["tmtg"], COL["email"], COL["hp"]]
    ].fillna("-")
    ctx_csv = ctx_df.to_csv(index=False)

    # Ambil data dari web
    web_snips = pick_snippets_web(query, DEFAULT_URLS, MAX_WEB_SNIPS)

    # Satukan konteks
    parts = []
    if not ctx_df.empty:
        parts.append("Sumber: [Excel]\n" + ctx_csv)
    for para in web_snips:
        parts.append("Sumber: [Web]\n" + para)

    context_block = "\n\n---\n\n".join(parts) if parts else "(KONTEKS KOSONG)"
    prompt = (
        "Gunakan gaya bahasa alami seperti mengetik manual. "
        "Saat menulis fakta, tambahkan penanda sumber sesuai konteks (hanya [Excel] atau [Web]).\n\n"
        f"KONTEKS TERKURASI (boleh mengacu, jangan tampilkan mentah):\n{context_block}\n\n"
        f"PERTANYAAN:\n{query}"
    )

    try:
        with st.spinner("💬 Menyusun jawaban dari Excel & Website..."):
            answer = ask_openrouter(prompt, temperature=0.75)
        st.success("Jawaban:")
        st.write(answer if answer else "data tidak tersedia")

        with st.expander("🔎 Konteks yang digunakan (tanpa URL)"):
            st.code(context_block)
    except Exception as e:
        st.error(f"Gagal memanggil model: {e}")


