# ================== IMPORT LIBRARY ==================
import streamlit as st
import pandas as pd
import requests
import re
import time
from pathlib import Path
from bs4 import BeautifulSoup

# ================== KONFIGURASI ==================
st.set_page_config(page_title="PSEKP AI Chat", layout="centered")
st.title("**PSEKP AI Chat** 🤖")
st.caption("Powered by OpenRouter")

DATA_XLSX = Path("data/kepegawaian.xlsx")
MODEL_DEFAULT = "meta-llama/llama-3-8b-instruct"  # bisa override via Secrets
MAX_CONTEXT_ROWS = 100
MAX_WEB_SNIPS = 3
DEFAULT_URLS = [
    "https://psekp.setjen.pertanian.go.id/web/",
    "https://psekp.setjen.pertanian.go.id/web/?page_id=396",
    "https://psekp.setjen.pertanian.go.id/web/?page_id=594",
]

# ================== BACA DATA EXCEL ==================
if not DATA_XLSX.exists():
    st.error("❌ File **data/kepegawaian.xlsx** tidak ditemukan di repo.")
    st.stop()

try:
    df = pd.read_excel(DATA_XLSX, sheet_name="DATA", dtype=str)
except ValueError:
    st.error("❌ Sheet 'DATA' tidak ditemukan. Ubah nama sheet Excel menjadi 'DATA'.")
    st.stop()
except Exception as e:
    st.error(f"❌ Gagal membaca Excel: {e}")
    st.stop()

# Bersihkan karakter tak terlihat
for c in df.columns:
    if df[c].dtype == "object":
        df[c] = df[c].astype(str).str.replace("\u00A0", " ", regex=False).str.strip()

COL = {
    "nip": "NIP",
    "nama": "Nama",
    "jf": "Jabatan Fungsional Tertentu",
    "js": "Jabatan Struktural",
    "gol": "Golongan Pegawai Saat Ini",
    "pang": "Pangkat Pegawai Saat Ini",
    "tmtj": "TMT Jabatan",
    "tmtg": "TMT Golongan Saat Ini",
    "email": "Email",
    "hp": "No HP",
}

missing = [v for v in COL.values() if v not in df.columns]
if missing:
    st.error(f"Kolom berikut belum ada di Excel: {missing}")
    st.stop()

# ================== UTILITAS PENCARIAN ==================
def tokens_from_query(q: str):
    raw = [t for t in re.split(r"[^a-zA-Z0-9]+", (q or "").lower()) if len(t) >= 3]
    stop = {"nip"}
    return [t for t in raw if t not in stop]

def search_all(query: str) -> pd.DataFrame:
    q = (query or "").strip()
    ql = q.lower()
    out_idx = pd.Index([])

    nip_series = df[COL["nip"]].astype(str).str.replace(r"\s", "", regex=True)

    full_nips = re.findall(r"\d{8,20}", ql)
    if full_nips:
        mask_full = nip_series.isin(full_nips)
        out_idx = out_idx.union(df[mask_full].index)

    prefixes = re.findall(r"\b(\d{2,17})\b", ql)
    if prefixes:
        mask_pref = pd.Series(False, index=df.index)
        for pref in prefixes:
            mask_pref = mask_pref | nip_series.str.startswith(pref, na=False)
        out_idx = out_idx.union(df[mask_pref].index)

    base = df[COL["nama"]].fillna("").str.lower()
    focus_tokens = []
    m = re.search(r"\bnip\s+([a-zA-Z][a-zA-Z\s\-']+)", ql)
    if m:
        focus_tokens = [t for t in re.split(r"[^a-zA-Z]+", m.group(1)) if len(t) >= 3]
    name_tokens = tokens_from_query(ql)
    mask_name = pd.Series(False, index=df.index)
    for t in focus_tokens:
        mask_name = mask_name | base.str.contains(t, na=False)
    for t in [t for t in name_tokens if t not in focus_tokens]:
        mask_name = mask_name | base.str.contains(t, na=False)
    if mask_name.any():
        out_idx = out_idx.union(df[mask_name].index)

    return df.loc[out_idx].fillna("-")

def build_llm_context(query: str, limit_rows: int = MAX_CONTEXT_ROWS) -> str:
    hits = search_all(query)
    ctx_df = hits[
        [
            COL["nip"], COL["nama"], COL["jf"], COL["js"], COL["gol"],
            COL["pang"], COL["tmtj"], COL["tmtg"], COL["email"], COL["hp"],
        ]
    ].head(limit_rows)
    return ctx_df.to_csv(index=False)

# ================== WEBSITE PARSER ==================
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_url_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)
        lines = [ln for ln in text.splitlines() if len(ln.strip()) > 40]
        return "\n".join(lines[:1000])
    except Exception:
        return ""

def web_snippets(query: str, urls=DEFAULT_URLS, max_snips=MAX_WEB_SNIPS):
    toks = tokens_from_query(query)
    if not toks:
        return []

    def score_text(t: str):
        tl = (t or "").lower()
        return sum(tl.count(tok) for tok in toks)

    snips = []
    for url in urls:
        raw = fetch_url_text(url)
        if not raw:
            continue
        paras = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
        top = sorted(paras, key=score_text, reverse=True)[:2]
        for p in top:
            if score_text(p) > 0:
                snips.append(p)
    snips = sorted(snips, key=score_text, reverse=True)[:max_snips]
    return snips

# ================== LLM (OpenRouter) ==================
def ask_openrouter(prompt: str, temperature=0.35) -> str:
    api_key = st.secrets.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise KeyError("❌ OPENROUTER_API_KEY belum diisi di Secrets.")
    model = st.secrets.get("OPENROUTER_MODEL", MODEL_DEFAULT)
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    system_prompt = (
        "Kamu asisten kepegawaian PSEKP yang ramah dan profesional.\n"
        "- Gaya bahasa alami seperti mengetik manual.\n"
        "- Jawaban harus berdasarkan konteks (Excel dan Website) yang diberikan.\n"
        "- Jika banyak pegawai, tampilkan daftar berpoin dengan tag [Excel].\n"
        "- Tambahkan penanda sumber di akhir kalimat: [Excel] atau [Web].\n"
        "- Jika data tidak memadai, jawab 'data tidak tersedia'.\n"
        "- Jangan tampilkan URL/domain."
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

# ================== DETEKSI PERTANYAAN ORGANISASI ==================
ORG_KEYWORDS = {
    "tugas","fungsi","tupoksi","struktur","organisasi","visi","misi",
    "layanan","sejarah","profil","mandat","peran","kewenangan","kebijakan",
}
def is_org_question(q: str) -> bool:
    ql = (q or "").lower()
    return any(k in ql for k in ORG_KEYWORDS)

# ================== INTRO SENA ==================
if "sena_intro_done" not in st.session_state:
    st.session_state.sena_intro_done = False

def typewriter(container, text, delay=0.05, style="###"):
    typed = ""
    for char in text:
        typed += char
        container.markdown(f"{style} {typed}")
        time.sleep(delay)

if not st.session_state.sena_intro_done:
    container1 = st.empty()
    container2 = st.empty()
    intro_line1 = "Hai 👋, saya SENA, Asisten Cerdas PSEKP"
    intro_line2 = "Ada yang mau kamu ketahui tentang PSEKP dan Kepegawaiannya?"
    typewriter(container1, intro_line1, delay=0.05, style="###")
    time.sleep(0.8)
    typewriter(container2, intro_line2, delay=0.05, style="#####")
    st.session_state.sena_intro_done = True
    st.markdown("---")

# ================== INPUT DENGAN TOMBOL HIJAU ==================
st.markdown(
    """
    <style>
    .chat-box {
        display: flex;
        align-items: center;
        background-color: #f8f9fa;
        border-radius: 2rem;
        border: 1px solid #ddd;
        padding: 0.3rem 0.8rem;
    }
    .chat-box input[type="text"] {
        border: none;
        outline: none;
        flex: 1;
        background: transparent;
        padding: 0.4rem;
        font-size: 1rem;
    }
    .send-btn {
        background-color: #16a34a;
        border: none;
        color: white;
        border-radius: 50%;
        width: 2.2rem;
        height: 2.2rem;
        font-size: 1.2rem;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .send-btn:hover {
        background-color: #15803d;
    }
    </style>
    """,
    unsafe_allow_html=True
)

with st.form("chat_form", clear_on_submit=True):
    st.markdown(
        """
        <div class="chat-box">
            <input type="text" name="query" placeholder="Tulis pertanyaanmu di sini..." />
            <button type="submit" class="send-btn">✈️</button>
        </div>
        """,
        unsafe_allow_html=True
    )
    submitted = st.form_submit_button("")

query = st.session_state.get("query", "")
if submitted:
    query = query.strip() if query else ""

# ================== PROSES JAWABAN ==================
if query:
    ctx_csv = build_llm_context(query)
    has_excel = bool(ctx_csv.strip())
    org_mode = is_org_question(query)
    web_snips = web_snippets(query, DEFAULT_URLS, MAX_WEB_SNIPS)

    parts = []
    if has_excel and not org_mode:
        parts.append("Sumber: [Excel]\n" + ctx_csv)
    for para in web_snips:
        parts.append("Sumber: [Web]\n" + para)
    context_block = "\n\n---\n\n".join(parts) if parts else "(KONTEKS KOSONG)"

    if has_excel and not org_mode:
        user_prompt = (
            "Gunakan gaya bahasa alami seperti mengetik manual.\n"
            "Jika banyak pegawai, tampilkan semua sebagai daftar berpoin.\n\n"
            f"KONTEKS:\n{context_block}\n\nPERTANYAAN:\n{query}"
        )
        temp = 0.35
    else:
        user_prompt = (
            "Jawablah dengan bahasa alami dan berdasarkan konteks Website.\n\n"
            f"KONTEKS:\n{context_block}\n\nPERTANYAAN:\n{query}"
        )
        temp = 0.4

    try:
        with st.spinner("💬 Menyusun jawaban..."):
            answer = ask_openrouter(user_prompt, temperature=temp)
        st.success("Jawaban:")
        st.write(answer or "data tidak tersedia")
        with st.expander("🔎 Konteks yang digunakan (tanpa URL)"):
            st.code(context_block)
    except Exception as e:
        st.error(f"Gagal memanggil model: {e}")
