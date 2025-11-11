import pandas as pd
import streamlit as st
import re, requests, time
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

# ================== UTILITAS ==================
def tokens_from_query(q: str):
    raw = [t for t in re.split(r"[^a-zA-Z0-9]+", (q or "").lower()) if len(t) >= 3]
    stop = {"nip"}
    return [t for t in raw if t not in stop]

def search_all(query: str) -> pd.DataFrame:
    q = (query or "").strip().lower()
    out_idx = pd.Index([])
    nip_series = df[COL["nip"]].astype(str).str.replace(r"\s", "", regex=True)
    full_nips = re.findall(r"\d{8,20}", q)
    if full_nips:
        out_idx = out_idx.union(df[nip_series.isin(full_nips)].index)
    prefixes = re.findall(r"\b(\d{2,17})\b", q)
    for pref in prefixes:
        out_idx = out_idx.union(df[nip_series.str.startswith(pref)].index)
    base = df[COL["nama"]].fillna("").str.lower()
    focus_tokens = []
    m = re.search(r"\bnip\s+([a-zA-Z][a-zA-Z\s\-']+)", q)
    if m:
        focus_tokens = [t for t in re.split(r"[^a-zA-Z]+", m.group(1)) if len(t) >= 3]
    name_tokens = tokens_from_query(q)
    mask = pd.Series(False, index=df.index)
    for t in focus_tokens + [t for t in name_tokens if t not in focus_tokens]:
        mask |= base.str.contains(t, na=False)
    out_idx = out_idx.union(df[mask].index)
    return df.loc[out_idx].fillna("-")

def build_llm_context(query: str, limit_rows=MAX_CONTEXT_ROWS) -> str:
    hits = search_all(query)
    ctx = hits[[COL["nip"], COL["nama"], COL["jf"], COL["js"],
                COL["gol"], COL["pang"], COL["tmtj"], COL["tmtg"],
                COL["email"], COL["hp"]]].head(limit_rows)
    return ctx.to_csv(index=False)

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_url_text(url: str) -> str:
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        lines = [ln for ln in soup.get_text("\n", strip=True).splitlines() if len(ln.strip()) > 40]
        return "\n".join(lines[:1000])
    except Exception:
        return ""

def web_snippets(query: str, urls=DEFAULT_URLS, max_snips=MAX_WEB_SNIPS):
    toks = tokens_from_query(query)
    if not toks:
        return []
    def score(t): return sum((t or "").lower().count(tok) for tok in toks)
    snips = []
    for url in urls:
        raw = fetch_url_text(url)
        if not raw: continue
        paras = [p.strip() for p in re.split(r"\n\s*\n", raw) if p.strip()]
        snips += sorted(paras, key=score, reverse=True)[:2]
    return sorted(snips, key=score, reverse=True)[:max_snips]

def ask_openrouter(prompt: str, temperature=0.35) -> str:
    api_key = st.secrets.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise KeyError("❌ OPENROUTER_API_KEY belum diisi di Secrets.")
    model = st.secrets.get("OPENROUTER_MODEL", MODEL_DEFAULT)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Kamu asisten kepegawaian PSEKP yang ramah dan profesional."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    r = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Gagal: {r.status_code} - {r.text}")
    return r.json()["choices"][0]["message"]["content"]

ORG_KEYWORDS = {"tugas","fungsi","tupoksi","struktur","organisasi","visi","misi","layanan","sejarah","profil","mandat","peran","kewenangan","kebijakan"}

def is_org_question(q: str) -> bool:
    return any(k in (q or "").lower() for k in ORG_KEYWORDS)

# ================== INTRO SENA ==================
def typewriter(container, text, delay=0.05, style="###"):
    typed = ""
    for ch in text:
        typed += ch
        container.markdown(f"{style} {typed}")
        time.sleep(delay)

c1 = st.empty()
c2 = st.empty()
typewriter(c1, "Hai 👋, saya SENA, Asisten Cerdas PSEKP", delay=0.05, style="###")
time.sleep(0.6)
typewriter(c2, "Ada yang mau kamu ketahui tentang PSEKP dan Kepegawaiannya?", delay=0.05, style="#####")
st.markdown("---")

# ================== INPUT CHAT SEDERHANA ==================
query = st.text_input(
    "Tuliskan pertanyaanmu, lalu tekan Enter",
    placeholder="contoh: 'Apa tugas PSEKP?', 'Siapa Restu?', 'NIP Frilla?'"
)

# ================== PROSES JAWABAN ==================
if "submitted_query" in st.session_state:
    query = st.session_state["submitted_query"]
    ctx_csv = build_llm_context(query)
    has_excel = bool(ctx_csv.strip())
    org_mode = is_org_question(query)
    web_snips = web_snippets(query)

    parts = []
    if has_excel and not org_mode:
        parts.append("Sumber: [Excel]\n" + ctx_csv)
    for para in web_snips:
        parts.append("Sumber: [Web]\n" + para)
    context_block = "\n\n---\n\n".join(parts) if parts else "(KONTEKS KOSONG)"

    if has_excel and not org_mode:
        user_prompt = (
            "Gunakan gaya bahasa alami seperti mengetik manual.\n"
            "Jika konteks berisi banyak pegawai, tampilkan semuanya sebagai daftar berpoin: "
            "Nama — NIP — Jabatan (ambil JF lalu JS jika JF kosong) dan beri tag [Excel].\n\n"
            f"KONTEKS:\n{context_block}\n\nPERTANYAAN:\n{query}"
        )
        temp = 0.35
    else:
        user_prompt = (
            "Jawablah dengan bahasa alami dan profesional. "
            "Fokus pada informasi dari konteks Website. Tambahkan tag [Web] di akhir fakta.\n\n"
            f"KONTEKS:\n{context_block}\n\nPERTANYAAN:\n{query}"
        )
        temp = 0.4

    try:
        with st.spinner("💬 Menyusun jawaban..."):
            ans = ask_openrouter(user_prompt, temperature=temp)
        st.success("Jawaban:")
        st.write(ans if ans else "data tidak tersedia")

        with st.expander("🔎 Konteks yang digunakan (tanpa URL)"):
            st.code(context_block)
    except Exception as e:
        st.error(f"Gagal memanggil model: {e}")


