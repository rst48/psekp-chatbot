import pandas as pd
import streamlit as st
import re, requests
from pathlib import Path
from bs4 import BeautifulSoup

# ================== KONFIGURASI ==================
st.set_page_config(page_title="PSEKP AI Chat", layout="centered")
st.title("Selamat datang di **PSEKP AI Chat** 🤖")
st.caption("Tanya apa saja seputar kepegawaian atau informasi PSEKP. Sumber: Excel & Website resmi PSEKP (OpenRouter).")

DATA_XLSX = Path("data/kepegawaian.xlsx")
MODEL_DEFAULT = "meta-llama/llama-3-8b-instruct"  # override via Secrets: OPENROUTER_MODEL
MAX_CONTEXT_ROWS = 100     # baris untuk LLM (narasi)
MAX_LIST_ROWS    = 500     # baris untuk daftar deterministik (tampilan)
MAX_WEB_SNIPS    = 3

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

# ================== UTIL PENCARIAN ==================
def tokens_from_query(q: str):
    return [t for t in re.split(r"[^a-zA-Z0-9]+", (q or '').lower()) if len(t) >= 3]

def search_all(query: str) -> pd.DataFrame:
    """
    Cari semua kandidat dari:
      - NIP lengkap (8–20 digit)
      - Prefix NIP (2–17 digit) -> semua yang diawali prefix tsb
      - Token nama (>=3 huruf) case-insensitive
    Hasil digabung (unik) TANPA BATAS, nanti dipotong saat tampil.
    """
    q = (query or "").strip()
    ql = q.lower()
    out_idx = pd.Index([])

    # NIP lengkap
    full_nips = re.findall(r"\d{8,20}", ql)
    if full_nips:
        nip_series = df[COL["nip"]].astype(str).str.replace(r"\s", "", regex=True)
        mask_full = nip_series.isin(full_nips)
        out_idx = out_idx.union(df[mask_full].index)

    # Prefix NIP (tangkap SEMUA token angka 2–17 digit DI DALAM kalimat)
    prefixes = re.findall(r"\b(\d{2,17})\b", ql)
    if prefixes:
        nip_series = df[COL["nip"]].astype(str).str.replace(r"\s", "", regex=True)
        mask_pref_total = pd.Series(False, index=df.index)
        for pref in prefixes:
            mask_pref_total = mask_pref_total | nip_series.str.startswith(pref, na=False)
        out_idx = out_idx.union(df[mask_pref_total].index)

    # Nama
    name_tokens = [t for t in re.split(r"[^a-zA-Z]+", ql) if len(t) >= 3]
    if name_tokens:
        base = df[COL["nama"]].fillna("").str.lower()
        mask_name = pd.Series(False, index=df.index)
        for t in name_tokens:
            mask_name = mask_name | base.str.contains(t, na=False)
        out_idx = out_idx.union(df[mask_name].index)

    return df.loc[out_idx].fillna("-")

def build_llm_context(query: str, limit_rows: int = MAX_CONTEXT_ROWS) -> str:
    hits = search_all(query)
    # pilih kolom penting untuk konteks
    ctx_df = hits[[COL["nip"], COL["nama"], COL["jf"], COL["js"], COL["gol"], COL["pang"], COL["tmtj"], COL["tmtg"], COL["email"], COL["hp"]]].head(limit_rows)
    ctx_csv = ctx_df.to_csv(index=False)
    return ctx_csv, hits  # return csv + full hits (untuk daftar deterministik)

# ================== WEBSITE (tanpa tampilkan URL) ==================
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_url_text(url: str) -> str:
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
def ask_openrouter(prompt: str, temperature=0.3) -> str:
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
        "- Jika konteks berisi BANYAK pegawai, tampilkan SELURUHNYA sebagai daftar berpoin "
        "(Nama — NIP — Jabatan; JF lalu JS jika JF kosong) dan beri tag [Excel] tiap poin.\n"
        "- Tambahkan penanda sumber di akhir kalimat fakta hanya: [Excel] atau [Web].\n"
        "- Jika data tidak memadai, jawab 'data tidak tersedia'.\n"
        "- Jangan menampilkan URL/domain."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": prompt},
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
    placeholder="contoh: 'Siapa saja yang punya NIP 1990?', 'Jabatan Restu apa?'"
)

if query:
    # 1) Bangun konteks untuk LLM + kumpulkan semua hit (full)
    ctx_csv, all_hits = build_llm_context(query, limit_rows=MAX_CONTEXT_ROWS)

    # 2) Ambil snippet website (tanpa URL)
    web_snips = web_snippets(query, DEFAULT_URLS, MAX_WEB_SNIPS)

    # 3) Tampilkan daftar deterministik SEMUA pegawai yang cocok (jika ada >0)
    if not all_hits.empty:
        st.markdown(f"### Hasil dari Excel (total {len(all_hits)}):")
        # Batasi tampilan panjang
        show_hits = all_hits.head(MAX_LIST_ROWS)
        for _, r in show_hits.iterrows():
            nama = r[COL["nama"]]
            nip  = r[COL["nip"]]
            jab  = (r.get(COL["jf"]) or r.get(COL["js"]) or "-")
            st.write(f"- **{nama}** — NIP: `{nip}` — {jab} [Excel]")
        if len(all_hits) > MAX_LIST_ROWS:
            st.info(f"Ditampilkan {MAX_LIST_ROWS} pertama dari {len(all_hits)} hasil.")

    # 4) Rangkai context block untuk LLM
    parts = []
    if not all_hits.empty:
        parts.append("Sumber: [Excel]\n" + ctx_csv)  # hanya subset (MAX_CONTEXT_ROWS) agar token hemat
    for para in web_snips:
        parts.append("Sumber: [Web]\n" + para)
    context_block = "\n\n---\n\n".join(parts) if parts else "(KONTEKS KOSONG)"

    # 5) Prompt ke LLM (narasi + aturan enumerasi)
    prompt = (
        "Gunakan gaya bahasa alami seperti mengetik manual.\n"
        "Jika konteks berisi BANYAK pegawai, TAMPILKAN SEMUANYA sebagai daftar berpoin: "
        "Nama — NIP — Jabatan (ambil JF lalu JS jika JF kosong) dan beri tag [Excel] tiap poin. "
        "Setelah daftar, boleh tambahkan ringkasan singkat.\n\n"
        f"KONTEKS TERKURASI (jangan tampilkan mentah):\n{context_block}\n\n"
        f"PERTANYAAN:\n{query}"
    )

    try:
        with st.spinner("💬 Menyusun jawaban (Excel & Website)..."):
            answer = ask_openrouter(prompt, temperature=0.3)
        st.success("Jawaban naratif:")
        st.write(answer if answer else "data tidak tersedia")
        with st.expander("🔎 Konteks yang digunakan (tanpa URL)"):
            st.code(context_block)
    except Exception as e:
        st.error(f"Gagal memanggil model: {e}")
