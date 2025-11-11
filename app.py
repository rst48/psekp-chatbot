import pandas as pd
import streamlit as st
import re
from pathlib import Path

st.set_page_config(page_title="Chatbot Kepegawaian (PoC)", layout="centered")
st.title("Chatbot Kepegawaian — PoC (PSEKP)")
st.caption("Membaca langsung dari file: `data/kepegawaian.xlsx` (sheet: DATA)")

# === BACA DATA DARI REPO ===
DATA_PATH = Path("data/kepegawaian.xlsx")
if not DATA_PATH.exists():
    st.error("File **data/kepegawaian.xlsx** tidak ditemukan di repo. Pastikan nama dan lokasinya benar.")
    st.stop()

try:
    # dtype=str supaya NIP tidak berubah jadi angka
    df = pd.read_excel(DATA_PATH, sheet_name="DATA", dtype=str)
except ValueError:
    st.error("Sheet **'DATA'** tidak ditemukan. Buka Excel dan ubah nama sheet menjadi 'DATA'.")
    st.stop()
except Exception as e:
    st.error(f"Gagal membaca Excel: {e}")
    st.stop()

# Bersihkan spasi tak terlihat
for c in df.columns:
    if df[c].dtype == "object":
        df[c] = df[c].astype(str).str.replace("\u00A0", " ", regex=False).str.strip()

# Kolom minimal yang wajib ada
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
missing = [v for v in COL.values() if v not in df.columns]
if missing:
    st.error(f"Kolom berikut belum ada di Excel: {missing}")
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

st.success("Ketik NIP (angka) atau `cari <nama>` lalu Enter.")
q = st.text_input("Ketik NIP atau: `cari <nama>`")

if q:
    q = q.strip()
    # Pencarian NIP persis (8–20 digit)
    if re.fullmatch(r"\d{8,20}", q):
        hit = df[df[COL["nip"]].astype(str).str.replace(r"\s","",regex=True).eq(q)]
        if hit.empty:
            st.error(f"NIP {q} tidak ditemukan.")
        else:
            st.markdown(format_row(hit.iloc[0].to_dict()))
    # Pencarian nama
    elif q.lower().startswith("cari "):
        name = q[5:].strip().lower()
        hits = df[df[COL["nama"]].fillna("").str.lower().str.contains(name)]
        if hits.empty:
            st.warning(f"Tidak ada nama mengandung: {name}")
        else:
            st.markdown(f"Ditemukan {len(hits)} data. Menampilkan maksimal 10 hasil:")
            for _, r in hits.head(10).iterrows():
                st.markdown("---")
                st.markdown(format_row(r.to_dict()))
    else:
        st.info("Format: ketik NIP (angka) atau `cari <nama>`.")
