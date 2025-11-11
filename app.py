import pandas as pd
import streamlit as st
import re

st.set_page_config(page_title="Chatbot Kepegawaian (PoC)", layout="centered")

st.title("Chatbot Kepegawaian — PoC (PSEKP)")
st.write("Unggah *Excel* dengan sheet bernama **DATA** dan header kolom sesuai template.")

uploaded = st.file_uploader("Unggah Excel (sheet DATA)", type=["xlsx"])
if not uploaded:
    st.info("Unggah file Excel kamu dulu (kolom seperti contoh).")
    st.stop()

# Baca sheet DATA
try:
    df = pd.read_excel(uploaded, sheet_name="DATA")
except Exception:
    st.error("Gagal membaca sheet 'DATA'. Pastikan sheet bernama 'DATA' (huruf besar semua).")
    st.stop()

# Bersihkan spasi tak terlihat
for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].astype(str).str.replace("\xa0", " ", regex=False).str.strip()

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

st.success("File berhasil dimuat. Ketik NIP (angka) atau `cari <nama>` lalu tekan Enter.")

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

    # Pencarian nama mengandung kata
    elif q.lower().startswith("cari "):
        name = q[5:].strip().lower()
        hits = df[df[COL["nama"]].str.lower().str.contains(name, na=False)]
        if hits.empty:
            st.warning(f"Tidak ada nama mengandung: {name}")
        else:
            st.markdown(f"Ditemukan {len(hits)} data. Menampilkan maksimal 10 hasil:")
            for _, r in hits.head(10).iterrows():
                st.markdown("---")
                st.markdown(format_row(r.to_dict()))

    else:
        st.info("Format: ketik NIP (angka) atau `cari <nama>`.")
