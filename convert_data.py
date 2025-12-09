import pandas as pd
import json
import os

# --- KONFIGURASI NAMA FILE ---
# Pastikan nama file CSV sesuai dengan yang Anda download dari GitHub/Scopus
INPUT_CSV = "20241121-scopus.csv" 
OUTPUT_JSON = "20241121-scopus.json"

def convert_pure_raw(csv_path, json_path):
    print(f"📂 Membaca file: {csv_path}...")
    
    try:
        # 1. Baca CSV
        # 'keep_default_na=False' penting agar data kosong tidak jadi NaN (JSON tidak bisa baca NaN),
        # melainkan jadi string kosong "".
        # 'encoding' dicoba utf-8 dulu, kalau gagal pakai latin1 (common issue di file excel/csv).
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig', keep_default_na=False)
        except UnicodeDecodeError:
            print("⚠️ Encoding UTF-8 gagal, mencoba encoding Latin-1...")
            df = pd.read_csv(csv_path, encoding='latin1', keep_default_na=False)

        # 2. Konversi ke Dictionary (Raw)
        # orient='records' akan membuat list of objects: [{"kolom1": "isi", "kolom2": "isi"}, ...]
        # Ini akan mengambil SEMUA kolom yang ada di CSV tanpa kecuali.
        data_list = df.to_dict(orient='records')

        # 3. Simpan ke JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, indent=2, ensure_ascii=False)

        print(f"✅ SUKSES! Konversi selesai.")
        print(f"📊 Total Data: {len(data_list)} baris.")
        print(f"💾 Output disimpan di: {json_path}")
        
        # 4. Tampilkan Nama Kolom (PENTING untuk langkah selanjutnya)
        if data_list:
            print("\nℹ️  Daftar Kolom (Keys) yang tersimpan (Copy nama ini untuk script ekstraksi):")
            print(list(data_list[0].keys()))

    except FileNotFoundError:
        print(f"❌ ERROR: File '{csv_path}' tidak ditemukan. Pastikan file ada di folder yang sama.")
    except Exception as e:
        print(f"❌ ERROR: Terjadi kesalahan: {e}")

# Jalankan Fungsi
if __name__ == "__main__":
    convert_pure_raw(INPUT_CSV, OUTPUT_JSON)