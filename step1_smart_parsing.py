import json

def prepare_text_for_llm(paper_json):
    """
    Fungsi Smart Parsing.
    Mengambil Title, Abstract, dan menyaring bab Method/Experiment dari full_content.
    """
    title = paper_json.get('title', '')
    abstract = paper_json.get('abstract', '')
    full_content = paper_json.get('full_content', {})
    
    # 1. Mulai dengan Abstrak (Wajib)
    relevant_text = f"Title: {title}\nAbstract: {abstract}\n\n"
    
    # 2. Ambil Bagian Metode (Cari key yang mengandung kata kunci)
    method_keywords = ['method', 'approach', 'model', 'proposed', 'system design', 'architecture']
    method_text = ""
    
    # 3. Ambil Bagian Eksperimen (Untuk Dataset & Metric)
    exp_keywords = ['experiment', 'evaluation', 'result', 'dataset', 'performance']
    exp_text = ""

    if isinstance(full_content, dict):
        for key, text in full_content.items():
            key_lower = key.lower()
            
            # Cek Bab Metode
            if any(k in key_lower for k in method_keywords):
                method_text += f"--- SECTION: {key} ---\n{text[:2500]}\n"
            
            # Cek Bab Eksperimen
            elif any(k in key_lower for k in exp_keywords):
                exp_text += f"--- SECTION: {key} ---\n{text[:2500]}\n"
    
    # Gabungkan semua (Prioritas: Abstract -> Method -> Experiment)
    final_input = relevant_text + "\n=== METHODOLOGY DETAILS ===\n" + method_text + "\n=== EXPERIMENTAL RESULTS ===\n" + exp_text
    return final_input

# ============================================================================
# MAIN: SMART PARSING UNTUK SEMUA PAPER
# ============================================================================

input_filename = "mdpi_computer_science_2020-2025_20251205_234928.json"
output_filename = "hasil_smart_parsing.json"

print("="*70)
print("STEP 1: SMART PARSING (Ekstraksi Text dari JSON)")
print("="*70)

# Baca data input
try:
    with open(input_filename, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict) and "papers" in data:
            papers_list = data["papers"]
        else:
            papers_list = data
except FileNotFoundError:
    print(f"ERROR: File {input_filename} tidak ditemukan!")
    papers_list = []
    exit(1)

print(f"Total paper di dataset: {len(papers_list)}")

# Cek resume
all_smartparsing_data = []
parsed_titles = set()

try:
    with open(output_filename, "r", encoding="utf-8") as f:
        all_smartparsing_data = json.load(f)
        parsed_titles = set([item.get("title") for item in all_smartparsing_data])
        print(f"Resume: Sudah ada {len(all_smartparsing_data)} paper yang di-parse.")
except FileNotFoundError:
    print("Memulai smart parsing dari awal...")

# Loop Smart Parsing untuk SEMUA paper
print(f"\nMemulai proses parsing...")

for i, paper in enumerate(papers_list):
    title = paper.get('title', 'Unknown Title')
    
    if title in parsed_titles:
        continue  # Skip yang sudah di-parse
    
    print(f"[{i+1}/{len(papers_list)}] Parsing: {title[:60]}...")
    
    try:
        smart_text = prepare_text_for_llm(paper)
        
        smartparsing_entry = {
            "paper_id": i,
            "title": title,
            "smart_parsed_text": smart_text
        }
        all_smartparsing_data.append(smartparsing_entry)
        parsed_titles.add(title)
        
        # Autosave setiap 10 paper
        if len(all_smartparsing_data) % 10 == 0:
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(all_smartparsing_data, f, indent=2, ensure_ascii=False)
            print(f"  >> [AUTOSAVE] {len(all_smartparsing_data)} paper di-save.")
            
    except Exception as e:
        print(f"  ERROR: {e}")

# Save final
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(all_smartparsing_data, f, indent=2, ensure_ascii=False)

print("\n" + "="*70)
print(f"SELESAI! {len(all_smartparsing_data)} paper berhasil di-parse.")
print(f"File output: {output_filename}")
print("="*70)
print("\nSelanjutnya jalankan: python step2_graph_extraction.py")
