import pandas as pd
import json
import time
from datetime import datetime
import google.generativeai as genai 

# --- KONFIGURASI ---
# genai.configure(api_key='AIzaSyBnuHYy8o7xbCe7DSRH7CXrvZO09Bsqxcs')
genai.configure(api_key='AIzaSyA_IdHWQnRNXTzAogYtPyFR-5LlfUbN9-0')

# File Input (Dataset Baseline)
INPUT_FILENAME = "20241121-scopus.json" 
# File Output (Hasil Ekstraksi)
OUTPUT_FILENAME = "hasil_graph_baseline.json"

def call_llm_api(prompt):
    try:
        model = genai.GenerativeModel('gemma-3-27b-it') 
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"API Error (Retrying...): {e}")
        time.sleep(2)
        return "[]"

def prepare_text_for_llm(paper_json):
    """
    [MODIFIKASI] Format Sederhana untuk Baseline.
    Karena Baseline hanya punya Abstrak, kita tidak perlu memilah bab.
    """
    title = paper_json.get('Title', '')
    abstract = paper_json.get('Abstract', '')
    
    # Format input simpel
    return f"""
    PAPER TITLE: {title}
    
    ABSTRACT:
    {abstract}
    """

def run_llm_pipeline(title, processed_text):
    """
    [MODIFIKASI] Pipeline disesuaikan untuk Domain SUSTAINABILITY (Baseline)
    """
    
    # --- P1: EXTRACT GLOSSARY (Domain: General Science/Sustainability) ---
    prompt_p1 = f"""
    Role: Research Assistant.
    Task: Extract key scientific concepts from the abstract below.
    
    INPUT:
    {processed_text}
    
    INSTRUCTIONS:
    Extract specific terms focusing on:
    1. **Key Concepts**: Main topics (e.g., Greenwashing, Consumer Trust, Sustainable Consumption).
    2. **Methods**: Research methods (e.g., Survey, Structural Equation Modeling, Case Study).
    3. **Variables/Factors**: What is measured? (e.g., Purchase Intention, Willingness to Pay).
    
    Output: Return ONLY a valid JSON array of strings. Example: ["Greenwashing", "Survey", "Trust"]
    """
    
    raw_resp = call_llm_api(prompt_p1)
    try:
        glossary_raw = json.loads(raw_resp.replace("```json", "").replace("```", "").strip())
    except:
        glossary_raw = []
        
    if not glossary_raw: return []

    # --- P2: VERIFICATION (GANTI CSO DENGAN GEMET/TPB) ---
    # [PENTING] Ini bagian yang paling krusial diubah untuk Baseline
    prompt_p2 = f"""
    Role: Domain Expert in **Sustainable Consumer Behavior** and **Environmental Psychology**.
    Input List: {glossary_raw}
    
    TASK: Filter and Normalize terms using these standards as Ground Truth:
    1. **GEMET (General Multilingual Environmental Thesaurus)** for environmental terms.
    2. **Theory of Planned Behavior (TPB)** for psychological/behavioral terms.
    3. **SDG 12 (Responsible Consumption)** for policy/goal terms.
    
    RULES:
    1. KEEP valid concepts (e.g., 'Pro-environmental Behavior', 'Eco-label').
    2. REMOVE generic words (e.g., 'paper', 'significant results', 'aim', 'study').
    3. NORMALIZE to Canonical Noun Phrases (e.g., "behaving green" -> "Pro-environmental Behavior").
    
    Output: Return ONLY a valid JSON array of strings.
    """
    
    clean_resp = call_llm_api(prompt_p2)
    try:
        glossary_clean = json.loads(clean_resp.replace("```json", "").replace("```", "").strip())
    except:
        glossary_clean = glossary_raw

    # --- P3: GRAPH CONSTRUCTION (Predikat Umum) ---
    prompt_p3 = f"""
    Role: Knowledge Graph Architect.
    Terms: {glossary_clean}
    Title: "{title}"
    
    TASK: Map terms into a Knowledge Graph based on the abstract.
    
    USE PREDICATES:
    - INVESTIGATES (e.g., 'Study' INVESTIGATES 'Consumer Choice')
    - AFFECTS (e.g., 'Price' AFFECTS 'Purchase Decision')
    - MEASURED_BY (e.g., 'Attitude' MEASURED_BY 'Likert Scale')
    - USES_METHOD (e.g., 'Study' USES_METHOD 'Survey')
    - SUBCLASS_OF (e.g., 'Greenwashing' SUBCLASS_OF 'Deceptive Practice')
    
    Output: Return ONLY a valid JSON array of objects: [{{"subject":"...", "predicate":"...", "object":"..."}}]
    """
    
    graph_resp = call_llm_api(prompt_p3)
    try:
        triples = json.loads(graph_resp.replace("```json", "").replace("```", "").strip())
    except:
        triples = []
    
    return triples

# --- EKSEKUSI UTAMA ---

start_time = datetime.now()
print(f"Memulai Pilot Baseline pada file: {INPUT_FILENAME}")
print(f"Waktu mulai: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

# 1. Load Data
try:
    with open(INPUT_FILENAME, "r", encoding="utf-8") as f:
        # Menangani format JSON list biasa (tanpa key "papers")
        # karena dataset baseline biasanya list flat
        file_content = json.load(f)
        if isinstance(file_content, dict) and "papers" in file_content:
            papers_list = file_content["papers"]
        else:
            papers_list = file_content
except FileNotFoundError:
    print(f"ERROR: File {INPUT_FILENAME} tidak ditemukan!")
    print("Pastikan Anda sudah convert CSV baseline ke JSON dulu.")
    papers_list = []

# 2. Cek Resume (Agar aman kalau putus)
all_data = []
processed_titles = set()

try:
    with open(OUTPUT_FILENAME, "r", encoding="utf-8") as f:
        all_data = json.load(f)
        processed_titles = set([p['Title'] for p in all_data])
        print(f"Resume: {len(all_data)} paper sudah selesai sebelumnya.")
except:
    print("Memulai proses dari awal...")

# 3. Hitung paper yang belum diproses
unprocessed_papers = [p for p in papers_list if p.get('Title', 'Unknown') not in processed_titles]
print(f"Total paper di dataset: {len(papers_list)}")
print(f"Paper yang belum diproses: {len(unprocessed_papers)}")

if len(unprocessed_papers) == 0:
    print("\nSemua paper sudah selesai diproses!")
    exit(0)

# 4. Loop Processing
print(f"\nMemulai pemrosesan...\n")

for i, paper in enumerate(unprocessed_papers):
    title = paper.get('Title', 'Unknown')
    current_num = len(all_data) + 1
    
    print(f"[{current_num}] Processing: {title[:60]}...")
    
    try:
        # A. Siapkan Teks
        text_ready = prepare_text_for_llm(paper)
        
        # B. Jalankan Pipeline
        triples = run_llm_pipeline(title, text_ready)
        
        # C. Simpan
        res = {
            "paper_id": current_num - 1,
            "title": title,
            "extracted_knowledge": triples
        }
        all_data.append(res)
        processed_titles.add(title)
        
        # D. Autosave setiap 10 paper
        if len(all_data) % 10 == 0:
            with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
                json.dump(all_data, f, indent=2, ensure_ascii=False)
            elapsed = datetime.now() - start_time
            elapsed_str = str(elapsed).split('.')[0]  # Remove microseconds
            print(f"  >> [AUTOSAVE] {len(all_data)} paper tersimpan. Waktu berjalan: {elapsed_str}")
                
        time.sleep(5) # Jeda agar tidak kena rate limit
        
    except Exception as e:
        print(f"Error pada paper {i}: {e}")

# Final Save
with open(OUTPUT_FILENAME, "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)

end_time = datetime.now()
total_duration = end_time - start_time
total_duration_str = str(total_duration).split('.')[0]

print("\n" + "="*70)
print(f"SELESAI! Total {len(all_data)} paper berhasil diproses.")
print(f"File output: {OUTPUT_FILENAME}")
print(f"Waktu mulai: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Waktu selesai: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total durasi: {total_duration_str}")
print("="*70)