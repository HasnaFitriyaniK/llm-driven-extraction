import pandas as pd
import json
import time
import google.generativeai as genai 

# Konfigurasi API Key Anda
genai.configure(api_key='AIzaSyAJrQushtLUshxyT8IFqX7ihARimBj-LeQ')

def call_llm_api(prompt):
    """Fungsi untuk memanggil API Google Generative AI"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        # Setting safety settings agar tidak terlalu sensitif memblokir teks ilmiah
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"API Error: {e}")
        return "[]"

def prepare_text_for_llm(paper_json):
    """
    [BARU] Fungsi Smart Parsing.
    Mengambil Title, Abstract, dan menyaring bab Method/Experiment dari full_content.
    """
    title = paper_json.get('title', '')
    abstract = paper_json.get('abstract', '')
    full_content = paper_json.get('full_content', {})
    
    # 1. Mulai dengan Abstrak (Wajib)
    relevant_text = f"Title: {title}\nAbstract: {abstract}\n\n"
    
    # 2. Ambil Bagian Metode (Cari key yang mengandung kata kunci)
    # Kita cari bab yang mengandung kata 'method', 'approach', 'model', 'proposed'
    method_keywords = ['method', 'approach', 'model', 'proposed', 'system design', 'architecture']
    method_text = ""
    
    # 3. Ambil Bagian Eksperimen (Untuk Dataset & Metric)
    # Kita cari bab yang mengandung 'experiment', 'evaluation', 'result', 'dataset'
    exp_keywords = ['experiment', 'evaluation', 'result', 'dataset', 'performance']
    exp_text = ""

    if isinstance(full_content, dict):
        for key, text in full_content.items():
            key_lower = key.lower()
            
            # Cek Bab Metode
            if any(k in key_lower for k in method_keywords):
                # Ambil 2000 karakter pertama per bab agar hemat token tapi cukup konteks
                method_text += f"--- SECTION: {key} ---\n{text[:2500]}\n"
            
            # Cek Bab Eksperimen
            elif any(k in key_lower for k in exp_keywords):
                exp_text += f"--- SECTION: {key} ---\n{text[:2500]}\n"
    
    # Gabungkan semua (Prioritas: Abstract -> Method -> Experiment)
    final_input = relevant_text + "\n=== METHODOLOGY DETAILS ===\n" + method_text + "\n=== EXPERIMENTAL RESULTS ===\n" + exp_text
    return final_input

def run_llm_pipeline(title, processed_text):
    """
    Fungsi Pipeline Utama (Updated Input: processed_text)
    """
    
    # [cite_start]--- STEP P1: Extract Glossary (UPDATED PROMPT) [cite: 886] ---
    # Prompt diperbarui untuk menangani full text dan entitas spesifik
    prompt_p1 = f"""
    Role: You are a Computer Science researcher assistant.
    Task: Extract specific technical entities from the provided scientific text.
    
    INPUT TEXT:
    {processed_text}
    
    INSTRUCTIONS:
    Extract terms focusing on these specific categories:
    1. Method/Model Names (e.g., 'CLFF-NER', 'Multilingual BERT', 'GNN', 'Apriori').
    2. Dataset Names (e.g., 'CTFCDataSet', 'Weibo', 'Resume').
    3. Metrics (e.g., 'F1-score', 'Precision', 'Recall').
    4. Task/Problem (e.g., 'Named Entity Recognition', 'Recommendation').
    
    Output: Return ONLY a valid JSON array of strings. Example: ["CLFF-NER", "F1-score", "CTFCDataSet"]
    """
    
    glossary_raw_response = call_llm_api(prompt_p1)
    
    # Parsing JSON P1
    try:
        glossary_raw_response = glossary_raw_response.replace("```json", "").replace("```", "").strip()
        glossary_raw = json.loads(glossary_raw_response)
    except:
        glossary_raw = [] # Fallback
    
    # Jika hasil kosong, coba return kosong biar gak lanjut buang token
    if not glossary_raw:
        return []

    # [cite_start]--- STEP P2: Relevant Words Verification (Filter CSO) [cite: 892] ---
    prompt_p2 = f"""
    Role: You are a research assistant expert in Computer Science Ontology (CSO).
    I have extracted a raw list of terms: {glossary_raw}.
    
    TASK: Filter and normalize this list using the **CSO (Computer Science Ontology)** standards.
    1. KEEP terms that are valid CS concepts, specific algorithms, datasets, or metrics.
    2. REMOVE generic academic words (e.g., 'study', 'paper', 'significant improvement').
    3. CONVERT verb phrases to Canonical Noun Phrases (e.g., "using bert" -> "BERT").
    
    Output: Return ONLY a valid JSON array of strings.
    """
    
    glossary_clean_response = call_llm_api(prompt_p2)
    
    # Parsing JSON P2
    try:
        glossary_clean_response = glossary_clean_response.replace("```json", "").replace("```", "").strip()
        glossary_clean = json.loads(glossary_clean_response)
    except:
        glossary_clean = glossary_raw 

    # [cite_start]--- STEP P3: Build Taxonomy & Relations (Knowledge Graph) [cite: 896] ---
    prompt_p3 = f"""
    Role: Knowledge Graph Architect.
    Terms: {glossary_clean}
    Paper Title: "{title}"
    
    TASK: Map these terms into a Knowledge Graph using guidance from CSO.
    
    Use these predicates:
    - SUBCLASS_OF (Hierarchy, e.g., 'BERT' SUBCLASS_OF 'Language Model')
    - USES_METHOD (e.g., 'This Paper' USES_METHOD 'CLFF-NER')
    - EVALUATED_ON (e.g., 'CLFF-NER' EVALUATED_ON 'Weibo Dataset')
    - MEASURED_BY (e.g., 'Performance' MEASURED_BY 'F1-Score')
    
    Constraints:
    1. Subject and Object MUST be Noun Phrases.
    2. Try to capture the specific contribution of the paper.
    
    Output: Return ONLY a valid JSON array of objects: [{{"subject":"...", "predicate":"...", "object":"..."}}]
    """
    
    relationships_response = call_llm_api(prompt_p3)
    
    # Parsing JSON P3
    try:
        relationships_response = relationships_response.replace("```json", "").replace("```", "").strip()
        relationships = json.loads(relationships_response)
    except:
        relationships = []
    
    return relationships

# --- EKSEKUSI UTAMA (Main Loop Updated) ---

# 1. Baca Data JSON Scrapping Anda
# Pastikan nama file sesuai dengan file hasil scrapping teman Anda
input_filename = "dataset_scrapping_full.json" 
output_filename = "hasil_graph_final.json"

try:
    with open(input_filename, "r", encoding="utf-8") as f:
        data = json.load(f)
        # Handle jika formatnya list of dictionaries langsung atau dibungkus key "papers"
        if isinstance(data, dict) and "papers" in data:
            papers_list = data["papers"]
        else:
            papers_list = data
except FileNotFoundError:
    print(f"File {input_filename} tidak ditemukan. Pastikan file JSON ada di folder yang sama.")
    papers_list = []

all_graph_data = []
processed_ids = set()

# Cek resume file lama
try:
    with open(output_filename, "r", encoding="utf-8") as f:
        existing_data = json.load(f)
        all_graph_data = existing_data
        processed_ids = set([item.get("title") for item in all_graph_data]) # Gunakan Title sebagai ID sementara
        print(f"Resume: Sudah ada {len(all_graph_data)} paper.")
except FileNotFoundError:
    print("Memulai proses baru...")

# 2. Loop Processing
print(f"Total paper di input: {len(papers_list)}")

for i, paper in enumerate(papers_list):
    title = paper.get('title', 'Unknown Title')
    
    # Skip jika sudah diproses
    if title in processed_ids:
        continue
        
    print(f"Processing [{i+1}/{len(papers_list)}]: {title[:50]}...")
    
    try:
        # [PERUBAHAN PENTING]
        # Panggil fungsi prepare_text_for_llm sebelum masuk pipeline
        smart_text = prepare_text_for_llm(paper)
        
        # Jalankan pipeline dengan teks yang sudah disiapkan
        triples = run_llm_pipeline(title, smart_text)
        
        # Simpan hasil
        result_entry = {
            "paper_id": i,
            "title": title,
            "extracted_knowledge": triples,
            # Opsional: Simpan keywords asli jika ada
            "original_keywords": paper.get('keywords', '')
        }
        
        all_graph_data.append(result_entry)
        
        # Autosave per 5 paper
        if len(all_graph_data) % 5 == 0:
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(all_graph_data, f, indent=2, ensure_ascii=False)
            print(">> Autosaved.")
            
        time.sleep(3) # Jeda sopan untuk API
        
    except Exception as e:
        print(f"Error processing {title}: {e}")

# Save Final
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(all_graph_data, f, indent=2, ensure_ascii=False)
    
print("Selesai! Data siap untuk Neo4j.")