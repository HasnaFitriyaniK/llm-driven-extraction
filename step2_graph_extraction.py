import json
import time
import google.generativeai as genai 

# Konfigurasi API Key
# genai.configure(api_key='AIzaSyAJrQushtLUshxyT8IFqX7ihARimBj-LeQ')
genai.configure(api_key='AIzaSyAP93sJHWkz9pLFeaDUO8wlKZFT2hT4id4')


def call_llm_api(prompt):
    """Fungsi untuk memanggil API Google Generative AI"""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"  API Error: {e}")
        return "[]"

def run_llm_pipeline(title, processed_text):
    """
    Pipeline LLM untuk ekstraksi Knowledge Graph
    """
    
    # --- STEP P1: Extract Glossary ---
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
    
    try:
        glossary_raw_response = glossary_raw_response.replace("```json", "").replace("```", "").strip()
        glossary_raw = json.loads(glossary_raw_response)
    except:
        glossary_raw = []
    
    if not glossary_raw:
        return []

    # --- STEP P2: Relevant Words Verification ---
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
    
    try:
        glossary_clean_response = glossary_clean_response.replace("```json", "").replace("```", "").strip()
        glossary_clean = json.loads(glossary_clean_response)
    except:
        glossary_clean = glossary_raw 

    # --- STEP P3: Build Knowledge Graph ---
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
    
    try:
        relationships_response = relationships_response.replace("```json", "").replace("```", "").strip()
        relationships = json.loads(relationships_response)
    except:
        relationships = []
    
    return relationships

# ============================================================================
# MAIN: KNOWLEDGE GRAPH EXTRACTION
# ============================================================================

input_filename = "hasil_smart_parsing.json"
output_filename = "hasil_graph_final.json"
BATCH_SIZE = 100  # Proses 100 paper per run

print("="*70)
print("STEP 2: KNOWLEDGE GRAPH EXTRACTION (Menggunakan LLM API)")
print("="*70)

# Baca hasil smart parsing
try:
    with open(input_filename, "r", encoding="utf-8") as f:
        smartparsing_data = json.load(f)
except FileNotFoundError:
    print(f"ERROR: File {input_filename} tidak ditemukan!")
    print("Jalankan step1_smart_parsing.py terlebih dahulu!")
    exit(1)

print(f"Total paper di smart parsing: {len(smartparsing_data)}")

# Cek resume - lihat paper mana saja yang sudah diproses
all_graph_data = []
processed_titles = set()

try:
    with open(output_filename, "r", encoding="utf-8") as f:
        all_graph_data = json.load(f)
        processed_titles = set([item.get("title") for item in all_graph_data])
        print(f"Resume: Sudah ada {len(all_graph_data)} paper yang di-extract.")
except FileNotFoundError:
    print("Memulai graph extraction dari awal...")

# Hitung berapa paper yang belum diproses
unprocessed_papers = [p for p in smartparsing_data if p.get('title') not in processed_titles]
print(f"Paper yang belum diproses: {len(unprocessed_papers)}")

# Ambil 100 paper pertama yang belum diproses
papers_to_process = unprocessed_papers[:BATCH_SIZE]

if len(papers_to_process) == 0:
    print("\nSemua paper sudah diproses!")
    print(f"Total: {len(all_graph_data)} paper")
    exit(0)

print(f"Akan memproses {len(papers_to_process)} paper pada run ini.")
print(f"Range: Paper ke-{len(all_graph_data) + 1} sampai ke-{len(all_graph_data) + len(papers_to_process)}")

# Loop Graph Extraction
print(f"\nMemulai proses extraction...\n")

for i, parsed_paper in enumerate(papers_to_process):
    title = parsed_paper.get('title', 'Unknown Title')
    smart_text = parsed_paper.get('smart_parsed_text', '')
    
    current_num = len(all_graph_data) + 1
    print(f"[{current_num}] Extracting: {title[:60]}...")
    
    try:
        # Jalankan pipeline LLM
        triples = run_llm_pipeline(title, smart_text)
        
        result_entry = {
            "paper_id": parsed_paper.get('paper_id', i),
            "title": title,
            "extracted_knowledge": triples
        }
        
        all_graph_data.append(result_entry)
        processed_titles.add(title)
        
        # Autosave setiap 10 paper
        if len(all_graph_data) % 10 == 0:
            with open(output_filename, "w", encoding="utf-8") as f:
                json.dump(all_graph_data, f, indent=2, ensure_ascii=False)
            print(f"  >> [AUTOSAVE] {len(all_graph_data)} paper di-save.")
        
        time.sleep(5)  # Delay 5 detik per request
        
    except Exception as e:
        print(f"  ERROR: {e}")

# Save final
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(all_graph_data, f, indent=2, ensure_ascii=False)

print("\n" + "="*70)
print(f"SELESAI! {len(all_graph_data)} paper berhasil di-extract (total kumulatif).")
print(f"File output: {output_filename}")

remaining = len(unprocessed_papers) - len(papers_to_process)
if remaining > 0:
    print(f"\nMasih ada {remaining} paper yang belum diproses.")
    print("Jalankan script ini lagi untuk melanjutkan 100 paper berikutnya.")
else:
    print("\nSemua paper telah selesai diproses!")
print("="*70)
