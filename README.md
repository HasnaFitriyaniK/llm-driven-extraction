# LLM-Driven Knowledge Graph Generation

Tugas Akhir - Sistem Rekomendasi Paper menggunakan LLM-Powered Ontology Generation

## Overview

Project ini menghasilkan knowledge graph dari research papers menggunakan Gemma API, berdasarkan metodologi Tupayachi et al. (2024).

## Dataset

- **Dataset Utama**: Computer Science papers (dataset_terbaru_computer_science.json)
- **Dataset Baseline**: Sustainability papers dari Scopus (20241121-scopus.json)

## Pipeline

### Two-Stage Architecture

1. **Stage 1 - Smart Parsing** (`step1_smart_parsing.py`)
   - Ekstraksi text sections (title, abstract, methods, experiments)
   - No API calls (gratis)
   - Output: `hasil_smart_parsing.json`

2. **Stage 2 - Graph Generation** (`step2_graph_extraction.py`)
   - Generate knowledge graph via Gemini API
   - 3-step prompt chain (Extract → Filter → Build Graph)
   - Output: `hasil_graph_final.json`

### Baseline Processing

- **Script**: `prompting2.py`
- **Domain**: Sustainability (GEMET/TPB/SDG-12)
- **Model**: Gemma 3 27B
- **Output**: `hasil_graph_baseline.json`

## Requirements

```bash
pip install pandas google-generativeai
```

## Configuration

Ganti API key di masing-masing script:
```python
genai.configure(api_key='YOUR_API_KEY')
```

## Usage

### Dataset CS (Two-Stage)
```bash
# Step 1: Parse all papers (gratis)
python step1_smart_parsing.py

# Step 2: Extract graph (API calls)
python step2_graph_extraction.py
```

### Dataset Baseline
```bash
python prompting2.py
```

## Features

- ✅ Resume capability (title-based tracking)
- ✅ Auto-save every 10 papers
- ✅ Batch processing (100 papers per run)
- ✅ Time tracking
- ✅ Error handling & retry logic
- ✅ 5-second delay between requests

## Methodology

Mengikuti Tupayachi et al. (2024) - LLM-Powered Ontology Generation:
1. **P1**: Extract glossary dari paper
2. **P2**: Filter & normalize dengan domain standards (CSO/GEMET)
3. **P3**: Build taxonomy (Subject-Predicate-Object triples)

## Output Format

```json
{
  "paper_id": 0,
  "title": "...",
  "extracted_knowledge": [
    {
      "subject": "Deep Learning",
      "predicate": "SUBCLASS_OF",
      "object": "Machine Learning"
    }
  ]
}
```

## License

Tugas Akhir - Universitas
