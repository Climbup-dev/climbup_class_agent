# AI Engine Architecture: The "Single-Shot" Pipeline

Here is the graphical representation of how your newly optimized AI system works behind the scenes.

## Flow Diagram

```mermaid
graph TD
    %% Styling
    classDef user fill:#3498db,stroke:#2980b9,stroke-width:2px,color:#fff
    classDef gemini fill:#f1c40f,stroke:#f39c12,stroke-width:2px,color:#333
    classDef faiss fill:#2ecc71,stroke:#27ae60,stroke-width:2px,color:#fff
    classDef flashrank fill:#e74c3c,stroke:#c0392b,stroke-width:2px,color:#fff
    classDef llama fill:#9b59b6,stroke:#8e44ad,stroke-width:2px,color:#fff
    classDef output fill:#34495e,stroke:#2c3e50,stroke-width:2px,color:#fff

    %% Nodes
    User(("🧑‍🎓 Student Input")):::user
    
    subgraph Brain[1. The Intelligence Layer]
        Gemini{"⚡ Gemini Flash Lite\n(Intent & Angle Router)"}:::gemini
    end
    
    subgraph DataMining[2. The PDF Extraction Layer]
        MQ["Multi-Query Expansion\n(Original + 2 Alts)"]:::faiss
        FAISS[("FAISS / Vector DB\n(Scans entire PDF)")]:::faiss
        Dedup["Merge & Deduplicate\n(30+ scattered chunks)"]:::faiss
        FlashRank["🔥 FlashRank\n(Extracts Top 6 Chunks)"]:::flashrank
    end
    
    subgraph Engine[3. The Execution Layer]
        Llama["🧠 Groq Llama-3-70B\n(Formatting & Reasoning)"]:::llama
        GenMode["General Chat Mode\n(No PDF needed)"]:::llama
    end
    
    Output[/"Final JSON Output\n(board_content)"/]:::output

    %% Connections
    User -->|Sends Question| Gemini
    
    Gemini -->|source_needed: 'PDF'| MQ
    Gemini -.->|source_needed: 'General'| GenMode
    Gemini -->|user_angle| Llama
    
    MQ -->|Search 3 Queries| FAISS
    FAISS -->|Returns all matches| Dedup
    Dedup -->|Send unoptimized chunks| FlashRank
    FlashRank -->|Pass Top 6 High-Quality Chunks| Llama
    
    GenMode -.->|Direct Chat| Llama
    
    Llama --> Output
```

## How it works (Step-by-Step):

1. **The Intelligence Layer (Gemini):**
   - As soon as the student asks a question, **Gemini Flash Lite** reads it. 
   - It decides if the question needs the PDF (`source_needed`) and how the student is feeling (`user_angle`).

2. **The Extraction Layer (FAISS + FlashRank):**
   - If the PDF is needed, the system breaks the question into 3 variations.
   - It searches the **Vector DB** using all 3 variations to ensure no scattered data is missed.
   - It combines all the results and gives them to **FlashRank**.
   - FlashRank acts as a strict gatekeeper and only allows the **Top 6 absolute best** chunks to pass through.

3. **The Execution Layer (Llama-3):**
   - **Groq Llama-3-70B** receives the perfect 6 chunks AND the user's emotion instruction.
   - It formats the exact response needed and outputs it in JSON, ready for the UI.
