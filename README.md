# ChefGranite 🍽️
### Document Q&A RAG Recipe Generator — IBM watsonx.ai (Granite)

> **Problem Statement No. 8** · Retrieval-Augmented Generation · IBM Cloud

ChefGranite is an end-to-end RAG application that lets you chat with a recipe knowledge base. Upload any cookbook (PDF/TXT), set your dietary constraints, and ask ChefGranite to generate a fully adapted recipe — powered by **IBM Granite** (`ibm/granite-3-8b-instruct`) running on **IBM watsonx.ai**.

---

## Architecture

```
User Query + Constraints
        │
        ▼
  FAISS Vector Store  ◄──  data/recipes_kb.md  +  uploaded PDFs/TXT
  (HuggingFace MiniLM)
        │
        ▼ (top-4 retrieved chunks)
  IBM watsonx.ai Granite
  (ibm/granite-3-8b-instruct)
        │
        ▼
  Structured JSON Response
        │
        ▼
  Streamlit Dashboard
  ┌─────────────────┬──────────────────┐
  │  Adapted Recipe │  Substitutions   │
  ├─────────────────┼──────────────────┤
  │  Nutritional    │  Shopping List   │
  │  Facts          │  (checkboxes)    │
  └─────────────────┴──────────────────┘
```

---

## Quick Start (3 steps)

### Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Python 3.10+ recommended.**  
> On first run, the HuggingFace `all-MiniLM-L6-v2` embedding model (~90 MB) will be downloaded automatically and cached locally. No GPU or cloud vector DB required.

---

### Step 2 — Add IBM Cloud credentials

Copy the example file and fill in your credentials:

```bash
cp .env.example .env
```

Edit `.env`:

```
WATSONX_APIKEY=your_ibm_cloud_api_key_here
WATSONX_PROJECT_ID=your_watsonx_project_id_here
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

**How to get credentials:**
1. Log in to [IBM Cloud](https://cloud.ibm.com)
2. Navigate to **watsonx.ai** → your project → **Manage** → **Access (API key)**
3. Copy your **Project ID** from the project overview page
4. The URL depends on your region (`us-south`, `eu-de`, `jp-tok`, etc.)

---

### Step 3 — Run the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501** in your browser.

> On first launch, click **"Rebuild Knowledge Base"** in the sidebar to index the built-in recipes. This takes ~10 seconds.

---

## Features

| Feature | Detail |
|---|---|
| 📚 Knowledge Base | 6 built-in recipes across diverse cuisines + user-uploaded PDFs/TXT |
| 🔍 RAG Retrieval | Local FAISS + HuggingFace MiniLM embeddings (fully offline) |
| 🤖 LLM | IBM Granite `ibm/granite-3-8b-instruct` via watsonx.ai |
| 🥗 Dietary Filters | Vegan, Vegetarian, Gluten-Free, Dairy-Free, Sugar-Free, Nut-Free, Low-Carb |
| ⏱️ Time Constraints | Slider (10–120 min) to cap total cooking time |
| 🛒 Shopping List | Auto-generated from missing ingredients with checkboxes |
| 🔄 Substitutions | Explicit ingredient swap table with reasoning |
| 📊 Nutrition | Calories, protein, carbs, fat per serving |

---

## Project Structure

```
chefgranite/
├── app.py                  # Streamlit UI
├── rag_engine.py           # RAG logic (FAISS + watsonx.ai)
├── requirements.txt        # Python dependencies
├── .env.example            # Credentials template
├── .env                    # Your credentials (git-ignored)
├── README.md
└── data/
    └── recipes_kb.md       # Built-in recipe knowledge base
```

---

## Demo Prompts

- **"Make a sugar-free chocolate cake"** — adapts the Classic Chocolate Cake, substituting sugar with stevia/erythritol
- **"Quick 20-minute vegan dinner"** — adapts Chickpea Spinach Curry or Stir-Fry Noodles within time constraint
- **"High-protein meal with chicken and spinach"** — adapts Grilled Lemon Herb Chicken with spinach additions

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `401 Unauthorized` | Check your `WATSONX_APIKEY` is valid and not expired |
| `Model not found` | Ensure your project has access to `ibm/granite-3-8b-instruct` |
| `No documents loaded` | Click "Rebuild Knowledge Base" in the sidebar |
| JSON parse error | Model response shown in raw form; try regenerating |

---

## License

MIT · IBM Granite models subject to [IBM watsonx.ai Terms of Use](https://www.ibm.com/legal/terms-of-use)
