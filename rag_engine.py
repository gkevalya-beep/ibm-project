"""
rag_engine.py — ChefGranite RAG Engine
Loads recipes from data/, builds a FAISS vector store with HuggingFace embeddings,
and uses IBM watsonx.ai (Granite) to generate adapted recipes grounded in retrieved context.
"""

import os
import re
import json
import glob as glob_module
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from ibm_watsonx_ai import APIClient, Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).parent / "data"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GRANITE_MODEL = "ibm/granite-3-8b-instruct"


# ---------------------------------------------------------------------------
# Document loading helpers
# ---------------------------------------------------------------------------

def _load_file(filepath: str):
    """Return a list of LangChain Document objects for a given file path."""
    ext = Path(filepath).suffix.lower()
    try:
        if ext == ".pdf":
            loader = PyPDFLoader(filepath)
        elif ext == ".md":
            loader = TextLoader(filepath, encoding="utf-8")
        else:
            # .txt and anything else treated as plain text
            loader = TextLoader(filepath, encoding="utf-8")
        return loader.load()
    except Exception as e:
        print(f"[rag_engine] Warning: could not load {filepath}: {e}")
        return []


def load_documents(extra_files: list[str] | None = None) -> list:
    """
    Load all .md, .txt, and .pdf files from data/ plus any extra uploaded files.
    Returns a flat list of LangChain Document objects.
    """
    documents = []

    # Built-in knowledge base files
    for pattern in ("*.md", "*.txt", "*.pdf"):
        for filepath in glob_module.glob(str(DATA_DIR / pattern)):
            docs = _load_file(filepath)
            documents.extend(docs)

    # User-uploaded extra files (absolute paths)
    if extra_files:
        for filepath in extra_files:
            docs = _load_file(filepath)
            documents.extend(docs)

    if not documents:
        raise ValueError(
            "No documents loaded. Make sure data/recipes_kb.md exists "
            "or upload recipe files via the sidebar."
        )

    return documents


# ---------------------------------------------------------------------------
# Vector store builder
# ---------------------------------------------------------------------------

def build_vector_store(extra_files: list[str] | None = None) -> FAISS:
    """
    Load documents, split into chunks, embed with HuggingFace MiniLM,
    and return an in-memory FAISS vector store.
    """
    documents = load_documents(extra_files)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n## ", "\n### ", "\n\n", "\n", " "],
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


# ---------------------------------------------------------------------------
# IBM watsonx.ai client helper
# ---------------------------------------------------------------------------

def _get_watsonx_model(api_key: str, project_id: str, url: str) -> ModelInference:
    """Instantiate and return an IBM Granite ModelInference client."""
    credentials = Credentials(url=url, api_key=api_key)
    client = APIClient(credentials=credentials, project_id=project_id)

    params = {
        GenParams.MAX_NEW_TOKENS: 2048,
        GenParams.TEMPERATURE: 0.3,
        GenParams.TOP_P: 0.9,
        GenParams.REPETITION_PENALTY: 1.1,
        GenParams.STOP_SEQUENCES: ["```"],
    }

    model = ModelInference(
        model_id=GRANITE_MODEL,
        api_client=client,
        params=params,
        project_id=project_id,
    )
    return model


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    context: str,
    user_query: str,
    available_ingredients: str,
    dietary_restrictions: str,
    max_cook_time: int,
) -> str:
    """
    Build the structured prompt for IBM Granite, instructing it to return
    a valid JSON object grounded in the retrieved recipe context.
    """
    return f"""You are ChefGranite, an expert culinary AI assistant. Your task is to adapt a recipe from the provided CONTEXT to match the user's specific needs.

CONTEXT (retrieved recipes from the knowledge base):
\"\"\"
{context}
\"\"\"

USER REQUEST: {user_query}

CONSTRAINTS:
- Available ingredients the user already has: {available_ingredients if available_ingredients.strip() else "Not specified — assume a well-stocked kitchen"}
- Dietary restrictions: {dietary_restrictions}
- Maximum cooking time: {max_cook_time} minutes

INSTRUCTIONS:
1. Base your answer STRICTLY on the CONTEXT above. Do not invent recipes not present in the context.
2. Adapt the closest matching recipe to fit ALL constraints listed above.
3. If an ingredient conflicts with dietary restrictions or is unavailable, replace it with a suitable alternative and document the substitution.
4. If the total time exceeds max_cook_time, simplify the recipe to fit within the limit.
5. Calculate or estimate nutritional facts for the adapted recipe.
6. Identify any ingredients in the adapted recipe that are NOT in the user's available_ingredients list as the shopping_list.
7. Return ONLY a single valid JSON object — no markdown fences, no explanations, no preamble.

REQUIRED JSON SCHEMA (return exactly these keys):
{{
  "recipe_title": "string — name of the adapted recipe",
  "prep_and_cook_time": "string — e.g. '15 min prep + 20 min cook = 35 min total'",
  "adapted_ingredients": ["list of strings — full ingredient list with measurements"],
  "substitutions_made": [
    {{"original": "string", "replacement": "string", "reason": "string"}}
  ],
  "step_by_step_instructions": ["1. Step one", "2. Step two", "..."],
  "nutritional_facts": {{
    "calories": "string — e.g. '310 kcal'",
    "protein": "string — e.g. '12g'",
    "carbs": "string — e.g. '45g'",
    "fat": "string — e.g. '8g'"
  }},
  "shopping_list": ["list of missing ingredients the user needs to buy"],
  "source_recipe_used": "string — exact title of the retrieved recipe this is based on"
}}

JSON RESPONSE:
"""


# ---------------------------------------------------------------------------
# Main retrieval + generation function
# ---------------------------------------------------------------------------

def generate_adapted_recipe(
    vector_store: FAISS,
    user_query: str,
    available_ingredients: str,
    dietary_restrictions: str,
    max_cook_time: int,
    api_key: str,
    project_id: str,
    watsonx_url: str,
) -> dict:
    """
    Retrieve relevant recipe chunks from FAISS, then call IBM Granite to
    generate an adapted recipe JSON grounded in that context.

    Returns a dict with the parsed recipe fields plus a raw_response key.
    Raises RuntimeError on unrecoverable failures.
    """
    # --- Step 1: Build enriched query for retrieval ---
    retrieval_query = f"{user_query} dietary: {dietary_restrictions} time: {max_cook_time} min"
    docs = vector_store.similarity_search(retrieval_query, k=4)
    context = "\n\n---\n\n".join(doc.page_content for doc in docs)

    # --- Step 2: Build prompt ---
    prompt = _build_prompt(
        context=context,
        user_query=user_query,
        available_ingredients=available_ingredients,
        dietary_restrictions=dietary_restrictions,
        max_cook_time=max_cook_time,
    )

    # --- Step 3: Call IBM watsonx.ai ---
    try:
        model = _get_watsonx_model(api_key, project_id, watsonx_url)
        response = model.generate_text(prompt=prompt)
    except Exception as e:
        raise RuntimeError(f"IBM watsonx.ai API call failed: {e}") from e

    raw_text = response.strip() if isinstance(response, str) else str(response).strip()

    # --- Step 4: Parse JSON response ---
    parsed = _safe_parse_json(raw_text)
    parsed["raw_response"] = raw_text
    return parsed


# ---------------------------------------------------------------------------
# Robust JSON parser
# ---------------------------------------------------------------------------

def _safe_parse_json(text: str) -> dict:
    """
    Attempt to extract and parse a JSON object from the model output.
    Falls back to a structured error dict if parsing fails.
    """
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON object between first { and last }
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Try extracting from markdown code fence ```json ... ```
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Return a graceful fallback structure
    return {
        "recipe_title": "Recipe Generation Failed",
        "prep_and_cook_time": "N/A",
        "adapted_ingredients": [],
        "substitutions_made": [],
        "step_by_step_instructions": [
            "The model returned an unexpected response format.",
            "Raw model output has been preserved below for debugging.",
            f"Raw output: {text[:800]}",
        ],
        "nutritional_facts": {
            "calories": "N/A",
            "protein": "N/A",
            "carbs": "N/A",
            "fat": "N/A",
        },
        "shopping_list": [],
        "source_recipe_used": "Unknown",
        "parse_error": True,
    }
