"""
app.py — ChefGranite Streamlit Dashboard
Document Q&A RAG Recipe Generator powered by IBM watsonx.ai (Granite)
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from rag_engine import build_vector_store, generate_adapted_recipe

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ChefGranite: Document Q&A RAG Recipe Generator",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

# ---------------------------------------------------------------------------
# Custom CSS — clean, minimal, modern
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ---- Global ---- */
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    h1 { font-size: 2rem !important; font-weight: 700; }
    h2 { font-size: 1.3rem !important; font-weight: 600; }
    h3 { font-size: 1.1rem !important; font-weight: 600; }

    /* ---- Recipe cards ---- */
    .card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
        height: 100%;
    }
    .card-title {
        font-size: 1rem;
        font-weight: 700;
        color: #1f2328;
        margin-bottom: 0.6rem;
        border-bottom: 2px solid #3b82d4;
        padding-bottom: 0.4rem;
    }

    /* ---- Metric pill ---- */
    .metric-pill {
        display: inline-block;
        background: #3b82d4;
        color: #ffffff;
        border-radius: 20px;
        padding: 0.25rem 0.9rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem 0.2rem;
    }

    /* ---- Sub-section label ---- */
    .sub-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #57606a;
        margin-bottom: 0.3rem;
        margin-top: 0.8rem;
    }

    /* ---- Demo prompt button ---- */
    div[data-testid="column"] .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-size: 0.82rem;
        background: #f7f8fa;
        border: 1px solid #3b82d4;
        color: #3b82d4;
        font-weight: 600;
        padding: 0.5rem 0.6rem;
        transition: all 0.15s;
    }
    div[data-testid="column"] .stButton > button:hover {
        background: #3b82d4;
        color: #ffffff;
    }

    /* ---- Source badge ---- */
    .source-badge {
        background: #eef4ff;
        border: 1px solid #3b82d4;
        border-radius: 6px;
        padding: 0.3rem 0.7rem;
        font-size: 0.8rem;
        color: #3b82d4;
        display: inline-block;
        margin-top: 0.4rem;
    }

    /* ---- Error / warning box ---- */
    .err-box {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "kb_built" not in st.session_state:
    st.session_state.kb_built = False
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""
if "last_result" not in st.session_state:
    st.session_state.last_result = None


# ---------------------------------------------------------------------------
# Sidebar — credentials + file upload + constraint filters
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/IBM_logo.svg/200px-IBM_logo.svg.png",
        width=90,
    )
    st.title("⚙️ Configuration")

    st.markdown("#### IBM watsonx.ai Credentials")
    api_key = st.text_input(
        "WATSONX_APIKEY",
        value=os.getenv("WATSONX_APIKEY", ""),
        type="password",
        placeholder="Your IBM Cloud API Key",
    )
    project_id = st.text_input(
        "WATSONX_PROJECT_ID",
        value=os.getenv("WATSONX_PROJECT_ID", ""),
        placeholder="Your watsonx.ai Project ID",
    )
    watsonx_url = st.text_input(
        "WATSONX_URL",
        value=os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
        placeholder="https://us-south.ml.cloud.ibm.com",
    )

    st.divider()

    st.markdown("#### 📂 Upload Additional Recipes")
    uploaded_files = st.file_uploader(
        "Upload .pdf or .txt cookbooks",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Upload additional recipe files to expand the knowledge base.",
    )

    rebuild_btn = st.button("🔄 Rebuild Knowledge Base", use_container_width=True)

    if rebuild_btn or not st.session_state.kb_built:
        extra_paths = []
        if uploaded_files:
            for uf in uploaded_files:
                suffix = "." + uf.name.rsplit(".", 1)[-1]
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(uf.read())
                tmp.flush()
                extra_paths.append(tmp.name)

        with st.spinner("Building knowledge base… (first run downloads the embedding model)"):
            try:
                st.session_state.vector_store = build_vector_store(extra_paths or None)
                st.session_state.kb_built = True
                st.success("✅ Knowledge base ready!")
            except Exception as e:
                st.error(f"❌ Failed to build knowledge base: {e}")

    st.divider()

    st.markdown("#### 🧑‍🍳 Recipe Constraints")
    available_ingredients = st.text_area(
        "Available Ingredients",
        placeholder="e.g. eggs, flour, butter, cocoa powder, milk, olive oil",
        height=90,
        help="Comma-separated list of ingredients you have on hand.",
    )
    dietary_restriction = st.selectbox(
        "Dietary Restriction",
        options=[
            "None",
            "Vegan",
            "Vegetarian",
            "Gluten-Free",
            "Dairy-Free",
            "Sugar-Free",
            "Nut-Free",
            "Low-Carb / Keto",
        ],
    )
    max_cook_time = st.slider(
        "Max Cooking Time (minutes)",
        min_value=10,
        max_value=120,
        value=60,
        step=5,
    )


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    "# 🍽️ ChefGranite: Document Q&A RAG Recipe Generator",
)
st.markdown(
    "<span style='color:#57606a;font-size:0.95rem;'>"
    "Powered by <strong>IBM watsonx.ai</strong> (Granite) · "
    "Retrieval-Augmented Generation over your recipe knowledge base"
    "</span>",
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Demo prompt buttons
# ---------------------------------------------------------------------------
st.markdown("**✨ Try a demo prompt:**")
demo_col1, demo_col2, demo_col3 = st.columns(3)

demo_query = None
with demo_col1:
    if st.button("🍫 Sugar-Free Chocolate Cake"):
        demo_query = "Make a sugar-free chocolate cake"
with demo_col2:
    if st.button("🥦 Quick 20-Min Vegan Dinner"):
        demo_query = "Quick 20-minute vegan dinner"
with demo_col3:
    if st.button("💪 High-Protein Meal with Chicken & Spinach"):
        demo_query = "High-protein meal with chicken and spinach"

if demo_query:
    st.session_state.chat_input = demo_query

# ---------------------------------------------------------------------------
# Main query input
# ---------------------------------------------------------------------------
st.markdown("#### 💬 Ask ChefGranite")
user_query = st.text_input(
    "What would you like to cook?",
    value=st.session_state.chat_input,
    placeholder="e.g. I want a vegan pasta dish under 30 minutes",
    key="query_input",
)

generate_col, _ = st.columns([1, 3])
with generate_col:
    generate_btn = st.button("🔍 Generate Adapted Recipe", use_container_width=True, type="primary")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------
def _check_credentials():
    missing = []
    if not api_key:
        missing.append("WATSONX_APIKEY")
    if not project_id:
        missing.append("WATSONX_PROJECT_ID")
    if not watsonx_url:
        missing.append("WATSONX_URL")
    return missing


# ---------------------------------------------------------------------------
# Generate recipe on button click
# ---------------------------------------------------------------------------
if generate_btn:
    # Sync query input state
    st.session_state.chat_input = user_query

    if not user_query.strip():
        st.warning("⚠️ Please enter a recipe request before generating.")
    elif not st.session_state.kb_built or st.session_state.vector_store is None:
        st.error("❌ Knowledge base not ready. Use the sidebar to rebuild it.")
    else:
        missing_creds = _check_credentials()
        if missing_creds:
            st.error(
                f"❌ Missing credentials: **{', '.join(missing_creds)}**. "
                "Enter them in the sidebar or add them to your `.env` file."
            )
        else:
            with st.spinner("🔎 Retrieving relevant recipes & generating adapted version…"):
                try:
                    result = generate_adapted_recipe(
                        vector_store=st.session_state.vector_store,
                        user_query=user_query.strip(),
                        available_ingredients=available_ingredients,
                        dietary_restrictions=dietary_restriction,
                        max_cook_time=max_cook_time,
                        api_key=api_key,
                        project_id=project_id,
                        watsonx_url=watsonx_url,
                    )
                    st.session_state.last_result = result
                except RuntimeError as e:
                    st.error(f"❌ {e}")
                    st.session_state.last_result = None
                except Exception as e:
                    st.error(f"❌ Unexpected error: {e}")
                    st.session_state.last_result = None


# ---------------------------------------------------------------------------
# Results dashboard — 4 cards
# ---------------------------------------------------------------------------
result = st.session_state.last_result

if result:
    st.divider()

    # Parse error banner
    if result.get("parse_error"):
        st.markdown(
            '<div class="err-box">⚠️ The model response could not be parsed as valid JSON. '
            "The fallback output is shown below. Try regenerating or simplifying your query.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("")

    # Source badge
    source = result.get("source_recipe_used", "Unknown")
    time_str = result.get("prep_and_cook_time", "N/A")
    st.markdown(
        f"<span class='source-badge'>📖 Based on: <strong>{source}</strong></span>"
        f"&nbsp;&nbsp;"
        f"<span class='source-badge'>⏱️ {time_str}</span>",
        unsafe_allow_html=True,
    )
    st.markdown("### " + result.get("recipe_title", "Adapted Recipe"))
    st.markdown("")

    # --- Row 1: Recipe + Substitutions ---
    col_recipe, col_subs = st.columns([3, 2], gap="medium")

    with col_recipe:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📋 Adapted Recipe & Instructions</div>', unsafe_allow_html=True)

        # Ingredients
        st.markdown('<div class="sub-label">Ingredients</div>', unsafe_allow_html=True)
        ingredients = result.get("adapted_ingredients", [])
        if ingredients:
            for ing in ingredients:
                st.markdown(f"• {ing}")
        else:
            st.markdown("_No ingredients listed._")

        # Instructions
        st.markdown('<div class="sub-label">Step-by-Step Instructions</div>', unsafe_allow_html=True)
        instructions = result.get("step_by_step_instructions", [])
        if instructions:
            for step in instructions:
                st.markdown(f"{step}")
        else:
            st.markdown("_No instructions available._")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_subs:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🔄 Smart Substitutions</div>', unsafe_allow_html=True)

        substitutions = result.get("substitutions_made", [])
        if substitutions:
            for sub in substitutions:
                orig = sub.get("original", "")
                repl = sub.get("replacement", "")
                reason = sub.get("reason", "")
                st.markdown(
                    f"**❌ {orig}** → **✅ {repl}**  \n"
                    f"<span style='color:#57606a;font-size:0.82rem;'>_{reason}_</span>",
                    unsafe_allow_html=True,
                )
                st.markdown("---")
        else:
            st.markdown(
                "✅ **No substitutions needed!**  \n"
                "<span style='color:#57606a;font-size:0.85rem;'>The recipe fits your constraints as-is.</span>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    # --- Row 2: Nutritional Facts + Shopping List ---
    col_nutrition, col_shopping = st.columns([2, 3], gap="medium")

    with col_nutrition:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🥗 Nutritional Facts (per serving)</div>', unsafe_allow_html=True)

        nf = result.get("nutritional_facts", {})
        calories = nf.get("calories", "N/A")
        protein = nf.get("protein", "N/A")
        carbs = nf.get("carbs", "N/A")
        fat = nf.get("fat", "N/A")

        # Display as big metric cards
        m1, m2 = st.columns(2)
        with m1:
            st.metric("🔥 Calories", calories)
            st.metric("🥩 Protein", protein)
        with m2:
            st.metric("🌾 Carbs", carbs)
            st.metric("🧈 Fat", fat)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_shopping:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🛒 Shopping List</div>', unsafe_allow_html=True)

        shopping_list = result.get("shopping_list", [])
        if shopping_list:
            st.markdown(
                "<span style='color:#57606a;font-size:0.82rem;'>"
                "Items you'll need to pick up — not in your available ingredients:"
                "</span>",
                unsafe_allow_html=True,
            )
            st.markdown("")
            for i, item in enumerate(shopping_list):
                # Render as interactive checkboxes
                st.checkbox(item, key=f"shop_{i}_{item}")
        else:
            st.markdown(
                "✅ **You have everything you need!**  \n"
                "<span style='color:#57606a;font-size:0.85rem;'>All ingredients are in your available list.</span>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # --- Raw response expander (for debugging) ---
    with st.expander("🔧 Show raw model response (debug)"):
        st.code(result.get("raw_response", ""), language="json")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.divider()
st.markdown(
    "<div style='text-align:center;color:#57606a;font-size:0.78rem;'>"
    "ChefGranite · Built with IBM watsonx.ai Granite · Powered by LangChain + FAISS"
    "</div>",
    unsafe_allow_html=True,
)
