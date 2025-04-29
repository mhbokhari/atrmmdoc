import streamlit as st
import os
import glob
import time
import shutil
import base64 # For image encoding
from io import BytesIO # For image encoding
import traceback # Import traceback for detailed error printing

# --- Core Processing Logic Imports ---
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_path # Needs Poppler installed
from PIL import Image # Needs Pillow installed
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
# LLM Integrations - ENSURE YOU SELECT A VISION-CAPABLE MODEL LATER
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
# Use invoke directly for multimodal
from langchain.docstore.document import Document
from langchain_core.messages import HumanMessage # Import message type
# Search Tool Import
from langchain_community.tools import DuckDuckGoSearchRun


# --- Configuration & Constants ---
load_dotenv(override=True)
DOCUMENTS_DIR = "uploaded_docs_local_mm"
INDEX_PATH = "./faiss_index_local_mm"
# --- Use Model Name String Directly ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# --- End Model Name ---
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# --- Default Instructions ---
DEFAULT_INSTRUCTIONS = """You are an AI assistant analyzing documents containing text and images (tables, figures, graphs). Your goal is to answer questions based on the provided textual context AND visual elements.

Instructions:
- Prioritize information found in the provided document text snippets and images (if available and image analysis is enabled). Clearly indicate when answers are derived from these sources.
- If web search results are provided, use them to supplement or provide context, but clearly indicate information derived from the web.
- Analyze images (tables, figures, graphs) if they are provided and image analysis is enabled. Describe visual trends or extract data cautiously, noting any ambiguities.
- If the question is purely text-based or image analysis is disabled, rely on the text snippets and web results (if provided).
- Synthesize information from all available sources (documents, images, web) when necessary.
- If the answer cannot be found in any of the provided sources, state that clearly. Do not guess.
- Structure your answers clearly and avoid overly brief responses.
"""

# --- Tesseract Check ---
# Add Tesseract check here if desired

# --- Caching Embeddings ---
@st.cache_resource
def load_embeddings(model_name): # Parameter is the model name string
    # Status message indicates potential download
    st.info(f"Loading embedding model: {model_name} (will download if needed)...")
    try:
        # Load using the model name string from Hugging Face Hub
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        st.success("Embedding model loaded.")
        return embeddings
    except Exception as e:
        st.error(f"Failed to load embedding model '{model_name}': {e}")
        st.exception(e)
        st.stop()

# --- Helper Functions ---

def encode_image_to_base64(image: Image.Image, format="JPEG", quality=85):
    """Encodes a PIL image to a base64 string."""
    buffered = BytesIO()
    if image.mode == 'RGBA': image = image.convert('RGB')
    image.save(buffered, format=format, quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/{format.lower()};base64,{img_str}"

def process_pdf_to_texts_and_images(file_path):
    """Processes a PDF into text docs per page and generates images."""
    file_name = os.path.basename(file_path); file_name_lower = file_name.lower()
    st.write(f"Processing (Text & Image): {file_name}")
    text_docs = []; page_images = {}
    try: # Image conversion
        pil_images = convert_from_path(file_path, dpi=200)
        for i, img in enumerate(pil_images): page_images[i + 1] = img
        st.write(f"  Generated {len(pil_images)} images.")
    except Exception as e: st.error(f"PDF Img Convert Error: {e}"); st.exception(e); return [], {}
    try: # Text extraction
        loader = PyPDFLoader(file_path); loaded_pages = loader.load()
        avg_chars=sum(len(d.page_content) for d in loaded_pages)/len(loaded_pages) if loaded_pages else 0
        needs_ocr = not loaded_pages or avg_chars < 50
        if needs_ocr: # OCR
            st.write(f"Attempting OCR...")
            progress_bar = st.progress(0, text=f"OCR {file_name}")
            for page_num, img in page_images.items():
                try: text = pytesseract.image_to_string(img, lang='eng')
                except Exception as ocr_e_inner: st.warning(f"OCR Err P.{page_num}: {ocr_e_inner}"); continue
                if text.strip(): text_docs.append(Document(page_content=text, metadata={"source": file_name_lower, "page": int(page_num)}))
                progress_bar.progress(page_num / len(page_images), f"OCR P.{page_num}/{len(page_images)}")
            progress_bar.empty(); st.write(f"OCR Done: Got {len(text_docs)} text pages.")
        else: # Standard
            st.write(f"Got {len(loaded_pages)} text pages via parser.")
            processed_docs = [Document(page_content=d.page_content, metadata={"source": file_name_lower, "page": i+1}) for i, d in enumerate(loaded_pages)]
            text_docs = processed_docs
    except Exception as e: # Fallback OCR
        st.error(f"Text extract error: {e}. OCR fallback?"); st.exception(e)
        if not text_docs and page_images:
            st.write("Fallback OCR..."); fallback_count = 0
            for page_num, img in page_images.items():
                 try: text = pytesseract.image_to_string(img, lang='eng')
                 except Exception as ocr_e_fb: st.warning(f"Fallback OCR Err P.{page_num}: {ocr_e_fb}"); continue
                 if text.strip(): text_docs.append(Document(page_content=text, metadata={"source": file_name_lower, "page": int(page_num)})); fallback_count += 1
            st.write(f"Fallback OCR got {fallback_count} text pages.")
    return text_docs, page_images


# --- Text Only Processing Function (if needed when vision disabled) ---
def load_and_process_single_doc_text_only(file_path):
    docs = []; file_name = os.path.basename(file_path); file_name_lower = file_name.lower()
    st.write(f"Processing (Text Only): {file_name}")
    if file_path.lower().endswith(".pdf"):
        try:
            standard_loader = PyPDFLoader(file_path); pdf_docs = standard_loader.load(); is_likely_scanned = False
            if not pdf_docs: is_likely_scanned = True; st.write(f"  No text std. OCR?")
            else:
                avg_chars = sum(len(d.page_content) for d in pdf_docs) / len(pdf_docs) if pdf_docs else 0
                if avg_chars < 100: is_likely_scanned = True; st.write(f"  Low text ({avg_chars:.0f} char/pg). OCR?")
            if is_likely_scanned: ocr_docs = ocr_pdf_to_text(file_path); docs.extend(ocr_docs or [])
            else:
                st.write(f"  Loaded {len(pdf_docs)} pages std.")
                for i, doc in enumerate(pdf_docs): doc.metadata["source"] = file_name_lower; doc.metadata["page"] = i + 1
                docs.extend(pdf_docs)
        except Exception as e: st.error(f"  Err PDF {file_name}: {e}. OCR fallback?"); ocr_docs = ocr_pdf_to_text(file_path); docs.extend(ocr_docs or [])
    elif file_path.lower().endswith(".txt"):
        try:
            loader = TextLoader(file_path); loaded_docs = loader.load()
            for doc in loaded_docs: doc.metadata["source"] = file_name_lower # Add metadata
            docs.extend(loaded_docs); st.write(f"  Loaded TXT.")
        except Exception as e: st.error(f"  Err TXT {file_name}: {e}")
    else: st.warning(f"  Skip type: {file_name}")
    return docs


# ==============================================================================
# Streamlit Application UI and Logic
# ==============================================================================
st.set_page_config(page_title="Multimodal Document Q&A + Web", layout="wide")
st.title("📄 Multimodal Document Q&A Assistant with Web Search")
st.write("Upload PDFs/TXTs, configure the AI, ask questions, optionally include web search & image analysis.")

# --- Session State Init ---
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "known_sources" not in st.session_state: st.session_state.known_sources = set()
if "llm" not in st.session_state: st.session_state.llm = None
if "selected_llm_option" not in st.session_state: st.session_state.selected_llm_option = None
if "page_image_cache" not in st.session_state: st.session_state.page_image_cache = {}
if "current_instructions" not in st.session_state: st.session_state.current_instructions = DEFAULT_INSTRUCTIONS

# --- Load Embeddings ---
# Load using the model name constant directly
embeddings = load_embeddings(EMBEDDING_MODEL_NAME)

# --- Attempt to load existing vector store ---
if st.session_state.vector_store is None and os.path.exists(INDEX_PATH):
    try:
        st.session_state.vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        st.info("Loaded existing text vector store.")
        if hasattr(st.session_state.vector_store, 'docstore'): st.session_state.known_sources.update({str(d.metadata.get('source', '')).lower() for d in st.session_state.vector_store.docstore._dict.values() if d.metadata.get('source')})
    except Exception as e: st.warning(f"Could not auto-load index ({e})."); st.session_state.vector_store = None

# --- Sidebar ---
with st.sidebar:
    # ... (Keep all sidebar configuration widgets) ...
    st.header("⚙️ Configuration")
    openai_api_key = os.getenv("OPENAI_API_KEY"); google_api_key = os.getenv("GOOGLE_API_KEY")
    available_llms = []; llm_options_map = {}
    if openai_api_key: available_llms.append("OpenAI (GPT-4o/Turbo Vision)"); llm_options_map["OpenAI (GPT-4o/Turbo Vision)"] = "openai_vision"
    if google_api_key: available_llms.append("Google (Gemini 1.5 Pro Vision)"); llm_options_map["Google (Gemini 1.5 Pro Vision)"] = "google_vision"
    if not available_llms: st.error("No Vision LLM API keys found."); st.stop()
    default_index = 0
    if st.session_state.selected_llm_option:
        for i, opt in enumerate(available_llms):
             if llm_options_map.get(opt) == st.session_state.selected_llm_option: default_index = i; break
    selected_llm_display_option = st.selectbox("Select Vision Language Model:", available_llms, index=default_index, key="llm_select", help="Choose the AI model (must support vision for image analysis).")
    st.session_state.selected_llm_option = llm_options_map.get(selected_llm_display_option)
    llm_temperature = st.slider("LLM Temperature:", 0.0, 1.0, 0.3, 0.1, help="Lower=factual, Higher=creative.")

    st.divider(); st.subheader("Features & Retrieval")
    use_vision_analysis = st.checkbox("Enable Image Analysis", value=True, key="use_vision", help="Analyze images (figures, tables) in PDFs? Requires PDF uploads & Vision LLM.")
    use_web_search = st.checkbox("Include Web Search", value=False, key="use_web_search", help="Perform web search for query and add results to context?")
    retrieval_mode = st.radio("Text Retrieval Mode:", ["Similarity Search", "MMR (Diversity)"], index=0, key="retrieval_mode_select", help="Similarity=closest matches. MMR=balance relevance & diversity.")
    k_similarity = st.number_input("Relevant Text Chunks (Similarity):", 1, 10, 4, 1, key="k_similarity_input", help="# chunks for Similarity.")
    if retrieval_mode == "MMR (Diversity)":
        st.caption("MMR Parameters:")
        mmr_k = st.number_input("Relevant Text Chunks (MMR Output):", 1, 10, 4, 1, key="mmr_k_input", help="Final # diverse chunks for MMR.")
        mmr_fetch_k = st.number_input("Initial Candidates (Fetch K):", mmr_k, 30, max(mmr_k, 15), 1, key="mmr_fetch_k_input", help="# chunks to fetch initially for MMR.")
        mmr_lambda_mult = st.slider("Lambda (Diversity <-> Relevance):", 0.0, 1.0, 0.6, 0.05, key="mmr_lambda_input", help="0=Max Diversity, 1=Max Similarity.")

    st.divider(); st.subheader("AI Instructions")
    st.session_state.current_instructions = st.text_area("Edit General Instructions:", st.session_state.current_instructions, height=200, key="instructions_editor", help="Edit core instructions for the AI.")

    st.divider(); st.header("📄 Document Management")
    upload_help = "Upload PDFs (required for image analysis)" if use_vision_analysis else "Upload PDF or TXT files."
    allowed_types = ["pdf"] if use_vision_analysis else ["pdf", "txt"]
    uploaded_files = st.file_uploader(f"Upload New Documents ({'PDF only' if use_vision_analysis else 'PDF, TXT'})", type=allowed_types, accept_multiple_files=True, help=upload_help)
    process_button = st.button("Process Uploaded Documents", key="process_docs")
    clear_button = st.button("⚠️ Clear All Data", key="clear_context", type="secondary")

# --- Clear Context Logic ---
if clear_button:
    # ... (keep clear button logic) ...
    st.info("Clearing documents, index, and image cache...");
    if os.path.exists(INDEX_PATH):
        try: shutil.rmtree(INDEX_PATH); st.success(f"Deleted index: '{INDEX_PATH}'.")
        except Exception as e: st.error(f"Error deleting index: {e}"); st.exception(e)
    else: st.info("No index found.")
    st.session_state.vector_store = None; st.session_state.known_sources = set(); st.session_state.page_image_cache = {}
    st.rerun()

# --- Document Processing Logic ---
if process_button and uploaded_files:
    st.write("--- Processing Documents ---")
    try:
        new_files_processed = False;
        with st.spinner("Processing documents..."):
            st.write("Saving files..."); saved_file_paths = []
            process_vision = st.session_state.get("use_vision", True) # Check toggle status
            for f in uploaded_files:
                 is_pdf = f.name.lower().endswith(".pdf")
                 is_txt = f.name.lower().endswith(".txt")
                 if process_vision and not is_pdf: st.warning(f"Skip non-PDF {f.name} (Image analysis enabled)"); continue
                 if not is_pdf and not is_txt: st.warning(f"Skip unsupported type {f.name}"); continue
                 try: fp = os.path.join(DOCUMENTS_DIR, f.name); open(fp, "wb").write(f.getbuffer()); saved_file_paths.append(fp)
                 except Exception as e: st.error(f"Err saving {f.name}: {e}")

            st.write("Identifying new files..."); new_files_to_process = []
            for fp in saved_file_paths:
                 if os.path.basename(fp).lower() not in st.session_state.known_sources: new_files_to_process.append(fp)
            st.write(f"{len(new_files_to_process)} new files.")

            st.write("Processing text & images (if applicable)..."); new_docs_for_index = []
            if new_files_to_process:
                for file_path in new_files_to_process:
                     file_name = os.path.basename(file_path); file_name_lower = file_name.lower()
                     st.write(f"--- Processing File: {file_name} ---")
                     try:
                         processed_texts = []; processed_images = {}
                         is_pdf = file_name_lower.endswith(".pdf")
                         # Process based on toggle and file type
                         if is_pdf and process_vision:
                             processed_texts, processed_images = process_pdf_to_texts_and_images(file_path)
                         elif is_pdf or file_name_lower.endswith(".txt"):
                             processed_texts = load_and_process_single_doc_text_only(file_path)

                         valid_texts_for_file = []
                         if processed_texts:
                             for doc in processed_texts:
                                  doc.metadata['source'] = file_name_lower
                                  try: doc.metadata['page'] = int(doc.metadata.get('page',1)); valid_texts_for_file.append(doc)
                                  except: st.warning(f"WARN: Invalid page meta for {file_name} Pg{doc.metadata.get('page')}")
                             new_docs_for_index.extend(valid_texts_for_file);
                             if valid_texts_for_file: new_files_processed = True

                         if processed_images: # Cache images if they were generated
                              cached_count = 0
                              for page_num, img in processed_images.items():
                                   try: cache_key = f"{file_name_lower}_page{int(page_num)}"; st.session_state.page_image_cache[cache_key] = img; cached_count += 1
                                   except Exception as cache_e: st.warning(f"Warn: Failed cache img key {cache_key}: {cache_e}")
                              st.write(f"Cached {cached_count} images.")
                         # Update known sources if text OR images were processed
                         if valid_texts_for_file or processed_images: st.session_state.known_sources.add(file_name_lower)
                         else: st.warning(f"Failed useful process for {file_name}.")
                     except Exception as e: st.error(f"Err processing {file_name}: {e}"); st.exception(e)
            else: st.info("No *new* documents to process.")

            # ... (Splitting text and Updating vector store logic remains the same) ...
            st.write("Splitting text..."); new_texts_for_faiss = []
            if new_docs_for_index:
                try: text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150); new_texts_for_faiss = text_splitter.split_documents(new_docs_for_index); st.write(f"{len(new_texts_for_faiss)} text chunks.")
                except Exception as e: st.error(f"Err splitting: {e}"); st.exception(e)
            st.write("Updating vector store (Text Only)...");
            if new_texts_for_faiss:
                 try:
                     if st.session_state.vector_store is None and os.path.exists(INDEX_PATH):
                          try: st.session_state.vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
                          except: st.warning("Could not load store."); st.session_state.vector_store = None
                     if st.session_state.vector_store is not None: st.session_state.vector_store.add_documents(new_texts_for_faiss); st.write(f"Added {len(new_texts_for_faiss)} chunks.")
                     else: os.makedirs(INDEX_PATH, exist_ok=True); st.session_state.vector_store = FAISS.from_documents(new_texts_for_faiss, embeddings); st.write("Created new store.")
                     st.session_state.vector_store.save_local(INDEX_PATH); st.success("Text Vector Store updated & saved.")
                 except Exception as e: st.error(f"Err updating store: {e}"); st.exception(e)
            elif not new_files_processed: st.write("No new text chunks to add.")

        st.session_state.llm_needs_update = True
        st.write("--- Document Processing Finished ---")
        st.sidebar.expander("DEBUG: Show Image Cache Keys").write(sorted(list(st.session_state.page_image_cache.keys())))
    except Exception as main_proc_e: st.error("Doc processing workflow error."); st.exception(main_proc_e)


# --- Initialize or Update LLM ---
# ... (keep LLM initialization logic as before, using vision models) ...
llm_needs_update_flag = st.session_state.pop('llm_needs_update', False)
stored_llm = st.session_state.get("llm")
current_llm_type_selection = st.session_state.selected_llm_option
llm_needs_update = False
if llm_needs_update_flag or stored_llm is None or \
   (current_llm_type_selection == "openai_vision" and not isinstance(stored_llm, ChatOpenAI)) or \
   (current_llm_type_selection == "google_vision" and not isinstance(stored_llm, ChatGoogleGenerativeAI)) or \
   (stored_llm is not None and getattr(stored_llm, 'temperature', None) != llm_temperature):
    llm_needs_update = True

if llm_needs_update and current_llm_type_selection:
    st.info(f"Configuring {current_llm_type_selection.split('_')[0].capitalize()} Vision LLM (Temp: {llm_temperature})...")
    model_name = ""; vision_model_name = ""
    try:
        if current_llm_type_selection == "openai_vision":
            vision_model_name = "gpt-4o"
            if openai_api_key: st.session_state.llm = ChatOpenAI(model_name=vision_model_name, temperature=llm_temperature, openai_api_key=openai_api_key, max_tokens=2048)
            else: st.error("OpenAI Key missing."); st.session_state.llm = None
        elif current_llm_type_selection == "google_vision":
            vision_model_name = "gemini-1.5-pro-latest"
            if google_api_key: st.session_state.llm = ChatGoogleGenerativeAI(model=vision_model_name, temperature=llm_temperature, google_api_key=google_api_key)
            else: st.error("Google Key missing."); st.session_state.llm = None
        else: st.session_state.llm = None
        if st.session_state.llm: st.write(f"LLM Configured: {vision_model_name}")
        else: st.write("LLM Config failed.")
    except Exception as e: st.error(f"LLM Init failed for {vision_model_name}: {e}"); st.exception(e); st.session_state.llm = None

# --- Q&A Interface (Multimodal with Toggles) ---
st.divider()
st.header("💬 Ask Questions or Assign Tasks")
query = st.text_area("Enter your question or task:", key="query_input", height=100, label_visibility="collapsed")

if query:
    if st.session_state.vector_store is None: st.warning("Please process documents first.")
    elif st.session_state.llm is None: st.warning("Vision LLM not configured.")
    else:
        # --- Read toggle states ---
        process_images_flag = st.session_state.get("use_vision", True)
        use_web_search_flag = st.session_state.get("use_web_search", False)

        llm_display_name = selected_llm_display_option or "Selected Vision LLM"
        with st.spinner(f"Thinking using {llm_display_name}..."):
            try:
                # --- 1. Retrieve relevant TEXT chunks (Always done) ---
                st.write("Step 1: Retrieving relevant text context...")
                # ... (Keep retriever creation logic) ...
                retriever_kwargs = {}; search_type = "similarity"
                selected_retrieval_mode = st.session_state.get("retrieval_mode_select", "Similarity Search")
                if selected_retrieval_mode == "MMR (Diversity)":
                    mmr_k_val=st.session_state.get("mmr_k_input",4);mmr_fetch_k_val=st.session_state.get("mmr_fetch_k_input",15);mmr_lambda_val=st.session_state.get("mmr_lambda_input",0.6)
                    mmr_fetch_k_val=max(mmr_k_val,mmr_fetch_k_val);retriever_kwargs={'k':mmr_k_val,'fetch_k':mmr_fetch_k_val,'lambda_mult':mmr_lambda_val};search_type="mmr"
                else: retriever_kwargs={'k':st.session_state.get("k_similarity_input",4)};search_type="similarity"
                retriever = st.session_state.vector_store.as_retriever(search_type=search_type, search_kwargs=retriever_kwargs)
                relevant_text_docs = retriever.invoke(query)
                st.write(f"Found {len(relevant_text_docs)} relevant text chunks.")

                # --- 2. Perform Web Search (Conditional) ---
                web_search_results = None
                if use_web_search_flag:
                    st.write("Step 2a: Performing web search...")
                    try:
                        search_tool = DuckDuckGoSearchRun(max_results=3) # Limit results
                        web_search_results = search_tool.run(query)
                        st.write("Web search completed.")
                    except Exception as search_e: st.warning(f"Web search failed: {search_e}")

                # --- 3. Get and Encode Images (Conditional) ---
                image_content_parts = [] # Store image dicts for LangChain message
                image_sources_display = []
                if process_images_flag: # Check toggle status
                    st.write("Step 2b: Retrieving and encoding relevant images (if found)...")
                    relevant_pages_keys = set()
                    if relevant_text_docs:
                         for d in relevant_text_docs:
                              source=d.metadata.get('source'); page=d.metadata.get('page')
                              if source and page is not None:
                                   try: page_key = f"{str(source).lower()}_page{int(page)}"; relevant_pages_keys.add(page_key)
                                   except: pass # Ignore parsing errors

                    if relevant_pages_keys and "page_image_cache" in st.session_state:
                        max_images_to_send = 5; images_sent_count = 0
                        for page_key in relevant_pages_keys:
                             if page_key in st.session_state.page_image_cache and images_sent_count < max_images_to_send:
                                 try:
                                     img = st.session_state.page_image_cache[page_key]; base64_image = encode_image_to_base64(img)
                                     image_content_parts.append({"type": "image_url", "image_url": {"url": base64_image}})
                                     image_sources_display.append(page_key.replace("_page", " Page "))
                                     images_sent_count += 1
                                 except Exception as img_e: st.warning(f"Could not encode image {page_key}: {img_e}")
                        st.write(f"Prepared {images_sent_count} images.")
                    # else: st.write("No relevant/cached images found.") # Implicit
                else:
                    st.info("Image analysis is disabled via sidebar toggle.")


                # --- 4. Construct LangChain Message Objects ---
                st.write("Step 3: Constructing LangChain messages...")
                message_content_list_for_lc = []

                # Part 1: Instructions Text
                message_content_list_for_lc.append({"type": "text", "text": str(st.session_state.current_instructions or " ")})

                # Part 2: Text Context
                if relevant_text_docs:
                    context_text = "\n\n---\n\n".join([f"Source: {d.metadata.get('source')} Page: {d.metadata.get('page')}\n{str(d.page_content)}" for d in relevant_text_docs])
                    message_content_list_for_lc.append({"type": "text", "text": f"\n\nRelevant Document Text Context:\n{context_text or ' '}\n\n---"})
                else: message_content_list_for_lc.append({"type": "text", "text": "\n\nRelevant Document Text Context: None Found\n\n---"})

                # Part 3: Web Search Results (Conditional)
                if use_web_search_flag and web_search_results:
                     message_content_list_for_lc.append({"type": "text", "text": f"\n\nWeb Search Results:\n{str(web_search_results)}\n\n---"})
                elif use_web_search_flag:
                     message_content_list_for_lc.append({"type": "text", "text": "\n\nWeb Search Results: None Found or Search Failed\n\n---"})

                # Part 4: Images (Conditional)
                if process_images_flag and image_content_parts:
                    message_content_list_for_lc.extend(image_content_parts)
                    # message_content_list_for_lc.append({"type": "text", "text": "\n\n--- End Images ---"}) # Optional separator

                # Part 5: Final Question Text
                message_content_list_for_lc.append({"type": "text", "text": f"\n\nQuestion: {str(query) or ' '}"})

                # Create the LangChain HumanMessage object
                human_message = HumanMessage(content=message_content_list_for_lc)
                st.write("LangChain HumanMessage constructed.")
                # st.write("DEBUG: LangChain Message:", human_message) # Optional Debug

                # --- 5. Call LLM using the message object ---
                st.write(f"Step 4: Sending request to {llm_display_name}...")
                response = st.session_state.llm.invoke([human_message]) # Invoke expects a list
                st.write("Received response.")

                # --- 6. Display Result ---
                st.subheader("Response:")
                ai_response = response.content if hasattr(response, 'content') else "No response content."
                st.markdown(ai_response)

                # --- Display sources ---
                st.subheader("Sources Considered:")
                with st.expander("Show Sources Details"):
                     if relevant_text_docs:
                         st.write("**Text Context From:**")
                         unique_text_sources = set(f"{d.metadata.get('source')} (Pg {d.metadata.get('page')})" for d in relevant_text_docs)
                         for src in sorted(list(unique_text_sources)): st.write(f"- {src}")
                     if use_web_search_flag:
                          st.write(f"**Web Search:** {'Used (see results in context if provided)' if web_search_results else 'Attempted but no results/failed'}")
                     if process_images_flag and image_sources_display:
                         st.write("**Images From:**")
                         for src in sorted(image_sources_display): st.write(f"- {src}")
                     if not relevant_text_docs and not image_sources_display and not (use_web_search_flag and web_search_results):
                          st.write(" - No specific context sources were used for this query.")

            except Exception as e:
                st.error(f"An error occurred during Multimodal QA:"); st.exception(e)


# --- Display Known Sources in Sidebar ---
st.sidebar.divider()
st.sidebar.header("📚 Processed Documents")
if st.session_state.known_sources:
    for src in sorted(list(st.session_state.known_sources)): st.sidebar.write(f"- {src}")
else: st.sidebar.info("No documents processed yet.")