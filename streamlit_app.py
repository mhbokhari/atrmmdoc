import streamlit as st
import os
import glob
import time
import shutil
import base64 # For image encoding
from io import BytesIO # For image encoding
import traceback # Import traceback for detailed error printing
# --- Add sys import for PyInstaller check (or bundled path check) ---
import sys

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
# Use invoke directly, RetrievalQA/ConversationalRetrievalChain less suited for direct multimodal RAG
from langchain.docstore.document import Document
# Import LangChain message types
from langchain_core.messages import HumanMessage # Import message type

# --- Configuration & Constants ---
load_dotenv(override=True)
DOCUMENTS_DIR = "uploaded_docs_local_mm"
INDEX_PATH = "./faiss_index_local_mm"
# EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2" # Keep original name as default/fallback
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# --- Determine Embedding Model Path ---
DEFAULT_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedding_model_path_or_name = DEFAULT_EMBEDDING_MODEL_NAME # Default to name

# Construct the expected path to the bundled model *relative to the script*
# __file__ gives the path of the current script (streamlit_app_local.py)
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
except NameError: # __file__ not defined (e.g., interactive)
     script_dir = os.getcwd() # Fallback to current working directory

# Adjust the final folder name EXACTLY as you copied it from the cache
bundled_model_folder_name = "models--sentence-transformers--all-MiniLM-L6-v2"
potential_bundle_path = os.path.join(script_dir, "models", bundled_model_folder_name)

# Check if the bundled path exists
if os.path.isdir(potential_bundle_path):
    print(f"DEBUG: Found bundled embedding model at: {potential_bundle_path}")
    # Use st.query_params or another method if sidebar isn't ready yet
    # For now, just print to console, will show status in load_embeddings
    embedding_model_path_or_name = potential_bundle_path # Use the local path
else:
    print(f"DEBUG: Bundled model path '{potential_bundle_path}' not found. Using name for download/cache: {DEFAULT_EMBEDDING_MODEL_NAME}")

# --- Default Prompt Text (General Instructions) ---
DEFAULT_INSTRUCTIONS = """You are an AI assistant analyzing documents containing text and images (tables, figures, graphs). Your goal is to answer questions based on the provided textual context AND visual elements.

Instructions:
- Carefully examine both the text snippets and the images provided.
- If the question refers to a table, figure, or graph, primarily use the corresponding image(s) to answer, supplementing with text context if helpful for labels or descriptions.
- If the question is purely text-based, rely on the text snippets.
- If extracting data from a table/graph, be precise if possible, but state if the exact value is unclear from the image. Describe trends qualitatively if exact values are ambiguous.
- Synthesize information from both text and images when necessary.
- If the answer isn't found in the provided text or images, state that clearly.
"""

# --- Tesseract Check ---
# Add Tesseract check here if desired

# --- Caching Embeddings ---
@st.cache_resource
def load_embeddings(model_identifier): # Parameter name changed for clarity
    # Display status based on whether it's a path or name
    source_type = "bundled path" if os.path.isdir(model_identifier) else "download/cache name"
    st.info(f"Loading embedding model from {source_type}: {os.path.basename(model_identifier)}...")
    try:
        # Pass the determined path OR name to the HuggingFaceEmbeddings class
        embeddings = HuggingFaceEmbeddings(model_name=model_identifier)
        st.success("Embedding model loaded.")
        return embeddings
    except Exception as e:
        st.error(f"Failed to load embedding model from '{model_identifier}': {e}"); st.exception(e); st.stop()

# --- Helper Functions ---

def encode_image_to_base64(image: Image.Image, format="JPEG", quality=85):
    """Encodes a PIL image to a base64 string."""
    buffered = BytesIO()
    if image.mode == 'RGBA': image = image.convert('RGB') # Convert RGBA to RGB
    image.save(buffered, format=format, quality=quality)
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return f"data:image/{format.lower()};base64,{img_str}"

def process_pdf_to_texts_and_images(file_path):
    """Processes a PDF into text docs per page and generates images."""
    file_name = os.path.basename(file_path)
    file_name_lower = file_name.lower() # Use lowercase for consistency
    st.write(f"Processing (Text & Image): {file_name}")
    text_docs = []
    page_images = {} # {page_num (int): PIL.Image}

    # 1. Convert pages to images
    try:
        # st.write(f"  DEBUG: Attempting pdf2image.convert_from_path for {file_name}")
        pil_images = convert_from_path(file_path, dpi=200) # Adjust DPI as needed
        # st.write(f"  DEBUG: pdf2image SUCCESS, got {len(pil_images)} raw images.")
        for i, img in enumerate(pil_images): page_images[i + 1] = img # Use 1-based integer keys
        st.write(f"  Generated images dictionary for {len(pil_images)} pages.")
    except Exception as e:
        st.error(f"  CRITICAL Error converting PDF to images: {e}"); st.exception(e)
        return [], {}

    # 2. Extract text (standard or OCR)
    try:
        loader = PyPDFLoader(file_path); loaded_pages = loader.load()
        avg_chars = sum(len(d.page_content) for d in loaded_pages) / len(loaded_pages) if loaded_pages else 0
        needs_ocr = not loaded_pages or avg_chars < 50

        if needs_ocr:
            st.write(f"  Low/no text from parser. Attempting OCR...")
            progress_bar = st.progress(0, text=f"OCR Progress {file_name}")
            processed_text_count = 0
            for page_num, img in page_images.items():
                try:
                    text = pytesseract.image_to_string(img, lang='eng')
                    if text.strip():
                        metadata = {"source": file_name_lower, "page": int(page_num)}
                        text_docs.append(Document(page_content=text, metadata=metadata))
                        processed_text_count += 1
                    current_progress = page_num / len(page_images)
                    progress_bar.progress(current_progress, text=f"OCR P.{page_num}/{len(page_images)}")
                except Exception as e: st.warning(f"  Warn: OCR P.{page_num}: {e}")
            progress_bar.empty(); st.write(f"  OCR Done: Got text for {processed_text_count} pages.")
        else:
            st.write(f"  Got text for {len(loaded_pages)} pages via parser.")
            processed_text_docs = []
            for i, doc in enumerate(loaded_pages):
                 page_num_int = i + 1
                 processed_text_docs.append(Document(
                     page_content=doc.page_content,
                     metadata={"source": file_name_lower, "page": page_num_int}
                 ))
            text_docs = processed_text_docs

    except Exception as e:
        st.error(f"  Text extraction error: {e}. Relying on OCR if possible."); st.exception(e)
        if not text_docs and page_images:
            st.write("  Running OCR as fallback...")
            processed_text_count = 0
            for page_num, img in page_images.items():
                 try:
                     text = pytesseract.image_to_string(img, lang='eng')
                     if text.strip():
                         metadata = {"source": file_name_lower, "page": int(page_num)}
                         text_docs.append(Document(page_content=text, metadata=metadata))
                         processed_text_count +=1
                 except Exception as ocr_e: st.warning(f"  Warn: Fallback OCR P.{page_num}: {ocr_e}")
            st.write(f"  Fallback OCR: Got text for {processed_text_count} pages.")

    # st.write(f"DEBUG: process_pdf_to_texts_and_images returning {len(text_docs)} text docs and {len(page_images)} images.")
    return text_docs, page_images


# ==============================================================================
# Streamlit Application UI and Logic
# ==============================================================================
st.set_page_config(page_title="Multimodal Document Q&A", layout="wide")
st.title("📄 Multimodal Document Q&A Assistant")
st.write("Upload documents (PDF only for Vision). Ask questions about text, tables, figures, and graphs.")

# --- Session State Init ---
if "vector_store" not in st.session_state: st.session_state.vector_store = None
if "known_sources" not in st.session_state: st.session_state.known_sources = set()
if "llm" not in st.session_state: st.session_state.llm = None
if "selected_llm_option" not in st.session_state: st.session_state.selected_llm_option = None
if "page_image_cache" not in st.session_state: st.session_state.page_image_cache = {}
if "current_instructions" not in st.session_state: st.session_state.current_instructions = DEFAULT_INSTRUCTIONS

# --- Load Embeddings (using determined path/name) ---
# This should happen only once due to @st.cache_resource
embeddings = load_embeddings(embedding_model_path_or_name) # Pass the variable here

# --- Attempt to load existing vector store ---
if st.session_state.vector_store is None and os.path.exists(INDEX_PATH):
    try:
        st.session_state.vector_store = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        st.info("Loaded existing text vector store.")
        if hasattr(st.session_state.vector_store, 'docstore') and hasattr(st.session_state.vector_store.docstore, '_dict'):
             st.session_state.known_sources.update({str(d.metadata.get('source', '')).lower() for d in st.session_state.vector_store.docstore._dict.values() if d.metadata.get('source')})
             # st.write(f"DEBUG: Loaded {len(st.session_state.known_sources)} known sources from index.")
        else: st.warning("Could not read known sources from loaded index.")
    except Exception as e: st.warning(f"Could not auto-load index ({e})."); st.session_state.vector_store = None

# --- Sidebar ---
with st.sidebar:
    # ... (Keep all sidebar configuration widgets as before) ...
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
    selected_llm_display_option = st.selectbox("Select Vision Language Model:", available_llms, index=default_index, key="llm_select")
    st.session_state.selected_llm_option = llm_options_map.get(selected_llm_display_option)
    llm_temperature = st.slider("LLM Temperature:", 0.0, 1.0, 0.3, 0.1)
    st.divider(); st.subheader("Retrieval Settings")
    retrieval_mode = st.radio("Text Retrieval Mode:", ["Similarity Search", "MMR (Diversity)"], index=0, key="retrieval_mode_select")
    k_similarity = st.number_input("Relevant Text Chunks (Similarity):", 1, 10, 4, 1, key="k_similarity_input")
    if retrieval_mode == "MMR (Diversity)":
        st.caption("MMR Parameters:")
        mmr_k = st.number_input("Relevant Text Chunks (MMR Output):", 1, 10, 4, 1, key="mmr_k_input")
        mmr_fetch_k = st.number_input("Initial Candidates (Fetch K):", mmr_k, 30, max(mmr_k, 15), 1, key="mmr_fetch_k_input")
        mmr_lambda_mult = st.slider("Lambda (Diversity <-> Relevance):", 0.0, 1.0, 0.6, 0.05, key="mmr_lambda_input")
    st.divider(); st.subheader("AI Instructions")
    st.session_state.current_instructions = st.text_area("Edit General Instructions:", st.session_state.current_instructions, height=250, key="instructions_editor")
    st.divider(); st.header("📄 Document Management")
    uploaded_files = st.file_uploader("Upload New Documents (PDF only for Vision)", type=["pdf"], accept_multiple_files=True)
    process_button = st.button("Process Uploaded Documents", key="process_docs")
    clear_button = st.button("⚠️ Clear All Data", key="clear_context", type="secondary")
    # Display embedding source info
    st.divider()
    embedding_source_type = "Bundled" if os.path.isdir(embedding_model_path_or_name) else "Downloaded/Cached"
    st.caption(f"Embeddings Source: {embedding_source_type}")

# --- Clear Context Logic ---
if clear_button:
    st.info("Clearing documents, index, and image cache...")
    if os.path.exists(INDEX_PATH):
        try: shutil.rmtree(INDEX_PATH); st.success(f"Deleted index: '{INDEX_PATH}'.")
        except Exception as e: st.error(f"Error deleting index: {e}"); st.exception(e)
    else: st.info("No index found to delete.")
    st.session_state.vector_store = None; st.session_state.known_sources = set(); st.session_state.page_image_cache = {}
    st.rerun()

# --- Document Processing Logic ---
if process_button and uploaded_files:
    # ... (keep document processing logic with consistency fixes and debug prints) ...
    st.write("--- Processing Documents (Text & Images) ---")
    try:
        new_files_processed = False;
        with st.spinner("Processing documents..."):
            st.write("Saving files..."); saved_file_paths = []
            for f in uploaded_files:
                if not f.name.lower().endswith(".pdf"): st.warning(f"Skip non-PDF: {f.name}"); continue
                try: fp = os.path.join(DOCUMENTS_DIR, f.name); open(fp, "wb").write(f.getbuffer()); saved_file_paths.append(fp)
                except Exception as e: st.error(f"Err saving {f.name}: {e}")
            st.write("Identifying new files..."); new_files_to_process = []
            for fp in saved_file_paths:
                 if os.path.basename(fp).lower() not in st.session_state.known_sources: new_files_to_process.append(fp)
            st.write(f"{len(new_files_to_process)} new PDF files.")
            st.write("Processing text & images..."); new_docs_for_index = []
            if new_files_to_process:
                for file_path in new_files_to_process:
                     file_name = os.path.basename(file_path); file_name_lower = file_name.lower()
                     st.write(f"--- Processing File: {file_name} ---")
                     try:
                         processed_texts, processed_images = process_pdf_to_texts_and_images(file_path)
                         valid_texts_for_file = []
                         if processed_texts:
                             for doc in processed_texts:
                                  doc.metadata['source'] = file_name_lower
                                  try: doc.metadata['page'] = int(doc.metadata['page']); valid_texts_for_file.append(doc)
                                  except: st.warning(f"WARN: Invalid page metadata skipping text chunk {file_name} Pg{doc.metadata.get('page')}")
                             new_docs_for_index.extend(valid_texts_for_file);
                             if valid_texts_for_file: new_files_processed = True
                         if processed_images:
                              cached_count = 0
                              for page_num, img in processed_images.items():
                                   try: cache_key = f"{file_name_lower}_page{int(page_num)}"; st.session_state.page_image_cache[cache_key] = img; cached_count += 1
                                   except Exception as cache_e: st.warning(f"Warn: Failed cache img key {cache_key}: {cache_e}")
                              st.write(f"Cached {cached_count} images for {file_name}.")
                         if valid_texts_for_file or processed_images: st.session_state.known_sources.add(file_name_lower)
                         else: st.warning(f"Failed useful process for {file_name}.")
                     except Exception as e: st.error(f"Err processing {file_name}: {e}"); st.exception(e)
            else: st.info("No *new* PDF documents to process.")
            st.write("Splitting text..."); new_texts_for_faiss = []
            if new_docs_for_index:
                try: text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150); new_texts_for_faiss = text_splitter.split_documents(new_docs_for_index); st.write(f"{len(new_texts_for_faiss)} text chunks for index.")
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
    model_name = ""
    try:
        if current_llm_type_selection == "openai_vision":
            model_name = "gpt-4o"
            if openai_api_key: st.session_state.llm = ChatOpenAI(model_name=model_name, temperature=llm_temperature, openai_api_key=openai_api_key, max_tokens=2048)
            else: st.error("OpenAI Key missing."); st.session_state.llm = None
        elif current_llm_type_selection == "google_vision":
            model_name = "gemini-1.5-pro-latest"
            if google_api_key: st.session_state.llm = ChatGoogleGenerativeAI(model=model_name, temperature=llm_temperature, google_api_key=google_api_key)
            else: st.error("Google Key missing."); st.session_state.llm = None
        else: st.session_state.llm = None
        if st.session_state.llm: st.write(f"LLM Configured: {model_name}")
        else: st.write("LLM Config failed.")
    except Exception as e: st.error(f"LLM Init failed for {model_name}: {e}"); st.exception(e); st.session_state.llm = None

# --- Q&A Interface (Multimodal) ---
st.divider()
st.header("💬 Ask Questions about Text & Images")
query = st.text_area("Enter your question or task:", key="query_input", height=100, label_visibility="collapsed")

if query:
    if st.session_state.vector_store is None: st.warning("Please process documents first.")
    elif st.session_state.llm is None: st.warning("Vision LLM not configured.")
    else:
        llm_display_name = selected_llm_display_option or "Selected Vision LLM"
        with st.spinner(f"Thinking using {llm_display_name}..."):
            try:
                # --- 1. Retrieve relevant TEXT chunks ---
                # ... (Keep retriever creation logic) ...
                st.write("Step 1: Retrieving relevant text context...")
                retriever_kwargs = {}; search_type = "similarity"
                selected_retrieval_mode = st.session_state.get("retrieval_mode_select", "Similarity Search")
                if selected_retrieval_mode == "MMR (Diversity)":
                    mmr_k_val=st.session_state.get("mmr_k_input",4);mmr_fetch_k_val=st.session_state.get("mmr_fetch_k_input",15);mmr_lambda_val=st.session_state.get("mmr_lambda_input",0.6)
                    mmr_fetch_k_val=max(mmr_k_val,mmr_fetch_k_val);retriever_kwargs={'k':mmr_k_val,'fetch_k':mmr_fetch_k_val,'lambda_mult':mmr_lambda_val};search_type="mmr"
                else: retriever_kwargs={'k':st.session_state.get("k_similarity_input",4)};search_type="similarity"
                retriever = st.session_state.vector_store.as_retriever(search_type=search_type, search_kwargs=retriever_kwargs)
                relevant_text_docs = retriever.invoke(query)
                st.write(f"Found {len(relevant_text_docs)} relevant text chunks.")

                # --- 2. Identify unique relevant pages ---
                # ... (Keep relevant_pages_keys generation logic with consistency checks) ...
                relevant_pages_keys = set()
                if relevant_text_docs:
                    # st.write("DEBUG: Generating relevant page keys from metadata...")
                    for d in relevant_text_docs:
                         source = d.metadata.get('source'); page = d.metadata.get('page')
                         if source and page is not None:
                              try: page_key = f"{str(source).lower()}_page{int(page)}"; relevant_pages_keys.add(page_key)
                              except (ValueError, TypeError) as key_e: st.warning(f"Could not parse key: {key_e}")
                    st.write(f"Identified {len(relevant_pages_keys)} unique relevant page keys: {relevant_pages_keys}")

                # --- 3. Get and Encode Images ---
                # ... (Keep image retrieval and encoding logic) ...
                image_content_parts = [] # Store image dicts for LangChain message
                image_sources_display = []
                # st.write(f"DEBUG: All image cache keys available ({len(st.session_state.page_image_cache)}): {list(st.session_state.page_image_cache.keys())[:10]}...")
                if relevant_pages_keys and "page_image_cache" in st.session_state:
                    st.write("Step 2: Retrieving and encoding relevant images...")
                    max_images_to_send = 5; images_sent_count = 0
                    for page_key in relevant_pages_keys:
                         # st.write(f"DEBUG: Checking cache for key: '{page_key}'")
                         if page_key in st.session_state.page_image_cache and images_sent_count < max_images_to_send:
                             # st.write(f"DEBUG: Key '{page_key}' FOUND in cache.")
                             try:
                                 img = st.session_state.page_image_cache[page_key]
                                 base64_image = encode_image_to_base64(img)
                                 image_content_parts.append({"type": "image_url", "image_url": {"url": base64_image}})
                                 image_sources_display.append(page_key.replace("_page", " Page "))
                                 images_sent_count += 1
                             except Exception as img_e: st.warning(f"Could not encode image {page_key}: {img_e}")
                         # elif page_key not in st.session_state.page_image_cache:
                              # st.warning(f"DEBUG: Key '{page_key}' NOT FOUND in cache.")
                    st.write(f"Prepared {images_sent_count} images.")
                elif not relevant_pages_keys: st.write("No relevant pages identified from text.")
                else: st.write("Image cache is empty.")


                # --- 4. Construct LangChain Message Objects ---
                st.write("Step 3: Constructing LangChain messages...")
                message_content_list_for_lc = []

                # Part 1: Instructions Text
                instructions_content = str(st.session_state.current_instructions) or " "
                message_content_list_for_lc.append({"type": "text", "text": instructions_content}) # Use "text": key

                # Part 2: Text Context
                if relevant_text_docs:
                    context_text = "\n\n---\n\n".join([f"Source: {d.metadata.get('source')} Page: {d.metadata.get('page')}\n{str(d.page_content)}" for d in relevant_text_docs])
                    message_content_list_for_lc.append({"type": "text", "text": f"\n\nRelevant Text Context:\n{context_text or ' '}\n\n---"}) # Use "text": key
                else:
                     message_content_list_for_lc.append({"type": "text", "text": "\n\nRelevant Text Context: None Found\n\n---"})

                # Part 3: Images (if any)
                if image_content_parts:
                    message_content_list_for_lc.extend(image_content_parts)

                # Part 4: Final Question Text
                query_content = str(query) or " "
                final_query_text = f"\n\nQuestion: {query_content}"
                message_content_list_for_lc.append({"type": "text", "text": final_query_text}) # Use "text": key

                # Create the LangChain HumanMessage object
                human_message = HumanMessage(content=message_content_list_for_lc)
                st.write("LangChain HumanMessage constructed.")
                # st.write("DEBUG: LangChain Message:", human_message) # Optional Debug

                # Remove detailed prompt check as we use message objects now

                # --- 5. Call LLM using the message object ---
                st.write(f"Step 4: Sending request to {llm_display_name}...")
                # Pass the list containing the single HumanMessage object
                response = st.session_state.llm.invoke([human_message]) # Invoke expects a list of messages
                st.write("Received response.")

                # --- 6. Display Result ---
                st.subheader("Response:")
                # Response should be an AIMessage, access its content
                ai_response = response.content if hasattr(response, 'content') else "No valid response content received."
                st.markdown(ai_response)

                # --- Display sources ---
                st.subheader("Sources Considered:")
                expander_title = f"Show Sources ({selected_retrieval_mode} Text Retrieval)"
                with st.expander(expander_title):
                     if relevant_text_docs:
                         st.write("**Text Context From:**")
                         unique_text_sources = set(f"{d.metadata.get('source')} (Pg {d.metadata.get('page')})" for d in relevant_text_docs)
                         for src in sorted(list(unique_text_sources)): st.write(f"- {src}")
                     if image_sources_display:
                         st.write("**Images From:**")
                         for src in sorted(image_sources_display): st.write(f"- {src}")
                     if not relevant_text_docs and not image_sources_display:
                          st.write(" - No specific document context was retrieved for this query.")

            except Exception as e:
                st.error(f"An error occurred during Multimodal QA:"); st.exception(e)


# --- Display Known Sources in Sidebar ---
st.sidebar.divider()
st.sidebar.header("📚 Processed Documents")
if st.session_state.known_sources:
    # Display sorted, lowercased filenames
    for src in sorted(list(st.session_state.known_sources)): st.sidebar.write(f"- {src}")
else: st.sidebar.info("No documents processed yet.")