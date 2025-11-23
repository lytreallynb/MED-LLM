PYTHON ?= python
DATA_DIR ?= data
META_DIR ?= $(DATA_DIR)/meta
PARQUET_DIR ?= $(DATA_DIR)/clean/parquet
DOWNLOAD_LIMIT ?= 100
MAX_RECORDS ?= 1000
REQUEST_DELAY ?= 0.25
CHUNK_FILE ?= $(DATA_DIR)/clean/chunks.jsonl
META_FILE ?= $(DATA_DIR)/clean/fda_meta.jsonl
EMBEDDINGS_FILE ?= $(DATA_DIR)/clean/fda_embeddings.npy
INDEX_FILE ?= $(DATA_DIR)/clean/fda.index
EMBED_MODEL ?= Qwen/Qwen2.5-Embedding-1.8B

.PHONY: help dirs download parquet chunk embeddings index rag pipeline clean

help:
	@echo "Available targets:"
	@echo "  make download   # Fetch JSON batches from openFDA"
	@echo "  make parquet    # Convert JSON batches into Parquet"
	@echo "  make chunk      # Tokenize medical sections into retrieval chunks"
	@echo "  make embeddings # Generate Qwen embeddings + metadata"
	@echo "  make index      # Build FAISS index from embeddings"
	@echo "  make rag        # Ensure all RAG artifacts are built"
	@echo "  make pipeline   # Run download + parquet"
	@echo "  make clean      # Remove generated Parquet files"

# Ensure directory layout exists before running scripts
dirs:
	mkdir -p $(META_DIR)
	mkdir -p $(PARQUET_DIR)

# Download JSON batches from the openFDA API
download: dirs
	$(PYTHON) download_fda_labels.py \
		--save-dir $(META_DIR) \
		--limit $(DOWNLOAD_LIMIT) \
		--max-records $(MAX_RECORDS) \
		--sleep $(REQUEST_DELAY)

# Convert downloaded JSON into Parquet files
parquet: dirs
	$(PYTHON) convert_json_to_parquet.py \
		--input-dir $(META_DIR) \
		--output-dir $(PARQUET_DIR)

# Chunk flattened FDA sections into overlapping passages
chunk: parquet
	$(PYTHON) -m medllm.chunking \
		--parquet-dir $(PARQUET_DIR) \
		--output $(CHUNK_FILE)

# Generate Qwen embeddings for each chunk
embeddings: chunk
	$(PYTHON) -m medllm.embeddings \
		--chunks $(CHUNK_FILE) \
		--meta-output $(META_FILE) \
		--embeddings $(EMBEDDINGS_FILE) \
		--model $(EMBED_MODEL)

# Build a FAISS index for retrieval
index: embeddings
	$(PYTHON) -m medllm.indexer \
		--embeddings $(EMBEDDINGS_FILE) \
		--metadata $(META_FILE) \
		--index $(INDEX_FILE)

rag: index
	@echo "RAG artifacts ready: $(CHUNK_FILE), $(META_FILE), $(EMBEDDINGS_FILE), $(INDEX_FILE)"

# End-to-end pipeline
pipeline: download parquet

# Remove generated Parquet files so the pipeline can re-run cleanly
clean:
	rm -rf $(DATA_DIR)/clean
