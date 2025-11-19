PYTHON ?= python
DATA_DIR ?= data
META_DIR ?= $(DATA_DIR)/meta
PARQUET_DIR ?= $(DATA_DIR)/clean/parquet
DOWNLOAD_LIMIT ?= 100
MAX_RECORDS ?= 1000
REQUEST_DELAY ?= 0.25

.PHONY: help dirs download parquet pipeline clean

help:
	@echo "Available targets:"
	@echo "  make download   # Fetch JSON batches from openFDA"
	@echo "  make parquet    # Convert JSON batches into Parquet"
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

# End-to-end pipeline
pipeline: download parquet

# Remove generated Parquet files so the pipeline can re-run cleanly
clean:
	rm -rf $(DATA_DIR)/clean
