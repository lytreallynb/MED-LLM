# Med-LLM

Small utilities to download FDA drug label data from the [openFDA](https://open.fda.gov/) API and convert the JSON responses into columnar Parquet files for downstream processing.

## Environment setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data pipeline

The workflow is powered by the provided `Makefile` and writes intermediate files to `data/`.

```bash
make download            # fetch JSON batches into data/meta
make parquet             # convert existing JSON batches into Parquet (data/clean/parquet)
make pipeline            # run download + parquet in one command
make clean               # drop generated Parquet files
```

You can override defaults on the fly, for example:

```bash
make MAX_RECORDS=5000 DOWNLOAD_LIMIT=75
```

The download step respects openFDA's pagination and writes newline-delimited JSON files such as `data/meta/batch_0.json`. The conversion step flattens nested JSON via `pandas.json_normalize` and stores `batch_*.parquet`, which can be loaded efficiently in later stages of the project.
