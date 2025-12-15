
# Terminal C

Terminal C is the crypto research assistant built for CMPE 259. The goal is simple: answer realistic analyst questions with hard data, cite exactly where every number came from, and keep the entire system portable so anyone on the team can run it locally or in Colab.

## Start Guide

1.  **Clone this repository.** 
    Make sure the project directory name is `terminalC`.

2.  **Install files from Google Drive**
    There are 3 files to download
    1. `venv` folder: This if for virtual envrionment. You can make your own environment yourself, but it is better to just download from google drive. ([link](https://drive.google.com/file/d/1yotOST_PzHhu61ZFR3vLVKEHmwCnuXCL/view?usp=drive_link))
    2. `data` folder: Includes database and model. ([link](https://drive.google.com/file/d/1mNo3zTmn54OxCm39Yf3s4uQ7uHekCU49/view?usp=drive_link))
    3. `models` folder: Includes lora finetuned small model. ([link](https://drive.google.com/file/d/1OMvjTqq1yqMvhGGuGkMq7_mdtxE48NV1/view?usp=drive_link))

    After downloading three files, make sure you have your directory structure as below.
    ```bash
    .
    ├── data
    ├── .venv   # change venv file name to .venv
    ├── models
    ├── .env    # .env file is shown below
    ├── notebooks
    ├── ...
    ```

3.  **Configure Environment Variables**
    Create a `.env` file in the root directory and add the following configuration.

      * *Replace `PUT_YOUR_HUGGINGFACE_TOKEN` with your actual key.*

    <!-- end list -->

    ```ini
    # ========= 1. Project Path ==============
    PROJECT_NAME=terminalC
    TERMINALC_DATA_SUB_DIR=data
    TERMINALC_CACHE_SUB_DIR=data/cache
    TERMINALC_MODELS_SUB_DIR=data/model
    TERMINALC_DUCKDB_SUB_PATH=data/database/market.duckdb
    TERMINALC_DUCKDB_READONLY_SUB_PATH=data/database/market_readonly.duckdb

    # ========= 2. Hugging Face ======================
    HUGGINGFACE_TOKEN=PUT_YOUR_HUGGINGFACE_TOKEN
    LARGE_MODEL_ENDPOINT=meta-llama/Llama-3.3-70B-Instruct
    SMALL_MODEL_ENDPOINT=meta-llama/Llama-3.1-8B-Instruct
    SMALL_MODEL_ADAPTER_DIR=models/small_model_lora
    ```

3.  **Run the Assistant**
    Open and execute the main notebook:

    ```bash
    jupyter notebook terminalc.ipynb
    ```

## Directory structure

- `terminalc/data_scripts/` – end-to-end data builders. They fetch OHLCV candles, compute all technical indicators, derive Investing.com style signals, assemble the strategy catalog, scrape CoinDesk headlines, and load everything into DuckDB.
- `terminalc/runtime_core/` – the production pipeline. Inside you’ll find the intent parser, query planner, DuckDB adapter, prompt builder, cache layers, LLM clients, security guards, and now a lightweight CoinDesk web-search tool.
- `notebooks/` – scratchpads I used while validating the pipeline (data sanity checks, pipeline dry-runs, final demo notebooks).
- `models/` – the LoRA adapter checkpoints for the distilled small model.
- `results/` – CSVs from earlier model comparisons plus any benchmark reports you generate with the new tooling.
- `report/` – the official proposal, abstract, and final report submitted to the course staff.
