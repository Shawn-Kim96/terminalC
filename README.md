
# Terminal C

Terminal C is the crypto research assistant built for CMPE 259. The goal is simple: answer realistic analyst questions with hard data, cite exactly where every number came from, and keep the entire system portable so anyone on the team can run it locally or in Colab.

## Quick Start Guide

The execution environment depends on which model configuration you intend to test.

  * **Large Model (Llama-3.3-70B):** Uses the Hugging Face API. It requires a stable internet connection.
  * **Small Model (Llama-3.1-8B):** Runs entirely offline using local weights stored on the HPC cluster.

> **Note on Connectivity:** While it is technically possible to use the API from HPC GPU nodes, we frequently encounter intermittent HTTP connection errors with Hugging Face. Therefore, it is highly recommended to run the **Large Model locally** and the **Small Model on HPC**.

### Option 1: Local Setup (Recommended for Large Model / API)

Use this method if you are testing the Large Model or if you want to run the code on your own machine.

1.  **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

2.  **Configure Environment Variables**
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

-----

### Option 2: HPC Setup (Required for Small Model / Local Weights)

Use this method to test the Small Model (Base & LoRA). Due to the complexity of the HPC environment and the large size of the model weights, **please execute the project within my specific directory using the pre-configured conda environment.**

1.  **Connect to HPC**
    Replace `YOUR_SJSU_ID` with your student ID.

    ```bash
    ssh YOUR_SJSU_ID@coe-hpc1.sjsu.edu
    ```

2.  **Request a GPU Node**

    ```bash
    srun -p gpuqs --pty /bin/bash
    ```

3.  **Navigate to Project Directory & Activate Environment**
    You must use this specific path to access the pre-loaded model weights and environment.

    ```bash
    cd /home/018219422/terminalc
    conda activate terminalc
    ```

4.  **Run the Assistant**
    Launch the notebook (or convert/run as script if using CLI):

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
