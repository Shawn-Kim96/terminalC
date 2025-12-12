# 

CMPE 259 Project Proposal

**Terminal C**

**LLM Powered Crypto Research & Trading Assistant**

Su Hyun Kim (018219422)  
Group 13

# 

# Overview and Motivation

Cryptocurrency research requires the integration of quantitative market indicators with fast-moving qualitative signals such as news articles, social media activity, and trending lists. This project proposes the development of a domain-specific Virtual Assistant (VA) that combines a structured market database with targeted web retrieval to provide evidence-based answers to realistic user questions. Typical queries include: *“Is Bitcoin attractive to enter right now?”*, *“How should I allocate $100 today?”*, and *“What happened to the market over the last week?”*.

The project focuses on tool orchestration and prompting techniques that make open-source large language models (LLMs) reliable, efficient, and safe for use in this domain.

# Objectives

The project is structured around three main objectives:

1. **Data Backbone**  
   Construct a reproducible database for approximately thirty top-capitalization coins, covering twelve months of OHLCV (Open, High, Low, Close, Volume) data enriched with technical indicators such as RSI, moving averages, Bollinger Bands, MACD, and ATR.  
     
2. **Tool‑Augmented Assistant**  
   Implement a Virtual Assistant capable of executing schema-aware SQL queries, retrieving relevant web content (news and trending items), normalizing and caching results, and synthesizing risk-aware answers supported by citations.  
     
3. **Empirical Study**  
   Compare two open-source LLMs of different sizes on a fixed query set, and quantify the benefits of advanced prompting techniques such as prompt chaining, meta-prompting, self-reflection, and caching.

These objectives ensure that the system is reproducible offline, demonstrates practical tool use with transparent evidence citation, and provides measurable insights into the trade-offs between model size and prompting strategies under safety constraints.

# 

# Use Cases (Queries)

The assistant is intended for users seeking explainable, data-driven insights rather than opaque trading signals. Queries fall into two categories:

* **General reasoning questions**, such as:  
  * *Is BTC attractive right now?*  
  * *How should I allocate $100 today?*  
  * *How did the market perform over the past week?*  
  * *Which cryptocurrencies appear undervalued based on weak short-term momentum but strong hype or news coverage?*  
  * *What are the top three risks facing the crypto market this month, and how might they affect major assets?*  
  * *Which coins have shown the strongest recovery after recent drawdowns, and what factors support this rebound?*

* **Analytic queries**, which focus on identifying assets with specific technical or hype-driven characteristics, for example:  
  * *Which assets delivered the highest and lowest returns over the past 24 hours?*  
  * *Which coins were weak over the past seven days but showed a rebound in the last 24 hours, suggesting a potential momentum turnaround?*  
  * *Which assets are currently trading above their 50-day moving average but remain below their 200-day moving average (a possible squeeze condition)?*  
  * *Which coins are trading near the lower Bollinger Band and have recently bounced upward?*  
  * *Which coins display bullish RSI divergences on the one-hour chart with a strength score above 0.6 and a price level above the 50-day moving average?*  
  * *Which assets exhibit bearish RSI divergences on the one-day chart accompanied by declining trading volume?*  
  * *Which coins rank in the top decile for ATR-to-price ratio, indicating unusually high volatility?*  
  * *Which assets appear “undervalued” by definition—showing weak price momentum but strong hype scores over the past seven days?*  
  * *Which cryptocurrencies were the most frequently mentioned in news and social media posts over the past week, and what themes dominated those mentions?*  
  * *Which assets have been associated with negative news events (e.g., exploits or hacks) within the past 72 hours, and how severe was the impact?*  
  * *Why were the top five recommended assets selected, and what specific rules and numerical thresholds support each recommendation?*  
  * *What is the effect of prompt caching on query execution—specifically in terms of average latency and the number of tool calls required?*  
  * *Which assets show the lowest seven-day volatility while simultaneously recording rising trading volumes?*  
  * *Which coins recorded the lowest 24-hour returns among the most liquid assets?*

Such queries require multi-step synthesis: retrieving numerical evidence from the database, incorporating contextual web signals, and presenting both supportive and opposing factors before offering a non-advisory conclusion.

# System Components

## Models

To isolate the effect of model size, the project will compare two models from the same family:

* **Large model**: *Llama-3.3-70B-Instruct* (Mixture-of-Experts architecture, offering strong reasoning ability with moderate VRAM efficiency).  
* **Small model**: *Llama-3.1-8B-Instruct* (compact, fast, and compatible with common open-source toolchains).

## Dataset & Database

The database is implemented in DuckDB, chosen for its analytic performance, columnar storage, and portability. It is organized into three logical blocks:

* **Core tables**:  
  * *Assets*: Metadata about cryptocurrencies (ID, symbol, name).  
  * *Candles*: Sequential market data including OHLCV prices and precomputed indicators, aligned by timestamp.  
  * *Divergences*: Records of bullish or bearish divergence signals (e.g., RSI-based) that support strategy evaluation.  
* **Strategy tables**:  
  * *Strategies*: Descriptions and specifications of representative investment strategies (e.g., RSI entry, moving-average crossovers).  
  * *Signals*: Outputs generated from applying strategies to historical data.  
* **Web/Hype tables**:  
  * *Sources*: Metadata about news and social media platforms.  
  * *News Items*: Scraped news articles and social media posts with timestamps, titles, and content.  
  * *Hype Signals*: Links news or social events to specific assets, annotated with impact scores. These signals highlight which coins are receiving attention or are negatively affected by events such as hacks or exploits, enabling the assistant to incorporate real-time qualitative context into its responses.

The **core tables** will be populated with twelve months of OHLCV data at both one-hour and one-day intervals for approximately thirty assets. The **strategy tables** will include around ten representative strategies encoded in JSON to ensure flexibility and reproducibility. The **web and hype tables** will be seeded with two months of web-scraped content, including both news articles and Twitter (X) posts, providing qualitative context that complements the numerical market data.

## Execution Pipeline

During runtime, the assistant will operate in a ReAct-style loop:

1. Interpret the user query and generate a tool plan.  
2. Execute schema-constrained SQL queries over the database.  
3. Retrieve and normalize trending or news items, applying sentiment analysis and entity linking.  
4. Synthesize quantitative and qualitative evidence into a structured, risk-aware answer.  
5. Apply prompt caching and a self-reflection step to ensure consistency and safety.

# Experimental Components

The project will explore advanced prompting techniques to improve reliability and transparency:

* **Prompt chaining**: Decomposing tasks into a sequence of steps (parse → tool plan → SQL → evidence table → explanation) to make reasoning more reliable and auditable.  
* **Meta-prompting**: Applying governance rules such as read-only tools, schema constraints, numeric citations, risk-aware phrasing, and refusal policies.  
* **Self-reflection**: Adding a critique pass to check timeframe consistency, units, contradictory signals, and missing data before finalizing answers.

# Milestone

| Dates | Goal | Key Deliverables |
| ----- | ----- | ----- |
| 09/29–10/05 (W1) | Fill database (core) | Generate assets, candles data using python API and DuckDB Develop code that calculates indicators |
| 10/06–12 (W2) | Fill database (strategy, web) | Generate text strategies for crypto investment Make python script to web scrap informations in news & twitter (X) |
| 10/13–19 (W3) | Environment setting | Download LLM models and setup Colab (or HPC) Configure DB and web connection for LLMs |
| 10/20–26(W4) | Environment setting | Download LLM models and setup Colab (or HPC) Configure DB and web connection for LLMs |
| 10/27–11/02 (W5) | Experiments | Test with sample queries and evaluate models |
| 11/03–09 (W6) | Advance prompting | Try prompt chaining Try meta prompting |
| 11/10–16 (W7) | Advance prompting | Try self-reflection prompting Check for security testing |
| 11/17–23 (W7) | Packaging | Document reports and presentation |

