# 🛡️ InferenceGuard: The LLM Reliability & Trust Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Stateful_Agents-orange)](https://langchain-ai.github.io/langgraph/)
[![ClickHouse](https://img.shields.io/badge/ClickHouse-OLAP_Database-yellow)](https://clickhouse.com/)
[![Grafana](https://img.shields.io/badge/Grafana-Observability-green)](https://grafana.com/)

> **An end-to-end LLMOps platform that detects hallucinations, monitors agent latency, and automates regression testing using an "LLM-as-a-Judge" architecture.**

---

## 🛑 The Problem: The "Black Box" Risk
Building an AI Agent is easy. **Keeping it from lying to your customers is hard.**

Most developers deploy LLMs and hope for the best. But in production, APIs fail, users ask malicious questions, and models hallucinate facts. Without observability, you are flying blind. A chatbot that "works on my machine" can cause significant brand damage when it hallucinates in production.

## ⚡ The Solution: InferenceGuard
**InferenceGuard** is an observability platform designed to bring **software engineering rigor** to Generative AI. It replaces "vibes-based" testing with concrete, data-driven metrics.

It doesn't just run the agent; it **watches** the agent, **grades** its reasoning, and **alerts** engineering when trust metrics drop.

---

## 💰 Business Impact (The "Why")

| Metric | Business Value |
| :--- | :--- |
| **Trust & Safety** | Prevents brand damage by detecting hallucinations (via the **Faithfulness Metric**) before they reach users. |
| **Cost Reduction** | The **Loop Detector** identifies stuck agents immediately, preventing infinite API bill run-ups. |
| **User Retention** | The **Rage Click Monitor** identifies frustrated users instantly, highlighting exactly which prompts need engineering. |
| **Operational Efficiency** | **Reduced Mean-Time-To-Resolution (MTTR) by 90%** (from hours to minutes) by visualizing tool failures in real-time. |

---

## 📸 Dashboard Preview

![Dashboard](InferenceGuard_Dashboard.gif)

**Key Visualizations:**
* **The Trust Gauge:** Real-time score (0-100%) of how often the Agent follows tool outputs vs. hallucinating.
* **The Failure Feed:** A live log of every session that crashed or returned an "Unknown" tool error.
* **Latency Heatmap:** Tracks the average time-to-first-token and total completion time.

---

## 🏗️ Architecture

A high-throughput pipeline designed for scale, ensuring zero latency impact on the user experience.

![InferenceGuard Architecture](InferenceGuard%20Architecture.png)


### 🧩 Tech Stack
* **Orchestration:** [LangGraph](https://langchain-ai.github.io/langgraph/) (Stateful ReAct Agents)
* **Inference:** Llama-3-70b / GPT-4o (via Groq)
* **Telemetry Storage:** [ClickHouse](https://clickhouse.com/) (OLAP for high-speed ingestion)
* **Visualization:** [Grafana](https://grafana.com/) (Operational & Quality Dashboards)
* **Evaluation:** Custom "LLM-as-a-Judge" pipeline (Python)

---

### 🚀 Quick Start

#### 1. Prerequisites
* Docker & Docker Compose
* Python 3.10+
* Groq API Key

#### 2. Start Infrastructure
Spin up the observability stack (ClickHouse + Grafana):
```bash
docker run -d -p 8123:8123 --name clickhouse-server clickhouse/clickhouse server
./bin/grafana server
```

#### 3. Run the Agent (Generate Traffic)
Run the agent through the stress-test suite to generate live telemetry:
```bash
pip install -r requirements.txt
python langchain_agent.py
```

#### 4. Run the Judge (Evaluate Quality)
Execute the offline evaluation script to grade recent sessions for Hallucinations and Relevance:
```bash
python dre_referee.py
```

#### 5. Access Dashboard
Open http://localhost:3000 and login (admin / admin).

---

### 📊 Key Metrics Dictionary
#### 🛡️ Trust Layer
* **Faithfulness:** Measures if the Agent's answer is derived strictly from the Tool Output. (1 = Faithful, 0 = Hallucination).

* **Answer Relevance:** Measures if the Agent actually addressed the specific user question.

* **Refusal Rate:** Tracks how often the Agent correctly declines unsafe/out-of-scope requests.

#### ⚙️ System Layer
* **Global Failure Rate:** Percentage of sessions resulting in unhandled exceptions or tool errors.

* **Tool Reliability:** Success vs. Failure counts broken down by tool (e.g., get_weather vs multiply).

* **Verbosity Ratio:** Ratio of Output Length / Input Length (detects "chatty" or inefficient agents).
