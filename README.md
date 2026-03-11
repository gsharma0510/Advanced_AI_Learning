# Advanced LLM Learning Lab 🤖

A repository dedicated to exploring Agentic AI workflows, focusing on reasoning patterns, tool integration, and architectural reliability using **Gemini 3.1 Flash**.

## Projects & Learning Path

### 01. Router Chain Example
**Concept:** Conditional Logic & Efficiency.
- Demonstrates how to use an LLM as a "Router" to send user queries to specific expert agents or tools.
- Reduces token usage by only engaging relevant subsystems.

### 02. Parallel Consensus Example
**Concept:** Reliability & Hallucination Reduction.
- Implements a "Voting" system where multiple LLM instances generate answers in parallel.
- Compares results to find the most consistent and accurate response.

### 03. ReAct Agent (Regex-Based)
**Concept:** Manual Reasoning Loops.
- An implementation of the **Reasoning and Acting (ReAct)** pattern.
- Uses Regex parsing to bridge the gap between AI "Thoughts" and Python "Actions."
- Includes a local simulation and a real-world Yahoo Finance version.

### 04. Native Tool-Calling Agent
**Concept:** Production-Grade Tool Integration.
- Demonstrates how agents work.
- Uses **Native Function Calling** within the Gemini SDK for a more robust "handshake" between the LLM and real-time APIs.
- Features parallel tool execution and built-in error handling.

---

## Technical Stack
- **Model:** Gemini 3.1 Flash-Lite
- **Language:** Python 3.13
- **APIs:** Yahoo Finance (via `yfinance`)
- **Key Libraries:** `google-genai`, `python-dotenv`

## Setup
1. Clone the repo.
2. Create a `.env` file based on `.env.example`.
3. Run `pip install -r requirements.txt`.
