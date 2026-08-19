# ZARA Motion AI Agent

An AI agent for analyzing human motion from wearable sensor data. The project uses the **ZARA motion feature-extraction framework** together with an AI agent, analysis tools, similarity search, and RAG to understand motion patterns.

The main idea is simple: give the agent motion data, let it use the right tools to analyze the signals, and return an explanation of what is happening.

## How It Works

```text
Motion Sensor Data
        ↓
Preprocessing
        ↓
ZARA Feature Extraction
        ↓
AI Motion Agent
        ↓
Tools + RAG + Similarity Search
        ↓
Motion Analysis
```

The agent does not work directly from raw sensor values alone. ZARA first converts the sensor signals into useful motion features, which the agent can then use for analysis.

## What the Agent Can Do

The agent can work with motion data to:

* Analyze accelerometer and gyroscope signals
* Extract ZARA motion features
* Look for unusual motion patterns
* Compare motion with similar previously analyzed windows
* Retrieve relevant motion knowledge using RAG
* Combine results from different analysis tools
* Return a readable interpretation of the motion

The goal is to make the system more than a simple activity classifier. The agent can use different tools and pieces of information when analyzing a motion sequence.

## ZARA Feature Extraction

ZARA is used as the main feature-extraction component.

It provides features from different parts of the signal, including:

* Time-domain statistics
* Frequency-domain features
* FFT and STFT
* Wavelet features
* Autocorrelation
* Jerk
* Autoregressive features
* Permutation entropy
* Motion magnitude
* Gravity and dynamic motion
* Cross-channel relationships

These features give the agent a structured representation of the motion instead of relying only on the raw sensor data.

## Agent Architecture

The agent is connected to several tools:

```text
                 AI Motion Agent
                        │
        ┌───────────────┼───────────────┐
        ↓               ↓               ↓
   ZARA Features   Motion Analysis   Similarity Search
                        │
                        ↓
                  Anomaly Analysis
                        │
                        ↓
                       RAG
```

This allows the agent to combine signal analysis, retrieved information, and similar motion patterns when producing its result.

## Backend & Frontend

The backend is built with **FastAPI** and handles the motion-processing and agent pipeline.

The frontend is built with **Streamlit** and provides a simple interface to upload motion data and view the analysis.

```text
Streamlit
    ↓
FastAPI
    ↓
Motion Agent
    ↓
ZARA + Analysis Tools + RAG
```

## Project Structure

```text
zara_motion_agent/
│
├── backend/
│   ├── app.py
│   ├── routers/
│   │   └── analyze.py
│   └── services/
│       └── pipeline.py
│
├── frontend/
│   └── pages/
│       └── 1_Upload_Analyze.py
│
├── zara_core/
│   ├── get_feats.py
│   ├── data_preprocess.py
│   └── ...
│
├── knowledge/
│   ├── har_knowledge.txt
│   └── activity_pair_knowledge.json
│
└── requirements.txt
```

## Tech Stack

* Python
* FastAPI
* Streamlit
* ZARA
* NumPy
* SciPy
* Scikit-learn
* XGBoost
* FAISS
* PyWavelets
* RAG / LLM Agent

## Current Status

### Done

* ZARA feature extraction integrated
* Motion preprocessing
* Motion analysis tools
* AI motion agent
* Similarity search
* Motion knowledge retrieval
* Anomaly-analysis framework
* FastAPI backend
* Streamlit frontend
* Backend–frontend integration

### Next

* Connect a labeled HAR dataset
* Train activity-recognition models
* Evaluate accuracy, precision, recall, and F1
* Test anomaly detection
* Compare different feature representations
* Evaluate how much RAG and agent reasoning improve the analysis

## Note

The current version focuses on building the **AI motion-analysis agent and its underlying pipeline**. No accuracy claims are made yet because the system has not been evaluated on a properly labeled test dataset.
