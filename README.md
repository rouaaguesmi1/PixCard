# PixCard: AI-Powered Credit Risk Platform

PixCard is a full-stack, AI-powered SaaS platform that automates credit scoring, risk forecasting, compliance reporting, and customer segmentation for financial institutions serving SMEs. It transforms raw financial documents into explainable credit decisions and regulatory outputs via OCR, NLP, and machine learning pipelines.

---

## 🔧 Features

### 1. Document Ingestion & Processing
- Ingests PDFs via upload or API
- OCR (Tesseract, Google Vision) + NLP (spaCy, FinBERT)
- Transforms unstructured financial documents into structured data

### 2. Credit Scoring Engine
- ML models (XGBoost, LightGBM) trained on behavioral and transactional features
- Outputs credit scores (AAA to D) with SHAP/LIME-based explainability

### 3. ECL Forecasting (IFRS 9)
- Computes ECL = PD × LGD × EAD
- Ingests macroeconomic data (IMF, OECD)
- Runs baseline, optimistic, and pessimistic scenarios

### 4. Customer Segmentation & Monitoring
- Behavioral clustering (KMeans, DBSCAN)
- Risk profiling and real-time fraud/compliance flagging

### 5. Automated Compliance Reporting
- IFRS 9, Fair Lending, and GDPR outputs
- GPT-generated summaries + regulator-ready formats (PDF, JSON, CSV)

### 6. Real-Time Dashboard
- React frontend with D3.js/Plotly visualizations
- Live updates on credit scores, ECL, client segments, alerts

### 7. Workflow Automation
- n8n/Airflow pipelines from ingestion to decision
- Rasa/Dialogflow assistant for simulation and data augmentation

### 8. Security & Access Control
- RBAC/ABAC, AES-256 encryption, TLS 1.3
- GDPR, PCI DSS, SOC 2 compliant

---

## 📁 Datasets Used

### Financial Documents (OCR/NLP)
- SROIE, FUNSD, RVL-CDIP, DocBank
- Synthetic PDFs (bank statements, credit reports) with templating

### Credit Scoring
- German Credit Dataset, LendingClub, GiveMeSomeCredit, FICO Challenge

### ECL Forecasting
- IMF, OECD macroeconomic indicators
- Synthetic LGD/EAD/PD data

### Segmentation
- Bank Marketing, Credit Card Fraud, Retail Transaction Data
- Simulated time-series of customer behavior

---

## 🛠 Tech Stack

- **Backend:** FastAPI, Python, XGBoost, LightGBM, SHAP, LIME, spaCy, FinBERT
- **Frontend:** React.js, D3.js, Plotly.js
- **Automation:** n8n, Apache Airflow
- **Security:** TLS 1.3, AES-256, RBAC, audit logging
- **DevOps:** Docker, GitHub Actions

---

## 📦 Folder Structure

```plaintext
.
├── data/                     # Datasets and data generators
├── models/                   # Training scripts and serialized models
├── services/
│   ├── ocr_nlp/              # OCR + NLP pipeline
│   ├── scoring/              # Credit scoring logic
│   ├── ecl_forecasting/      # IFRS 9 ECL computation
│   ├── compliance/           # Reporting logic
├── dashboard/                # React frontend
├── workflows/                # n8n or Airflow DAGs
├── docs/                     # Architecture, data schema, reports
└── README.md
