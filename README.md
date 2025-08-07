# Agentic AI Financial Chatbot

A locally-run AI financial assistant that helps with household budgeting and financial management using multiple LLMs in an agentic framework.

## Features

- CSV and PDF file processing for financial data
- Multi-LLM architecture (LLaMA, Qwen, Mistral) for enhanced accuracy
- Local execution with no cloud dependencies
- Budgeting analysis and recommendations
- Spending pattern visualization
- Private and secure - no login or cloud storage required

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download required LLM models:
- LLaMA
- Qwen
- Mistral

3. Run the notebook:
```bash
jupyter notebook main.ipynb
```

## Project Structure

```
agentic-finance-bot/
├── main.ipynb                    # Main interface
├── requirements.txt             # Dependencies
├── README.md                   # Documentation
├── agents/                     # LangChain agents
├── llms/                       # LLM wrappers
├── parsers/                    # File parsers
├── analysis/                   # Financial analysis
├── visualizations/             # Charts
├── utils/                      # Utilities
└── data/                       # Sample data
```

## Usage

1. Launch the notebook interface
2. Upload your financial documents (CSV/PDF)
3. Interact with the chatbot to:
   - Analyze spending patterns
   - Get budgeting advice
   - View financial visualizations
   - Receive personalized recommendations

## Privacy

All processing is done locally on your machine. No data is sent to external servers.
