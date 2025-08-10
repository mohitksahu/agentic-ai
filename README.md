# 💰 Agentic AI Financial System

A comprehensive, locally-run financial analysis system that processes your CSV data and provides AI-powered insights for budgeting and financial management.

## ✨ Features

- **📊 Smart CSV Processing** - Automatic column detection and data validation
- **💡 AI-Powered Q&A** - Ask questions about your financial data in natural language
- **📈 Interactive Visualizations** - Professional charts and spending analysis
- **🔒 Privacy-First** - All processing happens locally, your data never leaves your computer
- **⚡ Production-Ready** - Clean, optimized code with comprehensive error handling
- **🎯 User-Focused** - Works exclusively with your real financial data

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Jupyter Notebook or VS Code

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/mohitksahu/agentic-ai.git
cd agentic-ai
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Add your financial data:**
   - Place your CSV files in `data/input/` directory
   - Required CSV format: columns for date, amount, and category/description

4. **Run the analysis:**
```bash
jupyter notebook main.ipynb
```

## 📁 Project Structure

```
agentic-ai/
├── 📓 main.ipynb              # Complete financial analysis workflow
├── 🧪 test_setup.py           # System validation and testing
├── 📄 README.md               # This file
├── 📦 requirements.txt        # Python dependencies
├── 📜 LICENSE                 # MIT License
│
├── 📂 data/                   # Your financial data
│   ├── input/                # Put your CSV files here
│   └── output/               # Generated analysis and charts
│       └── visualizations/   # Chart outputs
│
├── 🔧 utils/                  # Core utilities
│   ├── environment_setup.py  # Environment configuration
│   ├── dependency_manager.py # Package management
│   ├── file_loader.py        # File loading with validation
│   └── system_status.py      # System monitoring
│
├── 📊 parsers/                # Data parsing modules
│   ├── csv_parser.py         # Smart CSV parsing
│   └── pdf_parser.py         # PDF financial statements
│
├── 🤖 agents/                 # AI agents and interfaces
│   ├── financial_agent.py    # Main financial Q&A agent
│   └── financial_qa_engine.py # RAG-based Q&A engine
│
├── 📈 analysis/               # Financial analysis modules
│   ├── budget_calculator.py  # Budget analysis and insights
│   └── trend_analyzer.py     # Trend analysis
│
├── 📊 visualizations/         # Chart generation
│   ├── budget_visualizer.py  # Budget and savings charts
│   ├── chart_generator.py    # General chart utilities
│   └── transaction_visualizer.py # Transaction analysis
│
└── 🧠 llms/                   # Language model integrations
    ├── distilbert_wrapper.py # DistilBERT integration
    ├── gpt2_wrapper.py       # GPT-2 integration
    └── ...                   # Other model wrappers
```

## 💡 How to Use

### 1. **Prepare Your Data**
- Export your bank transactions as CSV files
- Ensure columns for: date, amount, category/description
- Place files in `data/input/` directory

### 2. **Run the Analysis**
- Open `main.ipynb` in Jupyter Notebook or VS Code
- Run cells sequentially to:
  - Set up the environment
  - Load and process your CSV data
  - Generate budget analysis and insights
  - Create visualizations
  - Use the AI Q&A system

### 3. **Validate System**
```bash
python test_setup.py
```

### 4. **Ask Questions**
Use the built-in Q&A system to ask questions like:
- "What's my highest expense category?"
- "How much did I spend on groceries last month?"
- "What are some budget recommendations?"

## 🎯 CSV Format Requirements

Your CSV files should contain columns for:
- **Date**: Transaction dates (any common format)
- **Amount**: Transaction amounts (positive or negative numbers)
- **Category/Description**: Spending categories or descriptions

**Example CSV structure:**
```csv
date,amount,category
2025-01-01,100.50,Groceries
2025-01-02,-45.00,Gas
2025-01-03,25.75,Coffee
```

## 🔒 Privacy & Security

- **100% Local Processing** - No data sent to external servers
- **No Cloud Dependencies** - Everything runs on your machine  
- **Your Data Stays Yours** - Complete control over your financial information
- **No Sign-ups Required** - No accounts, no tracking, no external services

## 🛠️ System Validation

Run the test suite to ensure everything is working:

```bash
python test_setup.py
```

This validates:
- ✅ Directory structure
- ✅ Dependencies installed
- ✅ CSV file detection
- ✅ Data structure validation
- ✅ System readiness score

## 📋 Requirements

- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- jupyter or VS Code with Jupyter extension
- Optional: GPU for faster LLM processing

## 🤝 Contributing

This project is designed to be clean, modular, and easy to understand. Contributions are welcome!

## 📄 License

MIT License - see LICENSE file for details.

---

**🚀 Ready to analyze your finances? Add your CSV files to `data/input/` and run `main.ipynb`!**
