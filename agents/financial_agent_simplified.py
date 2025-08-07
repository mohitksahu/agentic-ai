from typing import List, Dict, Any
import os
import logging
import pandas as pd

from analysis.budget_calculator import BudgetCalculator
from analysis.trend_analyzer import TrendAnalyzer
from visualizations.chart_generator import ChartGenerator
from utils.file_loader import FileLoader
from utils.helpers import Helpers

logger = logging.getLogger(__name__)

class FinancialAgent:
    """
    Simplified Financial Agent that directly uses the model wrappers
    without relying on LangChain components
    """
    def __init__(self, llm):
        self.llm = llm
        
        # Initialize components
        self.budget_calculator = BudgetCalculator()
        self.trend_analyzer = TrendAnalyzer()
        self.chart_generator = ChartGenerator()
        self.file_loader = FileLoader()
        
        # Setup tools
        self.tools = {
            "Budget_Analysis": self._analyze_budget,
            "Document_Processing": self._process_document,
            "Generate_Visualization": self._generate_visualization,
            "Trend_Analysis": self._analyze_trends,
            "Set_Income": self._set_monthly_income
        }
        
        # Initialize helper utilities
        Helpers.setup_logging()
        
        # Store current financial data and chat history
        self.current_data = None
        self.monthly_income = None
        self.chat_history = []
    
    def run(self, query: str) -> str:
        """
        Process a user query and return a response
        
        Args:
            query (str): User's query
            
        Returns:
            str: Agent's response
        """
        try:
            # Add query to history
            self.chat_history.append(f"User: {query}")
            
            # Simple keyword-based tool selection
            result = None
            
            if "process" in query.lower() and "document" in query.lower():
                # Extract file path from query
                file_path = query.split("Process the document ")[-1].strip()
                result = self._process_document(file_path)
            
            elif "set" in query.lower() and "income" in query.lower():
                # Extract income amount from query
                try:
                    amount = float(''.join(filter(lambda i: i.isdigit() or i == '.', query)))
                    result = self._set_monthly_income(amount)
                except:
                    result = "Could not parse income amount. Please provide a valid number."
            
            elif "analyze" in query.lower() and "budget" in query.lower():
                if not self.current_data:
                    result = "Please load financial data first using 'Process the document' command."
                elif not self.monthly_income:
                    result = "Please set monthly income first using 'Set monthly income to X' command."
                else:
                    result = self._analyze_budget({"income": self.monthly_income, "data": self.current_data})
            
            elif "trend" in query.lower() or "analyze trends" in query.lower():
                if not self.current_data:
                    result = "Please load financial data first using 'Process the document' command."
                else:
                    result = self._analyze_trends(self.current_data)
            
            elif "visualization" in query.lower() or "chart" in query.lower():
                if not self.current_data:
                    result = "Please load financial data first using 'Process the document' command."
                else:
                    result = self._generate_visualization(self.current_data)
            
            else:
                # Default: just ask the LLM
                history_text = "\n".join(self.chat_history[-5:])  # Last 5 exchanges
                prompt = f"Financial assistant conversation:\n{history_text}\n\nBased on the above conversation, please provide a helpful response to the latest query."
                result = self.llm.generate(prompt)
            
            # Add result to history
            self.chat_history.append(f"Assistant: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"

    def _analyze_budget(self, data: Dict[str, Any]) -> str:
        """Analyze budget data"""
        try:
            income = data.get("income", self.monthly_income)
            financial_data = data.get("data", self.current_data)
            
            if not income:
                return "Please set monthly income first using 'Set monthly income to X' command."
                
            if not financial_data:
                return "Please load financial data first using 'Process the document' command."
            
            analysis = self.budget_calculator.calculate_budget(income, financial_data['data'])
            
            summary = []
            summary.append(f"Monthly Income: ${income:.2f}")
            summary.append(f"Total Expenses: ${analysis['total_expenses']:.2f}")
            summary.append(f"Remaining: ${analysis['remaining']:.2f}")
            
            for category, amount in analysis['by_category'].items():
                summary.append(f"{category}: ${amount:.2f} ({amount/analysis['total_expenses']*100:.1f}%)")
            
            return "\n".join(summary)
        except Exception as e:
            logger.error(f"Error in budget analysis: {str(e)}")
            return f"Error analyzing budget: {str(e)}"
    
    def _process_document(self, file_path: str) -> str:
        """Process financial document"""
        try:
            self.current_data = self.file_loader.load_file(file_path)
            return f"Successfully loaded data from {file_path}.\n{self.current_data['summary']}"
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return f"Error processing document: {str(e)}"
    
    def _generate_visualization(self, data: Dict[str, Any]) -> str:
        """Generate financial visualizations"""
        try:
            charts = self.chart_generator.create_charts(data['data'])
            return f"Created {len(charts)} charts. Use the view option to display them."
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            return f"Error generating visualization: {str(e)}"
    
    def _analyze_trends(self, data: Dict[str, Any]) -> str:
        """Analyze spending trends"""
        try:
            trends = self.trend_analyzer.analyze_trends(data['data'])
            
            summary = ["Spending Trends:"]
            for category, trend in trends['trends'].items():
                summary.append(f"{category}: {trend['description']}")
            
            if trends.get('anomalies'):
                summary.append("\nPotential anomalies detected:")
                for anomaly in trends['anomalies']:
                    summary.append(f"- {anomaly}")
            
            return "\n".join(summary)
        except Exception as e:
            logger.error(f"Error analyzing trends: {str(e)}")
            return f"Error analyzing trends: {str(e)}"
    
    def _set_monthly_income(self, amount: float) -> str:
        """Set monthly income"""
        try:
            self.monthly_income = float(amount)
            return f"Monthly income set to ${self.monthly_income:.2f}"
        except Exception as e:
            logger.error(f"Error setting income: {str(e)}")
            return f"Error setting income: {str(e)}"
