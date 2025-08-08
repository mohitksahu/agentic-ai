from typing import List, Dict, Any
import os
import pandas as pd
import logging

from analysis.budget_calculator import BudgetCalculator
from analysis.trend_analyzer import TrendAnalyzer
from visualizations.chart_generator import ChartGenerator
from utils.file_loader import FileLoader
from utils.helpers import Helpers

# Define a simple Tool class to maintain compatibility
class Tool:
    def __init__(self, name, func, description):
        self.name = name
        self.func = func
        self.description = description

# Define a simple memory class for compatibility
class ConversationBufferMemory:
    def __init__(self, memory_key="chat_history"):
        self.memory_key = memory_key
        self.buffer = ""
    
    def add_user_message(self, message):
        self.buffer += f"User: {message}\n"
    
    def add_ai_message(self, message):
        self.buffer += f"AI: {message}\n"

logger = logging.getLogger(__name__)

class FinancialAgent:
    def __init__(self, llm):
        self.llm = llm
        self.memory = ConversationBufferMemory(memory_key="chat_history")
        
        # Initialize components
        self.budget_calculator = BudgetCalculator()
        self.trend_analyzer = TrendAnalyzer()
        self.chart_generator = ChartGenerator()
        self.file_loader = FileLoader()
        
        # Setup tools and agent
        self.tools = self._setup_tools()
        
        # Initialize helper utilities
        Helpers.setup_logging()
        
        # Store current financial data
        self.current_data = None
        self.monthly_income = None

    def _setup_tools(self) -> List[Tool]:
        """Setup tools available to the agent"""
        tools = [
            Tool(
                name="Budget_Analysis",
                func=self._analyze_budget,
                description="Analyze budget and spending patterns. Input should be a dictionary with 'income' and 'data' keys."
            ),
            Tool(
                name="Document_Processing",
                func=self._process_document,
                description="Process financial documents (CSV/PDF). Input should be the file path."
            ),
            Tool(
                name="Generate_Visualization",
                func=self._generate_visualization,
                description="Create financial visualizations. Input should be the data dictionary."
            ),
            Tool(
                name="Trend_Analysis",
                func=self._analyze_trends,
                description="Analyze spending trends over time. Input should be the data dictionary."
            ),
            Tool(
                name="Set_Income",
                func=self._set_monthly_income,
                description="Set the monthly income for budget calculations. Input should be a number."
            )
        ]
        return tools

    def _setup_agent(self):
        """Create a simple agent executor replacement"""
        # This is now just a stub since we're implementing the agent logic directly
        # in the run method instead of using LangChain components
        return None

    def _analyze_budget(self, data: Dict[str, Any]) -> str:
        """Analyze budget data"""
        try:
            if not self.monthly_income:
                return "Please set monthly income first using Set_Income tool."
                
            if not self.current_data:
                return "Please load financial data first using Document_Processing tool."
            
            analysis = self.budget_calculator.calculate_budget(
                self.monthly_income,
                self.current_data['data']
            )
            
            summary = []
            for key, value in analysis['summary'].items():
                if isinstance(value, float):
                    value = Helpers.format_currency(value)
                summary.append(f"{key}: {value}")
            
            recommendations = "\n\nRecommendations:\n" + "\n".join(
                f"- {rec}" for rec in analysis['recommendations']
            )
            
            return "\n".join(summary) + recommendations
            
        except Exception as e:
            return f"Error in budget analysis: {str(e)}"

    def _process_document(self, file_path: str) -> Dict[str, Any]:
        """Process financial documents"""
        try:
            self.current_data = self.file_loader.load_file(file_path)
            summary = []
            for key, value in self.current_data['summary'].items():
                summary.append(f"{key}: {value}")
            return "\n".join(summary)
        except Exception as e:
            return f"Error processing document: {str(e)}"

    def _generate_visualization(self, data: Dict[str, Any]) -> str:
        """Generate financial visualizations"""
        try:
            if not self.current_data:
                return "Please load financial data first using Document_Processing tool."
            
            charts = self.chart_generator.create_spending_summary(self.current_data['data'])
            
            # Save charts
            output_dir = os.path.join(os.getcwd(), 'visualizations', 'output')
            Helpers.create_directory_if_not_exists(output_dir)
            self.chart_generator.save_charts(charts, output_dir)
            
            return f"Charts have been generated and saved to {output_dir}"
        except Exception as e:
            return f"Error generating visualizations: {str(e)}"

    def _analyze_trends(self, data: Dict[str, Any]) -> str:
        """Analyze spending trends"""
        try:
            if not self.current_data:
                return "Please load financial data first using Document_Processing tool."
            
            trends = self.trend_analyzer.analyze_trends(self.current_data['data'])
            
            return trends['summary']
        except Exception as e:
            return f"Error analyzing trends: {str(e)}"

    def _set_monthly_income(self, income: float) -> str:
        """Set monthly income"""
        try:
            self.monthly_income = float(income)
            return f"Monthly income set to {Helpers.format_currency(self.monthly_income)}"
        except ValueError:
            return "Please provide a valid number for monthly income"

    def run(self, query: str) -> str:
        """Run the agent with a user query"""
        try:
            # Add user query to memory
            self.memory.add_user_message(query)
            
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
                history_text = self.memory.buffer
                prompt = f"Financial assistant conversation:\n{history_text}\nBased on the above conversation, please provide a helpful response to the latest query: {query}"
                result = self.llm(prompt)
            
            # Add result to memory
            self.memory.add_ai_message(result)
            return result
            
        except Exception as e:
            error_msg = f"I encountered an error: {str(e)}\nPlease try rephrasing your question."
            self.memory.add_ai_message(error_msg)
            return error_msg
