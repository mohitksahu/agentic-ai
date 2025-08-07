from typing import List, Dict, Any
import os
import pandas as pd
from typing import List, Dict, Any
import os
import pandas as pd

from analysis.budget_calculator import BudgetCalculator
from analysis.trend_analyzer import TrendAnalyzer
from visualizations.chart_generator import ChartGenerator
from utils.file_loader import FileLoader
from utils.helpers import Helpers

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
        self.agent_executor = self._setup_agent()
        
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

    def _setup_agent(self) -> AgentExecutor:
        """Setup the agent executor"""
        
        # Create a simplified agent that just uses the LLM directly
        class SimpleLLMChain:
            def __init__(self, llm, memory):
                self.llm = llm
                self.memory = memory
                
            def run(self, query):
                context = self.memory.buffer
                prompt = f"Answer the question based on the context below.\n\nContext: {context}\n\nQuestion: {query}\n\nAnswer:"
                return self.llm(prompt)
        
        # Create a simple chain
        chain = SimpleLLMChain(self.llm, self.memory)
        
        # Create the agent
        agent = LLMSingleActionAgent(
            llm_chain=chain,
            tools=self.tools,
            max_iterations=3
        )
        
        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=self.tools,
            memory=self.memory
        )

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
            return self.agent_executor.run(query)
        except Exception as e:
            return f"I encountered an error: {str(e)}\nPlease try rephrasing your question."
