"""
RAG Financial Q&A Module
Handles intelligent question answering about financial data
"""

from typing import Dict, Any, Optional


class FinancialQAEngine:
    """Enhanced financial question answering engine with RAG capabilities"""
    
    def __init__(self, financial_agent=None):
        """
        Initialize the Q&A engine
        
        Args:
            financial_agent: Optional advanced financial agent for complex queries
        """
        self.financial_agent = financial_agent
        self.context_data = {}
        self.viz_context = {}
    
    def update_context(self, financial_context: Dict[str, Any], 
                      viz_context: Optional[Dict[str, Any]] = None) -> None:
        """
        Update the context data for question answering
        
        Args:
            financial_context: Financial analysis context
            viz_context: Visualization context (optional)
        """
        self.context_data = financial_context
        if viz_context:
            self.viz_context = viz_context
    
    def ask_question(self, question: str) -> str:
        """
        Ask a question about financial data
        
        Args:
            question: User's financial question
            
        Returns:
            Intelligent answer based on available context
        """
        if not self.context_data:
            return "⚠️ No financial analysis available. Please run the budget analysis first."
        
        # Try advanced agent first if available
        if self.financial_agent:
            try:
                answer = self.financial_agent.ask_question(question, self.context_data)
                return self._enhance_answer_with_viz_context(answer, question)
            except Exception as e:
                print(f"⚠️ Advanced agent failed: {e}, falling back to rule-based system")
        
        # Fallback to rule-based answers
        return self._rule_based_answer(question)
    
    def _rule_based_answer(self, question: str) -> str:
        """
        Generate rule-based answers for financial questions
        
        Args:
            question: User's question
            
        Returns:
            Generated answer based on rules and context
        """
        question_lower = question.lower()
        
        # Savings-related questions
        if self._is_savings_question(question_lower):
            return self._answer_savings_question(question_lower)
        
        # Spending-related questions
        elif self._is_spending_question(question_lower):
            return self._answer_spending_question(question_lower)
        
        # Budget-related questions
        elif self._is_budget_question(question_lower):
            return self._answer_budget_question(question_lower)
        
        # Health assessment questions
        elif self._is_health_question(question_lower):
            return self._answer_health_question(question_lower)
        
        # Improvement/advice questions
        elif self._is_advice_question(question_lower):
            return self._answer_advice_question(question_lower)
        
        # Visualization questions
        elif self._is_visualization_question(question_lower):
            return self._answer_visualization_question(question_lower)
        
        # Category-specific questions
        elif self._is_category_question(question_lower):
            return self._answer_category_question(question_lower)
        
        # Default response
        else:
            return self._get_default_response()
    
    def _is_savings_question(self, question: str) -> bool:
        """Check if question is about savings"""
        return any(word in question for word in ['save', 'saving', 'savings'])
    
    def _answer_savings_question(self, question: str) -> str:
        """Answer savings-related questions"""
        savings = self.context_data.get('remaining_budget', 0)
        savings_rate = self.context_data.get('savings_rate', 0)
        
        if savings_rate >= 20:
            return f"Great job! You're saving ${savings:.2f}/month ({savings_rate:.1f}% rate). This exceeds the recommended 20% savings rate."
        elif savings_rate >= 10:
            return f"You're saving ${savings:.2f}/month ({savings_rate:.1f}% rate). Try to increase to 20% for optimal financial health."
        else:
            return f"Your current savings rate is {savings_rate:.1f}% (${savings:.2f}/month). Aim for 20% by reducing expenses or increasing income."
    
    def _is_spending_question(self, question: str) -> bool:
        """Check if question is about spending"""
        return any(word in question for word in ['spend', 'expense', 'spending'])
    
    def _answer_spending_question(self, question: str) -> str:
        """Answer spending-related questions"""
        expenses = self.context_data.get('total_expenses', 0)
        categories = self.context_data.get('category_breakdown', {})
        
        if categories:
            top_category = max(categories, key=categories.get)
            top_amount = categories[top_category]
            
            # Add visualization context if available
            viz_info = ""
            if self.viz_context and 'total_transactions' in self.viz_context:
                viz_info = f" Based on analysis of {self.viz_context['total_transactions']} transactions, "
            
            return f"{viz_info}Your total monthly expenses are ${expenses:.2f}. Your biggest spending category is {top_category} at ${top_amount:.2f}."
        else:
            return f"Your total monthly expenses are ${expenses:.2f}."
    
    def _is_budget_question(self, question: str) -> bool:
        """Check if question is about budget"""
        return any(word in question for word in ['budget', 'income'])
    
    def _answer_budget_question(self, question: str) -> str:
        """Answer budget-related questions"""
        income = self.context_data.get('total_income', 0)
        expenses = self.context_data.get('total_expenses', 0)
        remaining = self.context_data.get('remaining_budget', 0)
        return f"Your monthly budget: ${income:.2f} income, ${expenses:.2f} expenses, ${remaining:.2f} remaining for savings/investments."
    
    def _is_health_question(self, question: str) -> bool:
        """Check if question is about financial health"""
        return any(word in question for word in ['healthy', 'health', 'good', 'bad'])
    
    def _answer_health_question(self, question: str) -> str:
        """Answer financial health questions"""
        savings_rate = self.context_data.get('savings_rate', 0)
        
        # Check if we have detailed health assessment from budget visualization
        if ('budget_analysis' in self.viz_context and 
            'health_score' in self.viz_context['budget_analysis']):
            
            health_score = self.viz_context['budget_analysis']['health_score']
            health_assessment = self.viz_context['budget_analysis']['health_assessment']
            
            assessment_text = "; ".join(health_assessment)
            return f"Your financial health score is {health_score}/100. {assessment_text}"
        
        # Fallback health assessment
        if savings_rate >= 20:
            return f"Your financial health looks excellent with a {savings_rate:.1f}% savings rate!"
        elif savings_rate >= 10:
            return f"Your financial health is decent with a {savings_rate:.1f}% savings rate. Room for improvement."
        else:
            return f"Your {savings_rate:.1f}% savings rate suggests room for financial health improvement. Aim for 20%."
    
    def _is_advice_question(self, question: str) -> bool:
        """Check if question is asking for advice"""
        return any(word in question for word in ['improve', 'better', 'advice', 'recommend'])
    
    def _answer_advice_question(self, question: str) -> str:
        """Answer advice/improvement questions"""
        recommendations = self.context_data.get('recommendations', [])
        if recommendations:
            return f"Here are your personalized recommendations: {'; '.join(recommendations[:3])}"
        else:
            return "Focus on increasing your savings rate to 20% and tracking your spending categories."
    
    def _is_visualization_question(self, question: str) -> bool:
        """Check if question is about charts or visualizations"""
        return any(word in question for word in ['chart', 'graph', 'visual', 'plot'])
    
    def _answer_visualization_question(self, question: str) -> str:
        """Answer visualization-related questions"""
        if not self.viz_context:
            return "No visualization data available. Please run the visualization cells first."
        
        charts_count = (self.viz_context.get('charts_generated', 0) + 
                       self.viz_context.get('budget_analysis', {}).get('budget_charts_generated', 0))
        
        return f"I've generated {charts_count} interactive charts for your financial data, including spending breakdowns, trends, and budget analysis. These visualizations help identify patterns in your financial behavior."
    
    def _is_category_question(self, question: str) -> bool:
        """Check if question is about specific categories"""
        return any(word in question for word in ['category', 'categories', 'groceries', 'rent', 'food', 'transport'])
    
    def _answer_category_question(self, question: str) -> str:
        """Answer category-specific questions"""
        categories = self.context_data.get('category_breakdown', {})
        income = self.context_data.get('total_income', 1)
        
        if not categories:
            return "No category breakdown available. Please ensure your CSV data includes category information."
        
        # Look for specific category mentions
        question_words = question.lower().split()
        category_found = None
        
        for category in categories.keys():
            if category.lower() in question.lower():
                category_found = category
                break
        
        if category_found:
            amount = categories[category_found]
            percentage = (amount / income * 100) if income > 0 else 0
            return f"Your {category_found} spending is ${amount:.2f} per month, which is {percentage:.1f}% of your income."
        
        # General category overview
        sorted_categories = sorted(categories.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_categories[:3]
        
        response = "Your top spending categories are: "
        category_details = []
        for cat, amount in top_3:
            percentage = (amount / income * 100) if income > 0 else 0
            category_details.append(f"{cat} ${amount:.2f} ({percentage:.1f}%)")
        
        return response + ", ".join(category_details)
    
    def _get_default_response(self) -> str:
        """Get default response for unrecognized questions"""
        available_topics = [
            "savings and savings rate",
            "spending and expenses", 
            "budget and income",
            "financial health assessment",
            "improvement recommendations",
            "spending categories",
            "charts and visualizations"
        ]
        
        return (f"I can help analyze your financial data. Try asking about: "
                f"{', '.join(available_topics)}. "
                f"For example: 'How much am I saving?' or 'What's my biggest expense?'")
    
    def _enhance_answer_with_viz_context(self, answer: str, question: str) -> str:
        """
        Enhance answers with visualization context when relevant
        
        Args:
            answer: Original answer
            question: Original question
            
        Returns:
            Enhanced answer with visualization context
        """
        if not self.viz_context:
            return answer
        
        # Add chart information for relevant questions
        if any(word in question.lower() for word in ['chart', 'visual', 'see', 'show']):
            charts_info = []
            
            if 'charts_generated' in self.viz_context:
                charts_info.append(f"{self.viz_context['charts_generated']} transaction charts")
            
            if ('budget_analysis' in self.viz_context and 
                'budget_charts_generated' in self.viz_context['budget_analysis']):
                budget_charts = self.viz_context['budget_analysis']['budget_charts_generated']
                charts_info.append(f"{budget_charts} budget analysis charts")
            
            if charts_info:
                chart_text = " and ".join(charts_info)
                answer += f" I've also created {chart_text} to visualize this information."
        
        return answer
    
    def run_interactive_session(self) -> None:
        """Run an interactive Q&A session"""
        if not self.context_data:
            print("⚠️ Financial Q&A not available.")
            print("   Please run the budget analysis cell first to generate financial context.")
            return
        
        print("🤖 Financial Q&A Ready!")
        print("=" * 30)
        print("Ask questions about your financial data.")
        print("Type 'quit' to exit, 'help' for examples.")
        print("=" * 30)
        
        while True:
            try:
                question = input("\n💬 Your question: ").strip()
                
                if question.lower() == 'quit':
                    print("👋 Q&A session ended!")
                    break
                
                if question.lower() == 'help':
                    self._show_help()
                    continue
                
                if not question:
                    continue
                
                # Get answer using the Q&A engine
                answer = self.ask_question(question)
                print(f"\n🤖 {answer}")
                
            except KeyboardInterrupt:
                print("\n👋 Q&A session interrupted!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                continue
    
    def _show_help(self) -> None:
        """Show help examples"""
        print("\n💡 Example questions:")
        print("   • How much am I saving?")
        print("   • What's my biggest expense?")
        print("   • Is my savings rate healthy?")
        print("   • How can I improve my budget?")
        print("   • What percentage goes to groceries?")
        print("   • Show me my spending categories")
        print("   • How do my charts look?")
    
    def create_demo_session(self, demo_context: Dict[str, Any]) -> None:
        """Create a demo Q&A session with sample data"""
        print("💡 Demo Mode - Example Q&A:")
        
        demo_questions = [
            "How much am I saving?",
            "What's my biggest expense?", 
            "Is my savings rate healthy?",
            "How can I improve my budget?"
        ]
        
        # Temporarily use demo context
        original_context = self.context_data
        self.context_data = demo_context
        
        for question in demo_questions:
            print(f"\n❓ {question}")
            answer = self._rule_based_answer(question.lower())
            print(f"🤖 {answer}")
        
        # Restore original context
        self.context_data = original_context
