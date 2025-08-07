import pandas as pd
from typing import Dict, Any
import logging

class BudgetCalculator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def calculate_budget(self, income: float, expenses: pd.DataFrame) -> Dict[str, Any]:
        """Calculate budget based on income and expenses"""
        try:
            # Calculate total expenses by category
            category_totals = expenses.groupby('category')['amount'].sum()
            
            # Calculate overall metrics
            total_expenses = expenses['amount'].sum()
            remaining_budget = income - total_expenses
            
            # Calculate percentage of income for each category
            category_percentages = (category_totals / income * 100).round(2)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                income, 
                category_percentages,
                remaining_budget
            )
            
            return {
                'summary': {
                    'total_income': income,
                    'total_expenses': total_expenses,
                    'remaining_budget': remaining_budget,
                    'savings_rate': ((income - total_expenses) / income * 100).round(2)
                },
                'category_breakdown': category_totals.to_dict(),
                'category_percentages': category_percentages.to_dict(),
                'recommendations': recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating budget: {str(e)}")
            raise

    def _generate_recommendations(self, 
                                income: float, 
                                category_percentages: pd.Series,
                                remaining_budget: float) -> list:
        """Generate budget recommendations based on spending patterns"""
        recommendations = []
        
        # Define ideal category percentages
        ideal_percentages = {
            'Housing': 30,
            'Transportation': 15,
            'Food': 15,
            'Utilities': 10,
            'Insurance': 10,
            'Savings': 20
        }
        
        # Compare actual vs ideal percentages
        for category, actual_pct in category_percentages.items():
            ideal_pct = ideal_percentages.get(category)
            if ideal_pct and actual_pct > ideal_pct:
                recommendations.append(
                    f"Consider reducing {category} expenses from {actual_pct}% to {ideal_pct}%"
                )
        
        # Check savings rate
        if remaining_budget < 0:
            recommendations.append("Warning: You are spending more than your income")
        elif (remaining_budget / income * 100) < 20:
            recommendations.append("Try to increase your savings rate to at least 20%")
        
        return recommendations
