"""
Budget Analysis Visualization Module
Handles creation of budget and savings analysis charts
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List


class BudgetVisualizer:
    """Create comprehensive budget analysis visualizations"""
    
    def __init__(self):
        """Initialize the budget visualizer"""
        pass
    
    def create_budget_analysis_charts(self, financial_context: Dict[str, Any], 
                                    budget_analysis: Dict[str, Any], 
                                    output_dir: Path) -> Dict[str, Any]:
        """
        Create comprehensive budget analysis visualizations
        
        Args:
            financial_context: Dictionary with financial data
            budget_analysis: Dictionary with budget analysis results
            output_dir: Directory to save visualization files
            
        Returns:
            Dictionary with budget visualization context
        """
        print("✅ Budget visualization libraries loaded")
        
        # Get budget data
        income = financial_context['total_income']
        expenses = financial_context['total_expenses']
        savings = financial_context['remaining_budget']
        savings_rate = financial_context['savings_rate']
        categories = financial_context['category_breakdown']
        
        print(f"\n💼 Budget Overview:")
        print(f"   💰 Income: ${income:,.2f}")
        print(f"   💸 Expenses: ${expenses:,.2f}")
        print(f"   💵 Savings: ${savings:,.2f}")
        print(f"   📊 Savings Rate: {savings_rate:.1f}%")
        
        charts_created = []
        
        # 1. Income vs Expenses Comparison
        print("\n🎨 Creating income vs expenses comparison...")
        fig_income_exp = self._create_income_expenses_chart(income, expenses, savings)
        fig_income_exp.show()
        charts_created.append(('income_vs_expenses.html', fig_income_exp))
        
        # 2. Budget Allocation Donut Chart
        print("   📊 Creating budget allocation donut chart...")
        fig_donut = self._create_budget_allocation_chart(categories, savings, income)
        fig_donut.show()
        charts_created.append(('budget_allocation.html', fig_donut))
        
        # 3. Savings Rate Analysis
        print("   📈 Creating savings rate analysis...")
        fig_gauge = self._create_savings_gauge(savings_rate)
        fig_gauge.show()
        charts_created.append(('savings_rate_gauge.html', fig_gauge))
        
        # 4. Category Spending Waterfall Chart
        print("   🌊 Creating spending waterfall chart...")
        fig_waterfall = self._create_waterfall_chart(income, categories, savings)
        fig_waterfall.show()
        charts_created.append(('budget_waterfall.html', fig_waterfall))
        
        # 5. Financial Health Dashboard
        print("   📊 Creating financial health dashboard...")
        fig_dashboard = self._create_financial_dashboard(income, expenses, savings, 
                                                       savings_rate, categories)
        fig_dashboard.show()
        charts_created.append(('financial_dashboard.html', fig_dashboard))
        
        # Save all charts
        saved_files = self._save_budget_charts(charts_created, output_dir)
        
        # Calculate financial health metrics
        expense_ratio = (expenses / income) * 100 if income > 0 else 0
        target_savings = income * 0.20
        health_metrics = self._calculate_health_metrics(
            savings_rate, expense_ratio, categories, savings, target_savings
        )
        
        # Create budget visualization context
        budget_viz_context = {
            'budget_analysis': {
                'income': income,
                'expenses': expenses,
                'savings': savings,
                'savings_rate': savings_rate,
                'expense_ratio': expense_ratio,
                'target_savings': target_savings,
                'savings_vs_target': savings / target_savings if target_savings > 0 else 0,
                'category_breakdown': categories,
                'top_expense_category': max(categories, key=categories.get) if categories else None,
                'budget_charts_generated': len(saved_files),
                'budget_visualization_files': saved_files,
                **health_metrics
            }
        }
        
        self._print_summary(savings_rate, len(saved_files), health_metrics)
        
        return budget_viz_context
    
    def _create_income_expenses_chart(self, income: float, expenses: float, 
                                    savings: float) -> go.Figure:
        """Create income vs expenses comparison chart"""
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Income',
            x=['Monthly Financial Summary'],
            y=[income],
            marker_color='#2E8B57',
            text=[f'${income:,.2f}'],
            textposition='inside',
            textfont=dict(color='white', size=14)
        ))
        
        fig.add_trace(go.Bar(
            name='Expenses',
            x=['Monthly Financial Summary'],
            y=[expenses],
            marker_color='#DC143C',
            text=[f'${expenses:,.2f}'],
            textposition='inside',
            textfont=dict(color='white', size=14)
        ))
        
        fig.add_trace(go.Bar(
            name='Savings',
            x=['Monthly Financial Summary'],
            y=[savings],
            marker_color='#4169E1',
            text=[f'${savings:,.2f}'],
            textposition='inside',
            textfont=dict(color='white', size=14)
        ))
        
        fig.update_layout(
            title='💰 Income vs Expenses vs Savings',
            yaxis_title='Amount ($)',
            font=dict(size=12),
            height=500,
            barmode='group',
            margin=dict(t=80, b=60, l=80, r=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return fig
    
    def _create_budget_allocation_chart(self, categories: Dict[str, float], 
                                      savings: float, income: float) -> go.Figure:
        """Create budget allocation donut chart"""
        allocation_data = categories.copy()
        allocation_data['Savings'] = savings
        
        fig = go.Figure(data=[go.Pie(
            labels=list(allocation_data.keys()), 
            values=list(allocation_data.values()),
            hole=0.4,
            marker_colors=px.colors.qualitative.Set3
        )])
        
        fig.update_traces(
            textposition='inside', 
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Amount: $%{value:,.2f}<br>Percentage: %{percent}<extra></extra>'
        )
        
        fig.update_layout(
            title='🥧 Budget Allocation (Including Savings)',
            font=dict(size=12),
            height=500,
            margin=dict(t=80, b=60, l=80, r=80),
            annotations=[dict(text=f'Total<br>${income:,.2f}', x=0.5, y=0.5, font_size=16, showarrow=False)]
        )
        
        return fig
    
    def _create_savings_gauge(self, savings_rate: float) -> go.Figure:
        """Create savings rate gauge chart"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = savings_rate,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "💾 Savings Rate %", 'font': {'size': 20}},
            delta = {'reference': 20, 'suffix': '%'},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 10], 'color': "lightgray"},
                    {'range': [10, 20], 'color': "yellow"},
                    {'range': [20, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 20
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            font={'color': "darkblue", 'family': "Arial"},
            margin=dict(t=80, b=60, l=80, r=80)
        )
        
        return fig
    
    def _create_waterfall_chart(self, income: float, categories: Dict[str, float], 
                              savings: float) -> go.Figure:
        """Create budget waterfall chart"""
        waterfall_x = ['Income'] + list(categories.keys()) + ['Net Savings']
        waterfall_y = [income] + [-abs(val) for val in categories.values()] + [savings]
        waterfall_text = [f'${abs(val):,.2f}' for val in waterfall_y]
        
        fig = go.Figure(go.Waterfall(
            name="Budget Flow",
            orientation="v",
            measure=["absolute"] + ["relative"] * len(categories) + ["total"],
            x=waterfall_x,
            textposition="outside",
            text=waterfall_text,
            y=waterfall_y,
            connector={"line":{"color":"rgb(63, 63, 63)"}},
            increasing={"marker":{"color":"#2E8B57"}},
            decreasing={"marker":{"color":"#DC143C"}},
            totals={"marker":{"color":"#4169E1"}}
        ))
        
        fig.update_layout(
            title="🌊 Budget Flow Analysis",
            font=dict(size=12),
            height=500,
            margin=dict(t=80, b=100, l=80, r=20),
            xaxis_title="Categories",
            yaxis_title="Amount ($)"
        )
        
        return fig
    
    def _create_financial_dashboard(self, income: float, expenses: float, savings: float,
                                  savings_rate: float, categories: Dict[str, float]) -> go.Figure:
        """Create comprehensive financial health dashboard"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Monthly Budget', 'Spending Categories', 'Savings Progress', 'Financial Ratios'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Budget comparison
        fig.add_trace(
            go.Bar(x=['Budget'], y=[income], name='Income', marker_color='green'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=['Budget'], y=[expenses], name='Expenses', marker_color='red'),
            row=1, col=1
        )
        
        # Category pie
        fig.add_trace(
            go.Pie(labels=list(categories.keys()), values=list(categories.values()), name="Categories"),
            row=1, col=2
        )
        
        # Savings progress (target 20%)
        target_savings = income * 0.20
        fig.add_trace(
            go.Scatter(x=['Current', 'Target'], y=[savings, target_savings], 
                      mode='markers+lines', name='Savings', marker_size=10),
            row=2, col=1
        )
        
        # Financial ratios
        expense_ratio = (expenses / income) * 100 if income > 0 else 0
        fig.add_trace(
            go.Bar(x=['Expense Ratio', 'Savings Rate'], 
                  y=[expense_ratio, savings_rate], 
                  name='Ratios (%)', marker_color=['orange', 'blue']),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="📊 Financial Health Dashboard",
            showlegend=False
        )
        
        return fig
    
    def _save_budget_charts(self, charts_created: List, output_dir: Path) -> List[str]:
        """Save budget visualization charts"""
        print("\n💾 Saving budget visualizations...")
        budget_viz_dir = output_dir / 'visualizations' / 'budget'
        budget_viz_dir.mkdir(parents=True, exist_ok=True)
        
        budget_files = []
        for filename, fig in charts_created:
            try:
                file_path = budget_viz_dir / filename
                fig.write_html(str(file_path))
                budget_files.append(filename)
            except Exception as e:
                print(f"   ⚠️ Could not save {filename}: {e}")
        
        print(f"✅ Saved {len(budget_files)} budget visualization files:")
        for file in budget_files:
            print(f"   📄 {file}")
        
        return budget_files
    
    def _calculate_health_metrics(self, savings_rate: float, expense_ratio: float,
                                categories: Dict[str, float], savings: float,
                                target_savings: float) -> Dict[str, Any]:
        """Calculate financial health metrics and assessment"""
        health_score = 0
        health_assessment = []
        
        if savings_rate >= 20:
            health_score += 40
            health_assessment.append("✅ Excellent savings rate")
        elif savings_rate >= 10:
            health_score += 25
            health_assessment.append("⚠️ Good savings rate, aim for 20%")
        else:
            health_score += 10
            health_assessment.append("❌ Low savings rate, needs improvement")
        
        if expense_ratio <= 80:
            health_score += 30
            health_assessment.append("✅ Healthy expense ratio")
        elif expense_ratio <= 90:
            health_score += 20
            health_assessment.append("⚠️ Moderate expense ratio")
        else:
            health_score += 10
            health_assessment.append("❌ High expense ratio")
        
        if len(categories) <= 8:
            health_score += 20
            health_assessment.append("✅ Well-organized spending categories")
        else:
            health_score += 10
            health_assessment.append("⚠️ Many spending categories")
        
        if savings > 0:
            health_score += 10
            health_assessment.append("✅ Positive cash flow")
        else:
            health_assessment.append("❌ Negative cash flow")
        
        return {
            'health_score': health_score,
            'health_assessment': health_assessment
        }
    
    def _print_summary(self, savings_rate: float, charts_count: int, 
                      health_metrics: Dict[str, Any]) -> None:
        """Print budget visualization summary"""
        print(f"\n📊 Budget Visualization Summary:")
        print(f"   💰 Income vs Expenses analyzed")
        print(f"   🥧 Budget allocation visualized")
        print(f"   📈 Savings rate: {savings_rate:.1f}% (Target: 20%)")
        print(f"   🏥 Financial Health Score: {health_metrics['health_score']}/100")
        print(f"   🎨 Generated {charts_count} interactive budget charts")
        
        print(f"\n🏥 Financial Health Assessment:")
        for assessment in health_metrics['health_assessment']:
            print(f"   {assessment}")
        
        print(f"\n✅ Budget visualization complete! Enhanced context ready for RAG model.")
    
    def create_basic_budget_summary(self, income: float, expenses: float, 
                                  savings: float, savings_rate: float, 
                                  categories: Dict[str, float]) -> Dict[str, Any]:
        """Create basic budget summary when advanced visualization fails"""
        print("💡 Using basic budget summary...")
        
        print(f"\n💼 Budget Summary:")
        print(f"   Income: ${income:,.2f}")
        print(f"   Expenses: ${expenses:,.2f}")
        print(f"   Savings: ${savings:,.2f} ({savings_rate:.1f}%)")
        print(f"\n🏷️ Category Breakdown:")
        for cat, amount in categories.items():
            percentage = (amount / income * 100) if income > 0 else 0
            print(f"   {cat}: ${amount:,.2f} ({percentage:.1f}%)")
        
        return {'basic_mode': True, 'charts_generated': 0}
