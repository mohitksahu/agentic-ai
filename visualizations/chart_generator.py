import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any
import logging

class ChartGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def create_spending_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create visualizations for spending summary"""
        try:
            # Create monthly spending trend
            monthly_trend = self._create_monthly_trend(df)
            
            # Create category breakdown
            category_pie = self._create_category_breakdown(df)
            
            # Create daily spending pattern
            daily_pattern = self._create_daily_pattern(df)
            
            return {
                'monthly_trend': monthly_trend,
                'category_breakdown': category_pie,
                'daily_pattern': daily_pattern
            }
            
        except Exception as e:
            self.logger.error(f"Error generating charts: {str(e)}")
            raise

    def _create_monthly_trend(self, df: pd.DataFrame) -> go.Figure:
        """Create monthly spending trend line chart"""
        monthly_totals = df.groupby(pd.Grouper(key='date', freq='M'))['amount'].sum()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=monthly_totals.index,
            y=monthly_totals.values,
            mode='lines+markers',
            name='Monthly Spending'
        ))
        
        fig.update_layout(
            title='Monthly Spending Trend',
            xaxis_title='Month',
            yaxis_title='Total Spending ($)',
            showlegend=True
        )
        
        return fig

    def _create_category_breakdown(self, df: pd.DataFrame) -> go.Figure:
        """Create category breakdown pie chart"""
        category_totals = df.groupby('category')['amount'].sum()
        
        fig = go.Figure(data=[go.Pie(
            labels=category_totals.index,
            values=category_totals.values,
            hole=.3
        )])
        
        fig.update_layout(
            title='Spending by Category',
            showlegend=True
        )
        
        return fig

    def _create_daily_pattern(self, df: pd.DataFrame) -> go.Figure:
        """Create daily spending pattern box plot"""
        df['day_of_week'] = df['date'].dt.day_name()
        
        fig = go.Figure()
        
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                     'Friday', 'Saturday', 'Sunday']
        
        fig.add_trace(go.Box(
            x=df['day_of_week'],
            y=df['amount'],
            name='Daily Pattern',
            boxpoints='outliers'
        ))
        
        fig.update_layout(
            title='Daily Spending Patterns',
            xaxis_title='Day of Week',
            yaxis_title='Amount ($)',
            xaxis={'categoryorder': 'array',
                  'categoryarray': days_order}
        )
        
        return fig

    def save_charts(self, charts: Dict[str, go.Figure], output_dir: str):
        """Save all charts to HTML files"""
        try:
            for name, fig in charts.items():
                fig.write_html(f"{output_dir}/{name}.html")
        except Exception as e:
            self.logger.error(f"Error saving charts: {str(e)}")
            raise
