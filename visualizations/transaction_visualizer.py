"""
Transaction Data Visualization Module
Handles creation of charts and graphs for transaction analysis
"""

import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Optional
import plotly.io as pio


class TransactionVisualizer:
    """Create comprehensive visualizations from transaction data"""
    
    def __init__(self):
        """Initialize the visualizer with default settings"""
        # Set up visualization environment
        plt.style.use('default')
        sns.set_palette("husl")
        pio.renderers.default = "notebook"
    
    def create_all_charts(self, combined_financial_data: pd.DataFrame, 
                         output_dir: Path) -> Dict[str, Any]:
        """
        Create all transaction analysis charts
        
        Args:
            combined_financial_data: DataFrame with transaction data
            output_dir: Directory to save visualization files
            
        Returns:
            Dictionary with visualization context and results
        """
        print("🎨 Generating transaction analysis charts...")
        
        # Prepare data
        category_totals = combined_financial_data.groupby('category')['amount'].sum().abs()
        combined_financial_data['month'] = combined_financial_data['date'].dt.to_period('M')
        monthly_spending = combined_financial_data.groupby('month')['amount'].sum().abs()
        combined_financial_data['day_of_week'] = combined_financial_data['date'].dt.day_name()
        daily_spending = combined_financial_data.groupby('day_of_week')['amount'].sum().abs()
        
        # Reorder days of week
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        daily_spending = daily_spending.reindex(day_order, fill_value=0)
        
        charts_created = []
        
        # 1. Category Spending Pie Chart
        print("   📊 Creating category breakdown...")
        fig_pie = self._create_category_pie_chart(category_totals)
        fig_pie.show()
        charts_created.append(('category_pie_chart.html', fig_pie))
        
        # 2. Monthly Spending Trend
        print("   📈 Creating monthly spending trend...")
        fig_trend = self._create_monthly_trend_chart(monthly_spending)
        fig_trend.show()
        charts_created.append(('monthly_trend.html', fig_trend))
        
        # 3. Category Spending Bar Chart
        print("   📊 Creating category spending bar chart...")
        fig_bar = self._create_category_bar_chart(category_totals)
        fig_bar.show()
        charts_created.append(('category_bar_chart.html', fig_bar))
        
        # 4. Daily Spending Distribution
        print("   📊 Creating daily spending distribution...")
        fig_daily = self._create_daily_distribution_chart(daily_spending)
        fig_daily.show()
        charts_created.append(('daily_spending.html', fig_daily))
        
        # 5. Transaction Amount Distribution
        print("   📊 Creating transaction amount histogram...")
        fig_hist = self._create_amount_histogram(combined_financial_data)
        fig_hist.show()
        charts_created.append(('amount_histogram.html', fig_hist))
        
        # Save all charts
        saved_files = self._save_charts(charts_created, output_dir)
        
        # Create visualization context
        viz_context = self._create_viz_context(
            combined_financial_data, category_totals, saved_files
        )
        
        return viz_context
    
    def _create_category_pie_chart(self, category_totals: pd.Series) -> go.Figure:
        """Create category spending pie chart"""
        fig = px.pie(
            values=category_totals.values, 
            names=category_totals.index,
            title="💰 Spending by Category",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(
            font=dict(size=12),
            showlegend=True,
            height=500,
            margin=dict(t=80, b=20, l=20, r=20)
        )
        return fig
    
    def _create_monthly_trend_chart(self, monthly_spending: pd.Series) -> go.Figure:
        """Create monthly spending trend chart"""
        fig = px.line(
            x=monthly_spending.index.astype(str), 
            y=monthly_spending.values,
            title="📅 Monthly Spending Trend",
            labels={'x': 'Month', 'y': 'Amount ($)'},
            markers=True,
            color_discrete_sequence=['#1f77b4']
        )
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Spending Amount ($)",
            font=dict(size=12),
            height=400,
            margin=dict(t=80, b=60, l=80, r=20)
        )
        return fig
    
    def _create_category_bar_chart(self, category_totals: pd.Series) -> go.Figure:
        """Create category spending bar chart"""
        fig = px.bar(
            x=category_totals.index,
            y=category_totals.values,
            title="🏷️ Total Spending by Category",
            labels={'x': 'Category', 'y': 'Amount ($)'},
            color=category_totals.values,
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            xaxis_title="Category",
            yaxis_title="Total Amount ($)",
            font=dict(size=12),
            height=400,
            margin=dict(t=80, b=60, l=80, r=20),
            showlegend=False
        )
        return fig
    
    def _create_daily_distribution_chart(self, daily_spending: pd.Series) -> go.Figure:
        """Create daily spending distribution chart"""
        fig = px.bar(
            x=daily_spending.index,
            y=daily_spending.values,
            title="📆 Spending by Day of Week",
            labels={'x': 'Day of Week', 'y': 'Amount ($)'},
            color=daily_spending.values,
            color_continuous_scale='Blues'
        )
        fig.update_layout(
            xaxis_title="Day of Week",
            yaxis_title="Total Amount ($)",
            font=dict(size=12),
            height=400,
            margin=dict(t=80, b=60, l=80, r=20),
            showlegend=False
        )
        return fig
    
    def _create_amount_histogram(self, data: pd.DataFrame) -> go.Figure:
        """Create transaction amount histogram"""
        fig = px.histogram(
            data,
            x='amount',
            nbins=20,
            title="💸 Transaction Amount Distribution",
            labels={'amount': 'Transaction Amount ($)', 'count': 'Frequency'},
            color_discrete_sequence=['#ff7f0e']
        )
        fig.update_layout(
            xaxis_title="Transaction Amount ($)",
            yaxis_title="Frequency",
            font=dict(size=12),
            height=400,
            margin=dict(t=80, b=60, l=80, r=20)
        )
        return fig
    
    def _save_charts(self, charts_created: list, output_dir: Path) -> list:
        """Save all charts to HTML files"""
        print("\n💾 Saving visualizations to output folder...")
        viz_output_dir = output_dir / 'visualizations'
        viz_output_dir.mkdir(exist_ok=True)
        
        saved_files = []
        for filename, fig in charts_created:
            try:
                file_path = viz_output_dir / filename
                fig.write_html(str(file_path))
                saved_files.append(filename)
            except Exception as e:
                print(f"   ⚠️ Could not save {filename}: {e}")
        
        print(f"✅ Saved {len(saved_files)} visualization files:")
        for file in saved_files:
            print(f"   📄 {file}")
        
        return saved_files
    
    def _create_viz_context(self, data: pd.DataFrame, category_totals: pd.Series, 
                           saved_files: list) -> Dict[str, Any]:
        """Create visualization context for RAG model"""
        viz_context = {
            'total_transactions': len(data),
            'date_range': {
                'start': data['date'].min().strftime('%Y-%m-%d'),
                'end': data['date'].max().strftime('%Y-%m-%d')
            },
            'category_breakdown': category_totals.to_dict(),
            'top_spending_category': category_totals.idxmax(),
            'top_spending_amount': float(category_totals.max()),
            'average_transaction': float(data['amount'].abs().mean()),
            'total_amount': float(data['amount'].abs().sum()),
            'charts_generated': len(saved_files),
            'visualization_files': saved_files
        }
        
        print(f"\n📊 Visualization Summary:")
        print(f"   📈 {viz_context['total_transactions']} transactions analyzed")
        print(f"   📅 Date range: {viz_context['date_range']['start']} to {viz_context['date_range']['end']}")
        print(f"   🏷️ Top category: {viz_context['top_spending_category']} (${viz_context['top_spending_amount']:.2f})")
        print(f"   💰 Average transaction: ${viz_context['average_transaction']:.2f}")
        print(f"   🎨 Generated {viz_context['charts_generated']} interactive charts")
        
        return viz_context
    
    def create_basic_fallback_charts(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Create basic matplotlib charts as fallback"""
        print("💡 Using basic matplotlib fallback...")
        
        category_totals = data.groupby('category')['amount'].sum().abs()
        
        # Simple category pie chart
        plt.figure(figsize=(10, 6))
        plt.pie(category_totals.values, labels=category_totals.index, autopct='%1.1f%%')
        plt.title('Spending by Category')
        plt.axis('equal')
        plt.tight_layout()
        plt.show()
        
        # Simple bar chart
        plt.figure(figsize=(12, 6))
        category_totals.plot(kind='bar', color='skyblue')
        plt.title('Total Spending by Category')
        plt.xlabel('Category')
        plt.ylabel('Amount ($)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        print("✅ Basic visualizations generated")
        
        return {
            'total_transactions': len(data),
            'category_breakdown': category_totals.to_dict(),
            'charts_generated': 2,
            'basic_mode': True
        }
