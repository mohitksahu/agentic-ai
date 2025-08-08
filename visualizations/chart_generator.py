import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime, timedelta
import math

class ChartGenerator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Color schemes for different chart types
        self.color_schemes = {
            'category': px.colors.qualitative.Set3,
            'trend': px.colors.sequential.Blues,
            'comparison': px.colors.qualitative.Pastel,
            'heatmap': 'RdYlBu_r'
        }
        
        # Chart themes
        self.themes = {
            'default': 'plotly_white',
            'dark': 'plotly_dark',
            'minimal': 'simple_white',
            'presentation': 'presentation'
        }

    def create_comprehensive_dashboard(self, df: pd.DataFrame, 
                                    monthly_income: Optional[float] = None) -> Dict[str, Any]:
        """Create a comprehensive financial dashboard with multiple visualizations"""
        try:
            self.logger.info(f"Creating comprehensive dashboard for {len(df)} transactions")
            
            charts = {}
            
            # 1. Overview Charts
            charts.update(self._create_overview_charts(df))
            
            # 2. Spending Analysis Charts
            charts.update(self._create_spending_analysis_charts(df))
            
            # 3. Time-based Analysis Charts
            charts.update(self._create_time_analysis_charts(df))
            
            # 4. Budget Analysis Charts (if income provided)
            if monthly_income:
                charts.update(self._create_budget_analysis_charts(df, monthly_income))
            
            # 5. Advanced Analytics Charts
            charts.update(self._create_advanced_analytics_charts(df))
            
            return charts
            
        except Exception as e:
            self.logger.error(f"Error creating comprehensive dashboard: {str(e)}")
            raise

    def create_custom_chart(self, df: pd.DataFrame, chart_type: str, 
                          **kwargs) -> go.Figure:
        """Create custom charts based on type and parameters"""
        try:
            chart_methods = {
                'line': self._create_line_chart,
                'bar': self._create_bar_chart,
                'pie': self._create_pie_chart,
                'scatter': self._create_scatter_chart,
                'box': self._create_box_plot,
                'violin': self._create_violin_plot,
                'heatmap': self._create_heatmap,
                'histogram': self._create_histogram,
                'area': self._create_area_chart,
                'waterfall': self._create_waterfall_chart,
                'sunburst': self._create_sunburst_chart,
                'treemap': self._create_treemap,
                'funnel': self._create_funnel_chart
            }
            
            if chart_type not in chart_methods:
                raise ValueError(f"Unsupported chart type: {chart_type}")
            
            return chart_methods[chart_type](df, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Error creating custom chart: {str(e)}")
            raise

    def _create_overview_charts(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create overview charts for financial summary"""
        charts = {}
        
        # Monthly spending trend
        charts['monthly_trend'] = self._create_monthly_trend(df)
        
        # Category breakdown pie chart
        charts['category_pie'] = self._create_category_pie(df)
        
        # Top categories bar chart
        charts['top_categories'] = self._create_top_categories_bar(df)
        
        # Weekly spending pattern
        charts['weekly_pattern'] = self._create_weekly_pattern(df)
        
        return charts

    def _create_spending_analysis_charts(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create detailed spending analysis charts"""
        charts = {}
        
        # Category comparison over time
        charts['category_timeline'] = self._create_category_timeline(df)
        
        # Daily spending distribution
        charts['daily_distribution'] = self._create_daily_distribution(df)
        
        # Amount distribution histogram
        charts['amount_histogram'] = self._create_amount_histogram(df)
        
        # Spending volatility
        charts['spending_volatility'] = self._create_volatility_chart(df)
        
        return charts

    def _create_time_analysis_charts(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create time-based analysis charts"""
        charts = {}
        
        # Monthly category heatmap
        charts['monthly_heatmap'] = self._create_monthly_category_heatmap(df)
        
        # Day of week analysis
        charts['weekday_analysis'] = self._create_weekday_analysis(df)
        
        # Cumulative spending
        charts['cumulative_spending'] = self._create_cumulative_spending(df)
        
        # Seasonal analysis
        charts['seasonal_analysis'] = self._create_seasonal_analysis(df)
        
        return charts

    def _create_budget_analysis_charts(self, df: pd.DataFrame, 
                                     monthly_income: float) -> Dict[str, go.Figure]:
        """Create budget-related analysis charts"""
        charts = {}
        
        # Budget vs actual spending
        charts['budget_comparison'] = self._create_budget_comparison(df, monthly_income)
        
        # Savings rate over time
        charts['savings_rate'] = self._create_savings_rate_chart(df, monthly_income)
        
        # Category budget allocation
        charts['budget_allocation'] = self._create_budget_allocation(df, monthly_income)
        
        return charts

    def _create_advanced_analytics_charts(self, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """Create advanced analytics and insights charts"""
        charts = {}
        
        # Spending pattern correlation
        charts['correlation_matrix'] = self._create_correlation_matrix(df)
        
        # Anomaly detection
        charts['anomaly_detection'] = self._create_anomaly_chart(df)
        
        # Forecast spending
        charts['spending_forecast'] = self._create_forecast_chart(df)
        
        return charts

    # Individual chart creation methods
    def _create_monthly_trend(self, df: pd.DataFrame) -> go.Figure:
        """Enhanced monthly spending trend with moving average"""
        monthly_data = df.groupby(df['date'].dt.to_period('M')).agg({
            'amount': ['sum', 'count', 'mean']
        }).round(2)
        
        monthly_data.columns = ['total', 'count', 'average']
        monthly_data.index = monthly_data.index.astype(str)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Total spending line
        fig.add_trace(
            go.Scatter(x=monthly_data.index, y=monthly_data['total'],
                      mode='lines+markers', name='Total Spending',
                      line=dict(width=3, color='#1f77b4')),
            secondary_y=False
        )
        
        # Transaction count bars
        fig.add_trace(
            go.Bar(x=monthly_data.index, y=monthly_data['count'],
                   name='Transaction Count', opacity=0.3,
                   marker_color='#ff7f0e'),
            secondary_y=True
        )
        
        # Add trend line
        if len(monthly_data) > 2:
            z = np.polyfit(range(len(monthly_data)), monthly_data['total'], 1)
            trend_line = np.poly1d(z)(range(len(monthly_data)))
            fig.add_trace(
                go.Scatter(x=monthly_data.index, y=trend_line,
                          mode='lines', name='Trend',
                          line=dict(dash='dash', color='red')),
                secondary_y=False
            )
        
        fig.update_yaxes(title_text="Amount ($)", secondary_y=False)
        fig.update_yaxes(title_text="Number of Transactions", secondary_y=True)
        fig.update_xaxes(title_text="Month")
        fig.update_layout(
            title='📈 Monthly Spending Trend with Transaction Volume',
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig

    def _create_category_pie(self, df: pd.DataFrame) -> go.Figure:
        """Enhanced category pie chart with percentage and amount"""
        category_data = df.groupby('category')['amount'].sum().sort_values(ascending=False)
        
        # Group small categories into "Other"
        total_amount = category_data.sum()
        threshold = total_amount * 0.03  # 3% threshold
        
        main_categories = category_data[category_data >= threshold]
        small_categories = category_data[category_data < threshold]
        
        if len(small_categories) > 0:
            main_categories['Other'] = small_categories.sum()
        
        fig = go.Figure(data=[go.Pie(
            labels=main_categories.index,
            values=main_categories.values,
            hole=.4,
            textinfo='label+percent',
            textposition='outside',
            hovertemplate='<b>%{label}</b><br>' +
                         'Amount: $%{value:,.2f}<br>' +
                         'Percentage: %{percent}<br>' +
                         '<extra></extra>',
            marker=dict(colors=self.color_schemes['category'][:len(main_categories)])
        )])
        
        fig.update_layout(
            title='🥧 Spending Distribution by Category',
            template='plotly_white',
            showlegend=True
        )
        
        return fig

    def _create_line_chart(self, df: pd.DataFrame, x: str, y: str, 
                          color: Optional[str] = None, **kwargs) -> go.Figure:
        """Create customizable line chart"""
        fig = px.line(df, x=x, y=y, color=color,
                     title=kwargs.get('title', f'{y} over {x}'),
                     template=kwargs.get('theme', 'plotly_white'))
        return fig

    def _create_bar_chart(self, df: pd.DataFrame, x: str, y: str,
                         orientation: str = 'v', **kwargs) -> go.Figure:
        """Create customizable bar chart"""
        if orientation == 'h':
            fig = px.bar(df, x=y, y=x, orientation='h',
                        title=kwargs.get('title', f'{y} by {x}'),
                        template=kwargs.get('theme', 'plotly_white'))
        else:
            fig = px.bar(df, x=x, y=y,
                        title=kwargs.get('title', f'{y} by {x}'),
                        template=kwargs.get('theme', 'plotly_white'))
        return fig

    def _create_scatter_chart(self, df: pd.DataFrame, x: str, y: str,
                             size: Optional[str] = None, color: Optional[str] = None,
                             **kwargs) -> go.Figure:
        """Create customizable scatter plot"""
        fig = px.scatter(df, x=x, y=y, size=size, color=color,
                        title=kwargs.get('title', f'{y} vs {x}'),
                        template=kwargs.get('theme', 'plotly_white'))
        return fig

    def _create_box_plot(self, df: pd.DataFrame, x: str, y: str, **kwargs) -> go.Figure:
        """Create box plot for distribution analysis"""
        fig = px.box(df, x=x, y=y,
                    title=kwargs.get('title', f'{y} Distribution by {x}'),
                    template=kwargs.get('theme', 'plotly_white'))
        return fig

    def _create_heatmap(self, df: pd.DataFrame, x: str, y: str, z: str,
                       **kwargs) -> go.Figure:
        """Create heatmap visualization"""
        pivot_data = df.pivot_table(values=z, index=y, columns=x, aggfunc='sum')
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale=kwargs.get('colorscale', 'RdYlBu_r'),
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=kwargs.get('title', f'{z} Heatmap: {y} vs {x}'),
            template=kwargs.get('theme', 'plotly_white')
        )
        
        return fig

    def _create_histogram(self, df: pd.DataFrame, x: str, **kwargs) -> go.Figure:
        """Create histogram for distribution analysis"""
        fig = px.histogram(df, x=x, 
                          title=kwargs.get('title', f'{x} Distribution'),
                          template=kwargs.get('theme', 'plotly_white'))
        return fig

    def _create_area_chart(self, df: pd.DataFrame, x: str, y: str,
                          color: Optional[str] = None, **kwargs) -> go.Figure:
        """Create area chart"""
        fig = px.area(df, x=x, y=y, color=color,
                     title=kwargs.get('title', f'{y} Area Chart'),
                     template=kwargs.get('theme', 'plotly_white'))
        return fig

    def _create_sunburst_chart(self, df: pd.DataFrame, path: List[str],
                              values: str, **kwargs) -> go.Figure:
        """Create sunburst chart for hierarchical data"""
        fig = px.sunburst(df, path=path, values=values,
                         title=kwargs.get('title', 'Sunburst Chart'),
                         template=kwargs.get('theme', 'plotly_white'))
        return fig

    def _create_treemap(self, df: pd.DataFrame, path: List[str],
                       values: str, **kwargs) -> go.Figure:
        """Create treemap for hierarchical data"""
        fig = px.treemap(df, path=path, values=values,
                        title=kwargs.get('title', 'Treemap Visualization'),
                        template=kwargs.get('theme', 'plotly_white'))
        return fig

    def _create_waterfall_chart(self, df: pd.DataFrame, x: str, y: str,
                               **kwargs) -> go.Figure:
        """Create waterfall chart for cumulative analysis"""
        fig = go.Figure(go.Waterfall(
            name="",
            orientation="v",
            measure=["relative"] * (len(df) - 1) + ["total"],
            x=df[x],
            textposition="outside",
            text=[f"${val:,.0f}" for val in df[y]],
            y=df[y],
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title=kwargs.get('title', 'Waterfall Chart'),
            template=kwargs.get('theme', 'plotly_white')
        )
        
        return fig

    def _create_violin_plot(self, df: pd.DataFrame, x: str, y: str,
                           **kwargs) -> go.Figure:
        """Create violin plot for distribution analysis"""
        fig = px.violin(df, x=x, y=y,
                       title=kwargs.get('title', f'{y} Distribution by {x}'),
                       template=kwargs.get('theme', 'plotly_white'))
        return fig

    def _create_funnel_chart(self, df: pd.DataFrame, x: str, y: str,
                            **kwargs) -> go.Figure:
        """Create funnel chart"""
        fig = go.Figure(go.Funnel(
            y=df[x],
            x=df[y],
            textinfo="value+percent initial"
        ))
        
        fig.update_layout(
            title=kwargs.get('title', 'Funnel Chart'),
            template=kwargs.get('theme', 'plotly_white')
        )
        
        return fig

    # Additional helper methods for complex charts
    def _create_monthly_category_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create monthly spending heatmap by category"""
        monthly_category = df.groupby([
            df['date'].dt.to_period('M'),
            'category'
        ])['amount'].sum().unstack(fill_value=0)
        
        fig = go.Figure(data=go.Heatmap(
            z=monthly_category.values,
            x=monthly_category.columns,
            y=[str(period) for period in monthly_category.index],
            colorscale='Blues',
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='🔥 Monthly Spending Heatmap by Category',
            xaxis_title='Category',
            yaxis_title='Month',
            template='plotly_white'
        )
        
        return fig

    def save_charts(self, charts: Dict[str, go.Figure], output_dir: str,
                   formats: List[str] = ['html']) -> Dict[str, str]:
        """Save charts in multiple formats"""
        saved_files = {}
        
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            for name, fig in charts.items():
                for fmt in formats:
                    filename = f"{name}.{fmt}"
                    filepath = os.path.join(output_dir, filename)
                    
                    if fmt == 'html':
                        fig.write_html(filepath)
                    elif fmt == 'png':
                        fig.write_image(filepath)
                    elif fmt == 'pdf':
                        fig.write_image(filepath)
                    elif fmt == 'svg':
                        fig.write_image(filepath)
                    
                    saved_files[f"{name}_{fmt}"] = filepath
            
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Error saving charts: {str(e)}")
            raise

    # Legacy compatibility method
    def create_spending_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Legacy method for backward compatibility"""
        return {
            'monthly_trend': self._create_monthly_trend(df),
            'category_breakdown': self._create_category_pie(df),
            'daily_pattern': self._create_daily_distribution(df)
        }
