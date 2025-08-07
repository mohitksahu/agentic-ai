import pandas as pd
import numpy as np
from typing import Dict, Any
import logging
from datetime import datetime, timedelta

class TrendAnalyzer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def analyze_trends(self, transactions: pd.DataFrame) -> Dict[str, Any]:
        """Analyze spending trends from transaction data"""
        try:
            # Ensure date column is datetime
            transactions['date'] = pd.to_datetime(transactions['date'])
            
            # Calculate various trends
            monthly_trends = self._calculate_monthly_trends(transactions)
            category_trends = self._calculate_category_trends(transactions)
            seasonal_patterns = self._detect_seasonal_patterns(transactions)
            anomalies = self._detect_anomalies(transactions)
            
            return {
                'monthly_trends': monthly_trends,
                'category_trends': category_trends,
                'seasonal_patterns': seasonal_patterns,
                'anomalies': anomalies,
                'summary': self._generate_trend_summary(
                    monthly_trends,
                    category_trends,
                    seasonal_patterns,
                    anomalies
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trends: {str(e)}")
            raise

    def _calculate_monthly_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate month-over-month spending trends"""
        monthly_totals = df.groupby(pd.Grouper(key='date', freq='M'))['amount'].sum()
        
        return {
            'values': monthly_totals.to_dict(),
            'growth_rate': monthly_totals.pct_change().mean() * 100,
            'trend_direction': 'increasing' if monthly_totals.pct_change().mean() > 0 
                             else 'decreasing'
        }

    def _calculate_category_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze spending trends by category"""
        category_monthly = df.pivot_table(
            index='category',
            columns=pd.Grouper(key='date', freq='M'),
            values='amount',
            aggfunc='sum'
        ).fillna(0)
        
        category_growth = category_monthly.pct_change(axis=1).mean(axis=1)
        
        return {
            'growth_rates': category_growth.to_dict(),
            'top_growing': category_growth.nlargest(3).to_dict(),
            'top_declining': category_growth.nsmallest(3).to_dict()
        }

    def _detect_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect seasonal spending patterns"""
        seasonal = df.groupby([
            df['date'].dt.month,
            'category'
        ])['amount'].mean().reset_index()
        
        high_seasons = seasonal.groupby('category').apply(
            lambda x: x.nlargest(1, 'amount')
        ).reset_index(drop=True)
        
        return {
            'monthly_averages': seasonal.to_dict(orient='records'),
            'peak_spending_months': high_seasons.to_dict(orient='records')
        }

    def _detect_anomalies(self, df: pd.DataFrame) -> list:
        """Detect unusual spending patterns"""
        anomalies = []
        
        # Calculate z-scores for amounts by category
        for category in df['category'].unique():
            cat_data = df[df['category'] == category]
            mean = cat_data['amount'].mean()
            std = cat_data['amount'].std()
            
            if std == 0:
                continue
                
            z_scores = (cat_data['amount'] - mean) / std
            
            # Flag transactions with z-score > 2
            unusual = cat_data[abs(z_scores) > 2]
            
            for _, row in unusual.iterrows():
                anomalies.append({
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'category': category,
                    'amount': row['amount'],
                    'z_score': z_scores[row.name]
                })
        
        return anomalies

    def _generate_trend_summary(self,
                              monthly_trends: Dict[str, Any],
                              category_trends: Dict[str, Any],
                              seasonal_patterns: Dict[str, Any],
                              anomalies: list) -> str:
        """Generate a natural language summary of the trends"""
        summary = []
        
        # Overall trend
        summary.append(
            f"Overall spending is {monthly_trends['trend_direction']} "
            f"at a rate of {monthly_trends['growth_rate']:.1f}% per month."
        )
        
        # Category trends
        if category_trends['top_growing']:
            top_growing = list(category_trends['top_growing'].keys())[0]
            summary.append(
                f"Highest growth in {top_growing} category."
            )
        
        # Seasonal patterns
        if seasonal_patterns['peak_spending_months']:
            summary.append(
                "Seasonal spending patterns detected in "
                f"{len(seasonal_patterns['peak_spending_months'])} categories."
            )
        
        # Anomalies
        if anomalies:
            summary.append(
                f"Detected {len(anomalies)} unusual transactions "
                "that may require attention."
            )
        
        return " ".join(summary)
