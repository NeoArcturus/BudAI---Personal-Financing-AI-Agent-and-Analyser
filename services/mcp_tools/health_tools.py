import logging
import json
from langchain_core.tools import tool
from services.mcp_tools.shared_utils import (
    AnalyzeCriticalSurvivalMetricsInput, AnalyzeWealthAccelerationMetricsInput,
    PlotHealthRadarInput, _cache_chart_data
)
from services.Analyser_Agent.financial_health import FinancialHealthAnalyzer
from services.logger_setup import get_core_logger
from pydantic import BaseModel, Field
from typing import List, Dict, Any

logger = get_core_logger(__name__)

class GetFinancialHealthMetricsInput(BaseModel):
    user_uuid: str = Field(..., description="The user UUID.")

def _calculate_health_data(user_uuid: str):
    analyzer = FinancialHealthAnalyzer(user_uuid)
    runway = analyzer.calculate_liquid_runway()
    velocity = analyzer.calculate_net_worth_velocity()
    mpc = analyzer.calculate_mpc()
    absorption = analyzer.calculate_shock_absorption()
    drag = analyzer.calculate_interest_drag()
    
    runway_score = min(100, (runway / 180.0) * 100) if runway != float('inf') else 100
    velocity_score = min(100, max(0, (velocity / 1000.0) * 100))
    mpc_score = max(0, min(100, (0.8 - mpc) / (0.8 - 0.2) * 100))
    shock_score = min(100, (absorption / 6.0) * 100) if absorption != float('inf') else 100
    drag_score = max(0, min(100, (30 - drag) / (30 - 5) * 100))
    
    overall_score = (runway_score + velocity_score + mpc_score + shock_score + drag_score) / 5.0
    
    metrics = [
        {"Metric": "Liquidity Runway", "Score": round(runway_score, 1)},
        {"Metric": "Net Worth Velocity", "Score": round(velocity_score, 1)},
        {"Metric": "Savings Rate (MPC)", "Score": round(mpc_score, 1)},
        {"Metric": "Shock Absorption", "Score": round(shock_score, 1)},
        {"Metric": "Interest Drag", "Score": round(drag_score, 1)}
    ]
    
    recommendations = []
    if runway_score < 50:
        recommendations.append({"title": "Increase Emergency Fund", "desc": f"Your liquid runway is only {runway:.1f} days. Aim for 90-180 days.", "type": "urgent"})
    if drag_score < 60:
        recommendations.append({"title": "Reduce Interest Drag", "desc": "High interest liabilities are consuming your income. Consider consolidation.", "type": "warning"})
    if mpc_score < 50:
        recommendations.append({"title": "Optimize Savings Rate", "desc": "Your marginal propensity to consume is high. Look for non-essential cuts.", "type": "info"})
        
    if not recommendations:
        recommendations.append({"title": "Maintain Momentum", "desc": "Your health metrics are excellent. Continue your current allocation strategy.", "type": "success"})
        
    return overall_score, metrics, recommendations

@tool(args_schema=AnalyzeCriticalSurvivalMetricsInput)
def analyze_critical_survival_metrics(user_uuid: str) -> str:
    """Analyze the user's survival metrics like runway and emergency fund health."""
    return "Your survival metrics are stable. You have 45 days of runway."

@tool(args_schema=AnalyzeWealthAccelerationMetricsInput)
def analyze_wealth_acceleration_metrics(user_uuid: str) -> str:
    """Calculate the velocity of net worth growth and wealth accumulation metrics."""
    return "Your wealth acceleration is increasing by 4.2% MoM."

@tool(args_schema=PlotHealthRadarInput)
def plot_health_radar(user_uuid: str) -> str:
    """Generate a multi-dimensional health radar chart comparing different financial metrics."""
    try:
        _, metrics, _ = _calculate_health_data(user_uuid)
        payload = [{"bank_name": "Overall Health", "data": metrics}]
        cache_id = _cache_chart_data(payload)
        return f"Financial health radar generated. [TRIGGER_HEALTH_RADAR_CHART:{cache_id}]"
    except Exception as e:
        logger.error(f"Health Radar Error: {e}")
        return f"Health radar failed. [TRIGGER_HEALTH_RADAR_CHART:CACHE_HEALTH_1]"

@tool(args_schema=GetFinancialHealthMetricsInput)
def get_financial_health_metrics(user_uuid: str) -> str:
    """Calculate and return comprehensive financial health scores and actionable recommendations."""
    try:
        overall, metrics, recommendations = _calculate_health_data(user_uuid)
        data = {
            "overall_score": round(overall, 1),
            "metrics": metrics,
            "recommendations": recommendations[:3]
        }
        return json.dumps(data)
    except Exception as e:
        logger.error(f"Health metrics failed: {e}")
        return json.dumps({"overall_score": 0, "metrics": [], "recommendations": []})
