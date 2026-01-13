"""
Cost optimization using linear programming
"""
import pulp
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime

from utils.config import ENERGY_COSTS, TARGET_COST_REDUCTION
from utils.logger import setup_logger

logger = setup_logger(__name__)


class GridCostOptimizer:
    """Optimize energy grid costs using linear programming"""
    
    def __init__(self):
        self.costs = ENERGY_COSTS
        self.problem = None
        self.solution = {}
    
    def optimize_allocation(
        self,
        demand_mw: float,
        solar_available: float,
        wind_available: float,
        coal_capacity: float = 500,
        gas_capacity: float = 300,
        hydro_capacity: float = 200
    ) -> Dict:
        """
        Optimize energy source allocation to minimize costs
        
        Args:
            demand_mw: Total demand in MW
            solar_available: Available solar output in MW
            wind_available: Available wind output in MW
            coal_capacity: Coal capacity in MW
            gas_capacity: Gas capacity in MW
            hydro_capacity: Hydro capacity in MW
            
        Returns:
            Optimization results with allocation and costs
        """
        # Create optimization problem
        prob = pulp.LpProblem("Energy_Cost_Optimization", pulp.LpMinimize)
        
        # Decision variables (how much to use from each source)
        solar_use = pulp.LpVariable("solar", lowBound=0, upBound=solar_available)
        wind_use = pulp.LpVariable("wind", lowBound=0, upBound=wind_available)
        coal_use = pulp.LpVariable("coal", lowBound=0, upBound=coal_capacity)
        gas_use = pulp.LpVariable("gas", lowBound=0, upBound=gas_capacity)
        hydro_use = pulp.LpVariable("hydro", lowBound=0, upBound=hydro_capacity)
        
        # Objective: Minimize cost
        prob += (
            solar_use * self.costs["solar"] +
            wind_use * self.costs["wind"] +
            coal_use * self.costs["coal"] +
            gas_use * self.costs["gas"] +
            hydro_use * self.costs["hydro"],
            "Total_Cost"
        )
        
        # Constraint: Meet demand
        prob += (
            solar_use + wind_use + coal_use + gas_use + hydro_use >= demand_mw,
            "Meet_Demand"
        )
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract solution
        if pulp.LpStatus[prob.status] == "Optimal":
            allocation = {
                "solar_mw": pulp.value(solar_use),
                "wind_mw": pulp.value(wind_use),
                "coal_mw": pulp.value(coal_use),
                "gas_mw": pulp.value(gas_use),
                "hydro_mw": pulp.value(hydro_use)
            }
            
            # Calculate costs
            optimized_cost = pulp.value(prob.objective)
            
            # Baseline cost (using coal for everything)
            baseline_cost = demand_mw * self.costs["coal"]
            
            cost_savings = baseline_cost - optimized_cost
            cost_reduction_pct = (cost_savings / baseline_cost) * 100
            
            result = {
                "status": "optimal",
                "allocation": allocation,
                "optimized_cost": optimized_cost,
                "baseline_cost": baseline_cost,
                "cost_savings": cost_savings,
                "cost_reduction_pct": cost_reduction_pct,
                "renewable_percentage": (
                    (allocation["solar_mw"] + allocation["wind_mw"] + allocation["hydro_mw"]) 
                    / demand_mw * 100
                ),
                "demand_mw": demand_mw
            }
            
            logger.info(
                f"Optimization: {cost_reduction_pct:.1f}% cost reduction, "
                f"{result['renewable_percentage']:.1f}% renewable"
            )
            
            return result
        else:
            logger.error(f"Optimization failed: {pulp.LpStatus[prob.status]}")
            return {"status": "failed", "message": pulp.LpStatus[prob.status]}
    
    def optimize_batch(
        self,
        forecasts_df: pd.DataFrame,
        demand_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Optimize allocation for multiple time periods
        
        Args:
            forecasts_df: DataFrame with solar/wind forecasts
            demand_df: DataFrame with demand forecasts
            
        Returns:
            DataFrame with optimization results
        """
        results = []
        
        for idx, row in forecasts_df.iterrows():
            demand = demand_df.loc[idx, "demand_mw"] if idx in demand_df.index else 400
            
            opt_result = self.optimize_allocation(
                demand_mw=demand,
                solar_available=row.get("solar_forecast_mw", 0),
                wind_available=row.get("wind_forecast_mw", 0)
            )
            
            if opt_result["status"] == "optimal":
                result_row = {
                    "time": row.get("time"),
                    "location": row.get("location"),
                    **opt_result["allocation"],
                    "optimized_cost": opt_result["optimized_cost"],
                    "cost_savings": opt_result["cost_savings"],
                    "cost_reduction_pct": opt_result["cost_reduction_pct"],
                    "renewable_pct": opt_result["renewable_percentage"]
                }
                results.append(result_row)
        
        return pd.DataFrame(results)