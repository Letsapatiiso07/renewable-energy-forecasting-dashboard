"""
One-hour least-cost dispatch with linear programming (PuLP + CBC).

Costs and capacities are illustrative placeholders from utils.config, not
market data. The saving is measured against the same dispatch with no solar
or wind, so it shows what the forecast renewable output is worth under
those assumptions.
"""
from typing import Dict, Optional

import pulp

from utils.config import CAPACITIES_MW, ENERGY_COSTS
from utils.logger import setup_logger

logger = setup_logger(__name__)


class GridCostOptimizer:
    """Least-cost dispatch of solar, wind, hydro, coal and gas for one hour"""

    def __init__(self, costs: Optional[Dict[str, float]] = None,
                 capacities: Optional[Dict[str, float]] = None):
        self.costs = costs or ENERGY_COSTS
        self.capacities = capacities or CAPACITIES_MW

    def _solve(self, demand_mw: float, limits: Dict[str, float]) -> Optional[Dict[str, float]]:
        prob = pulp.LpProblem("dispatch", pulp.LpMinimize)
        use = {s: pulp.LpVariable(s, lowBound=0, upBound=cap) for s, cap in limits.items()}
        prob += pulp.lpSum(use[s] * self.costs[s] for s in use)
        prob += pulp.lpSum(use.values()) == demand_mw
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        if pulp.LpStatus[prob.status] != "Optimal":
            return None
        return {s: float(v.value()) for s, v in use.items()}

    def _cost(self, allocation: Dict[str, float]) -> float:
        return sum(mw * self.costs[s] for s, mw in allocation.items())

    def optimize_allocation(self, demand_mw: float, solar_available: float,
                            wind_available: float) -> Dict:
        """Cheapest way to meet `demand_mw` for one hour. Costs are USD/MWh."""
        limits = {"solar": max(solar_available, 0.0), "wind": max(wind_available, 0.0),
                  **self.capacities}
        allocation = self._solve(demand_mw, limits)
        baseline = self._solve(demand_mw, {**limits, "solar": 0.0, "wind": 0.0})
        if allocation is None or baseline is None:
            total = sum(limits.values())
            logger.warning(f"No feasible dispatch for {demand_mw} MW (capacity {total:.0f} MW)")
            return {
                "status": "infeasible",
                "message": f"Demand {demand_mw:.0f} MW is above available capacity "
                           f"({total:.0f} MW with renewables, "
                           f"{total - limits['solar'] - limits['wind']:.0f} MW without)",
            }

        cost = self._cost(allocation)
        baseline_cost = self._cost(baseline)
        return {
            "status": "optimal",
            "demand_mw": demand_mw,
            "allocation": {f"{s}_mw": mw for s, mw in allocation.items()},
            "cost_usd": cost,
            "baseline_cost_usd": baseline_cost,
            "saving_usd": baseline_cost - cost,
            "saving_pct": (baseline_cost - cost) / baseline_cost * 100 if baseline_cost else 0.0,
            "solar_wind_share_pct": (allocation["solar"] + allocation["wind"]) / demand_mw * 100,
            "assumed_costs_usd_per_mwh": self.costs,
        }
