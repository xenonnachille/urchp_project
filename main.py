from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from schemas import HeatEquationInput
from solvers.heat_equation import solve_heat_equation
import plotly.graph_objects as go

app = FastAPI(
    title="PDE Solver API",
    description="API for solving 1D heat equation with different numerical schemes",
    version="1.0.0"
)

@app.post("/solve/heat-equation/")
async def solve_heat_eq(params: HeatEquationInput):
    """Solve 1D heat equation with given parameters.
    
    Args:
        params: HeatEquationInput containing all necessary parameters
        
    Returns:
        Dictionary containing solution matrix and visualization HTML
    """
    try:
        # Validate input array length matches nx
        if len(params.initial_condition) != params.nx:
            raise HTTPException(
                status_code=400,
                detail=f"Initial condition length ({len(params.initial_condition)}) must match nx ({params.nx})"
            )

        u0 = np.array(params.initial_condition)
        
        solution = solve_heat_equation(
            alpha=params.alpha,
            length=params.length,
            nx=params.nx,
            nt=params.nt,
            dt=params.dt,
            initial_condition=u0,
            bc=params.boundary_conditions
        )

        fig = go.Figure(data=[go.Surface(z=solution.T)])
        fig.update_layout(
            title="Heat Equation Solution",
            scene=dict(
                xaxis_title='Time',
                yaxis_title='Position',
                zaxis_title='Temperature'
            )
        )
        plot_html = fig.to_html(full_html=False)

        return {
            "solution": solution.tolist(),
            "visualization": plot_html
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))