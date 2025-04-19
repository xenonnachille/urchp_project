from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import numexpr as ne
from schemas import HeatEquationInput
from solvers.heat_equation import solve_heat_equation
import plotly.graph_objects as go
from typing import Union

app = FastAPI(
    title="Universal PDE Solver API",
    description="API for solving 1D heat equation with arbitrary initial/boundary conditions",
    version="1.0.0"
)

@app.post("/solve/heat-equation/")
async def solve_heat_eq(params: HeatEquationInput):
    """универсальный решатель уравнения теплопроводности."""
    try:
        # подготовка начальных условий
        x = np.linspace(0, params.length, params.nx)
        initial_condition = None
        
        if isinstance(params.initial_condition, list):
            initial_condition = np.array(params.initial_condition)
        else:  # это строка с выражением
            env = {'x': x, 'pi': np.pi, 't': 0.0}
            initial_condition = ne.evaluate(params.initial_condition, local_dict=env)
        
        # решение уравнения
        solution = solve_heat_equation(
            alpha=params.alpha,
            length=params.length,
            nx=params.nx,
            nt=params.nt,
            dt=params.dt,
            initial_condition=initial_condition,
            boundary_conditions=params.boundary_conditions,
            source_term=params.source_term,
            scheme=params.scheme
        )

        # визуализация
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
            "visualization": plot_html,
            "parameters": params.dict()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))