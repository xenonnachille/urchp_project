from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
import numexpr as ne
from schemas import HeatEquationInput
from solvers.heat_equation import solve_heat_equation
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI(
    title="Universal PDE Solver API",
    description="API for solving 1D heat equation with arbitrary initial/boundary conditions",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # можешь указать конкретные origin'ы, если хочешь
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("index.html") as f:
        return f.read()

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
        return {
            "solution": solution.tolist(),
            "parameters": params.dict()
         }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))