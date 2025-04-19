from pydantic import BaseModel, Field, validator
from typing import Dict, Literal, Union, Optional, List

class BoundaryCondition(BaseModel):
    type: Literal["dirichlet", "neumann", "robin"]
    value: Union[float, str]  # Число или выражение (например, "t" или "sin(x)")

class HeatEquationInput(BaseModel):
    alpha: float = Field(..., gt=0, description="коэффициент температуропроводности (> 0)")
    length: float = Field(..., gt=0, description="длина области в метрах")
    nx: int = Field(..., gt=2, description="число точек по пространству (> 2)")
    nt: int = Field(..., gt=0, description="число шагов по времени")
    dt: float = Field(..., gt=0, description="размер шага по времени")
    initial_condition: Union[List[float], str] = Field(..., 
        description="начальное распределение температуры (массив или выражение типа 'cos(x)')")
    boundary_conditions: Dict[str, BoundaryCondition]
    source_term: Optional[str] = None
    scheme: Optional[Literal["explicit", "implicit", "crank-nicolson"]] = "crank-nicolson"

    @validator('boundary_conditions')
    def validate_boundary_conditions(cls, v):
        if 'left' not in v or 'right' not in v:
            raise ValueError("граничные условия должны включать 'left' и 'right'")
        return v
    
    @validator('initial_condition')
    def validate_initial_condition(cls, v):
        if isinstance(v, list) and len(v) < 3:
            raise ValueError("для начального условия в виде списка требуется минимум 3 точки")
        return v