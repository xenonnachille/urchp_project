from pydantic import BaseModel, Field, validator
from typing import Dict, Literal, Union, Optional, List

class BoundaryCondition(BaseModel):
    type: Literal["dirichlet", "neumann", "robin"]
    value: Union[float, str]  # Число или выражение

class HeatEquationInput(BaseModel):
    alpha: float = Field(..., gt=0, description="коэффициент температуропроводности (> 0)")
    length: float = Field(..., gt=0, description="длина области в метрах")
    nx: int = Field(..., gt=2, description="число точек по пространству (> 2)")
    nt: int = Field(..., gt=0, description="число шагов по времени")
    dt: float = Field(..., gt=0, description="размер шага по времени")
    initial_condition: List[float] = Field(..., min_items=3, 
                                         description="начальное распределение температуры")
    boundary_conditions: Dict[str, BoundaryCondition]
    source_term: Optional[str] = None

    @validator('boundary_conditions')
    def validate_boundary_conditions(cls, v):
        # проверка, что граничные условия содержат нужные ключи
        if 'left' not in v or 'right' not in v:
            raise ValueError("граничные условия должны включать 'left' и 'right'")
        return v