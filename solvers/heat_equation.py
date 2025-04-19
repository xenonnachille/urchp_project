import numpy as np

def solve_heat_equation(alpha: float, length: float, nx: int, nt: int, dt: float, 
                       initial_condition: np.ndarray, bc: dict, scheme: str = 'explicit') -> np.ndarray:
    """
    решает одномерное уравнение теплопроводности разными методами.
    
    параметры:
        alpha: коэффициент температуропроводности
        length: длина области
        nx: число точек по пространству
        nt: число шагов по времени
        dt: шаг по времени
        initial_condition: начальное распределение температуры
        bc: граничные условия (словарь с 'left' и 'right')
        scheme: используемая схема ('explicit', 'implicit', 'crank-nicolson')
        
    возвращает:
        Матрицу решения u(x,t) в виде массива NumPy
    """
    dx = length / (nx - 1)  # шаг по пространству
    u = np.zeros((nx, nt))  
    u[:, 0] = initial_condition  
    
    r = alpha * dt / (dx ** 2)  # число Куранта (пар-р устойчивости)
    
    # выбор схемы в зависимости от параметра
    if scheme == 'explicit':
        return _explicit_scheme(u, nx, nt, r, bc)
    elif scheme == 'implicit':
        return _implicit_scheme(u, nx, nt, r, bc)
    elif scheme == 'crank-nicolson':
        return _crank_nicolson_scheme(u, nx, nt, r, bc)
    else:
        raise ValueError(f"неизвестная схема: {scheme}")

def _explicit_scheme(u, nx, nt, r, bc):
    """ явная конечно-разностная схема (4-точечная)"""
    for t in range(1, nt):  # Цикл по времени
        # Граничные условия
        u[0, t] = bc["left"]
        u[-1, t] = bc["right"]
        
        # Основная формула для внутренних точек
        for x in range(1, nx-1):
            u[x, t] = u[x, t-1] + r * (
                u[x+1, t-1] - 2*u[x, t-1] + u[x-1, t-1]
            )
    return u

def _implicit_scheme(u, nx, nt, r, bc):
    """ неявная конечно-разностная схема (6-точечная) """
    A = np.zeros((nx, nx))
    np.fill_diagonal(A, 1 + 2*r)
    np.fill_diagonal(A[1:], -r)    
    np.fill_diagonal(A[:, 1:], -r)  

    A[0, 0] = 1   
    A[0, 1] = 0     

    A[-1, -1] = 1  
    A[-1, -2] = 0   
    
    for t in range(1, nt):
        b = u[:, t-1].copy()
        b[0] = bc["left"]    
        b[-1] = bc["right"] 
        
        u[:, t] = np.linalg.solve(A, b)
    
    return u

def _crank_nicolson_scheme(u, nx, nt, r, bc):
    """ Схема Кранка-Николсона - усреднение явной и неявной схем. """
    A = np.zeros((nx, nx))  
    B = np.zeros((nx, nx))  
    
    np.fill_diagonal(A, 1 + r)
    np.fill_diagonal(A[1:], -r/2)     
    np.fill_diagonal(A[:, 1:], -r/2) 
    
    np.fill_diagonal(B, 1 - r)
    np.fill_diagonal(B[1:], r/2)      
    np.fill_diagonal(B[:, 1:], r/2)

    A[0, :] = 0    
    A[0, 0] = 1    
    A[-1, :] = 0   
    A[-1, -1] = 1   
    
    B[0, :] = 0     
    B[-1, :] = 0    
    
    for t in range(1, nt):
        b = B @ u[:, t-1]
        b[0] = bc["left"]    
        b[-1] = bc["right"]  
        
        u[:, t] = np.linalg.solve(A, b)
    
    return u

def solve_advanced(alpha, length, nx, nt, dt, u0, bc, source=None, kappa=None): 
    '''добавить источник или переменный коэффициент при производной'''
    dx = length / (nx - 1)
    u = np.zeros((nx, nt))
    u[:, 0] = u0
    
    for t in range(1, nt):
        # обработка граничных условий
        if bc.get("left") == "insulated":
            u[0, t] = u[1, t-1]  # Нулевой поток
        else:
            u[0, t] = bc["left"]
        
        # вычисление нового слоя с учетом источников
        for x in range(1, nx-1):
            k = kappa(x*dx) if kappa else alpha
            u[x, t] = u[x, t-1] + k*dt/dx**2 * (u[x+1, t-1] - 2*u[x, t-1] + u[x-1, t-1])
            
            if source:
                u[x, t] += source(x*dx, t*dt) * dt
                
    return u