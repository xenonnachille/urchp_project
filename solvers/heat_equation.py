import numpy as np
import numexpr as ne
from typing import Union

def solve_heat_equation(alpha: float, length: float, nx: int, nt: int, dt: float, 
                       initial_condition: Union[np.ndarray, str], boundary_conditions: dict, 
                       source_term: str = None, scheme: str = 'crank-nicolson') -> np.ndarray:
    """
    универсальный решатель уравнения теплопроводности.
    """
    dx = length / (nx - 1)
    x = np.linspace(0, length, nx)
    u = np.zeros((nx, nt))
    
    # инициализация начальных условий
    if isinstance(initial_condition, str):
        env = {'x': x, 'pi': np.pi, 't': 0.0}
        u[:, 0] = ne.evaluate(initial_condition, local_dict=env)
    else:
        if len(initial_condition) != nx:
            raise ValueError(f"длина начального условия ({len(initial_condition)}) должна соответствовать nx ({nx})")
        u[:, 0] = initial_condition
    
    r = alpha * dt / (dx ** 2)
    env = {'x': x, 'pi': np.pi, 't': 0.0, 'u': u[:, 0]}
    
    if scheme == 'explicit':
        return _explicit_scheme(u, x, nx, nt, dt, dx, r, boundary_conditions, source_term, env)
    elif scheme == 'implicit':
        return _implicit_scheme(u, x, nx, nt, dt, dx, r, boundary_conditions, source_term, env)
    elif scheme == 'crank-nicolson':
        return _crank_nicolson_scheme(u, x, nx, nt, dt, dx, r, boundary_conditions, source_term, env)
    else:
        raise ValueError(f"Неизвестная схема: {scheme}")
    

def _apply_boundary_condition(u, t, bc, x, dx, dt, env):
    """применяет граничные условия"""
    env['t'] = t * dt
    
    # левая граница
    left_bc = bc['left']
    if left_bc.type == "dirichlet":
        if isinstance(left_bc.value, (float, int)):
            u[0, t] = float(left_bc.value)
        else:
            u[0, t] = ne.evaluate(left_bc.value, local_dict=env)
    elif left_bc.type == "neumann":
        if isinstance(left_bc.value, (float, int)):
            value = float(left_bc.value)
        else:
            value = ne.evaluate(left_bc.value, local_dict=env)
        u[0, t] = u[1, t-1] - dx * value
    
    # правая граница
    right_bc = bc['right']
    if right_bc.type == "dirichlet":
        if isinstance(right_bc.value, (float, int)):
            u[-1, t] = float(right_bc.value)
        else:
            u[-1, t] = ne.evaluate(right_bc.value, local_dict=env)
    elif right_bc.type == "neumann":
        if isinstance(right_bc.value, (float, int)):
            value = float(right_bc.value)
        else:
            value = ne.evaluate(right_bc.value, local_dict=env)
        u[-1, t] = u[-2, t-1] + dx * value


def _apply_source_term(u, x, t, dt, source_term, env):
    """добавляет член источника"""
    if source_term:
        env['t'] = t * dt
        env['u'] = u[:, t-1]
        source = ne.evaluate(source_term, local_dict=env)
        u[:, t] += source * dt

def _explicit_scheme(u, x, nx, nt, dt, dx, r, bc, source_term, env):
    """явная схема"""
    for t in range(1, nt):
        # внутренние точки
        for i in range(1, nx-1):
            u[i, t] = u[i, t-1] + r * (u[i+1, t-1] - 2*u[i, t-1] + u[i-1, t-1]) - dt * u[i, t-1]
        
        # граничные условия
        _apply_boundary_condition(u, t, bc, x, dx, dt, env)
        
        # источник
        _apply_source_term(u, x, t, dt, source_term, env)
    
    return u

def _implicit_scheme(u, x, nx, nt, dt, dx, r, bc, source_term, env):
    """неявная схема"""
    A = np.zeros((nx, nx))
    np.fill_diagonal(A, 1 + 2*r + dt)  # +dt для члена -u
    np.fill_diagonal(A[1:], -r)
    np.fill_diagonal(A[:, 1:], -r)
    
    left_bc = bc['left']
    right_bc = bc['right']
    
    if left_bc.type == 'dirichlet':
        A[0, :] = 0
        A[0, 0] = 1
    elif left_bc.type == 'neumann':
        A[0, 0] = 1 + r + dt
        A[0, 1] = -r
    
    if right_bc.type == 'dirichlet':
        A[-1, :] = 0
        A[-1, -1] = 1
    elif right_bc.type == 'neumann':
        A[-1, -1] = 1 + r + dt
        A[-1, -2] = -r
    
    for t in range(1, nt):
        b = u[:, t-1].copy()
        
        if left_bc.type == 'dirichlet':
            b[0] = float(left_bc.value) if isinstance(left_bc.value, (float, int)) \
                else ne.evaluate(left_bc.value, local_dict={'t': t*dt, 'pi': np.pi})
        elif left_bc.type == 'neumann':
            pass 
            
        if right_bc.type == 'dirichlet':
            b[-1] = float(right_bc.value) if isinstance(right_bc.value, (float, int)) \
                else ne.evaluate(right_bc.value, local_dict={'t': t*dt, 'pi': np.pi})
        elif right_bc.type == 'neumann':
            pass  

        if source_term:
            env['t'] = (t-1)*dt
            env['u'] = u[:, t-1]
            b += ne.evaluate(source_term, local_dict=env) * dt
        
        u[:, t] = np.linalg.solve(A, b)
    
    return u

def _crank_nicolson_scheme(u, x, nx, nt, dt, dx, r, bc, source_term, env):
    """схема Кранка-Николсона"""
    half_r = r / 2
    half_dt = dt / 2
    
    A = np.zeros((nx, nx))
    B = np.zeros((nx, nx))
    
    np.fill_diagonal(A, 1 + half_r + half_dt)
    np.fill_diagonal(A[1:], -half_r)
    np.fill_diagonal(A[:, 1:], -half_r)
    
    np.fill_diagonal(B, 1 - half_r - half_dt)
    np.fill_diagonal(B[1:], half_r)
    np.fill_diagonal(B[:, 1:], half_r)
    
    left_bc = bc['left']
    right_bc = bc['right']
   
    if left_bc.type == 'dirichlet':
        A[0, :] = 0
        A[0, 0] = 1
        B[0, :] = 0
    elif left_bc.type == 'neumann':
        A[0, 0] = 1 + half_r + half_dt
        A[0, 1] = -half_r
        B[0, 0] = 1 - half_r - half_dt
        B[0, 1] = half_r
    elif left_bc.type == 'robin':
        A[0, 0] = 1 + half_r + dx*0.5; A[0, 1] = -half_r
        B[0, 0] = 1 - half_r - dx*0.5; B[0, 1] = half_r
    
    if right_bc.type == 'dirichlet':
        A[-1, :] = 0
        A[-1, -1] = 1
        B[-1, :] = 0
    elif right_bc.type == 'neumann':
        A[-1, -1] = 1 + half_r + half_dt
        A[-1, -2] = -half_r
        B[-1, -1] = 1 - half_r - half_dt
        B[-1, -2] = half_r
    
    for t in range(1, nt):
        b = B @ u[:, t-1]
        env['t'] = (t-1)*dt + dt/2  
        
        if left_bc.type == 'dirichlet':
            b[0] = left_bc.value if isinstance(left_bc.value, (float, int)) \
                   else ne.evaluate(left_bc.value, local_dict=env)
        elif left_bc.type == 'robin':
            val = ne.evaluate(left_bc.value, env)
            b[0] += dx * val
        
        if right_bc.type == 'dirichlet':
            b[-1] = right_bc.value if isinstance(right_bc.value, (float, int)) \
                    else ne.evaluate(right_bc.value, local_dict=env)
        
        if source_term:
            env['u'] = u[:, t-1]
            b += ne.evaluate(source_term, local_dict=env) * dt
        
        u[:, t] = np.linalg.solve(A, b)
    
    return u