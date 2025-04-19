import pytest
import numpy as np
from solvers.heat_equation import solve_heat_equation

def test_explicit_scheme():
    alpha = 0.01
    length = 1.0
    nx = 5
    nt = 3
    dt = 0.1
    u0 = np.array([0, 1, 1, 1, 0])
    bc = {"left": 0.0, "right": 0.0}
    
    solution = solve_heat_equation(alpha, length, nx, nt, dt, u0, bc, 'explicit')
    assert solution.shape == (nx, nt)
    assert np.all(solution[0, :] == 0)  # Left BC
    assert np.all(solution[-1, :] == 0)  # Right BC

def test_implicit_scheme():
    alpha = 0.01
    length = 1.0
    nx = 5
    nt = 3
    dt = 0.1
    u0 = np.array([0, 1, 1, 1, 0])
    bc = {"left": 0.0, "right": 0.0}
    
    solution = solve_heat_equation(alpha, length, nx, nt, dt, u0, bc, 'implicit')
    assert solution.shape == (nx, nt)
    assert np.all(solution[0, :] == 0)
    assert np.all(solution[-1, :] == 0)

def test_crank_nicolson_scheme():
    alpha = 0.01
    length = 1.0
    nx = 5
    nt = 3
    dt = 0.1
    u0 = np.array([0, 1, 1, 1, 0])
    bc = {"left": 0.0, "right": 0.0}
    
    solution = solve_heat_equation(alpha, length, nx, nt, dt, u0, bc, 'crank-nicolson')
    assert solution.shape == (nx, nt)
    assert np.all(solution[0, :] == 0)
    assert np.all(solution[-1, :] == 0)