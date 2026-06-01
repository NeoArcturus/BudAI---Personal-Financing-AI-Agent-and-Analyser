import ctypes
import os
import subprocess
import pandas as pd
import numpy as np
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

def _ensure_compiled():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    algorithm_dir = os.path.join(base_dir, "algorithm")
    algo_cpp = os.path.join(algorithm_dir, "algorithm.cpp")
    hybrid_cpp = os.path.join(algorithm_dir, "hybrid_algorithm.cpp")
    output_so = os.path.join(base_dir, "hybrid_forecaster.so")
    should_compile = not os.path.exists(output_so)
    if not should_compile:
        try:
            ctypes.CDLL(output_so)
        except Exception:
            try: os.remove(output_so)
            except: pass
            should_compile = True
    if should_compile:
        logger.info(f"Compiling C++ forecaster engine to {output_so}...")
        subprocess.run([
            "g++", "-O3", "-shared", "-fPIC", "-std=c++17",
            f"-I{algorithm_dir}", "-o", output_so, algo_cpp, hybrid_cpp
        ], check=True)
    return output_so

def run_hybrid_engine(S0, mu, params, days, paths, account_id, deterministic_calendar):
    output_so = _ensure_compiled()
    lib = ctypes.CDLL(output_so)
    lib.run_hybrid_forecast_v3.argtypes = [
        ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_double, ctypes.c_double, ctypes.c_double,
        ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)
    ]
    lib.run_hybrid_forecast_v3.restype = ctypes.c_int
    buffer_size = days + 1
    cal_ptr = (ctypes.c_double * buffer_size)(*deterministic_calendar)
    mean_out = (ctypes.c_double * buffer_size)()
    careless_out = (ctypes.c_double * buffer_size)()
    optimal_out = (ctypes.c_double * buffer_size)()
    res = lib.run_hybrid_forecast_v3(
        S0, mu,
        params.get('kappa', 2.0), params.get('theta', 0.04), params.get('xi', 0.1), params.get('rho', -0.5),
        params.get('lambda', 0.1), params.get('mu_J', -0.05), params.get('sigma_J', 0.1),
        buffer_size, paths, account_id.encode('utf-8'),
        cal_ptr, mean_out, careless_out, optimal_out
    )
    if res != 0: raise RuntimeError("Balance Engine Failed.")
    return pd.DataFrame([list(mean_out), list(careless_out), list(optimal_out)])

def run_converged_expense_engine(E0, mu, days, paths, account_id, deterministic_calendar):
    output_so = _ensure_compiled()
    lib = ctypes.CDLL(output_so)
    lib.run_converged_expense_forecast_v2.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, 
        ctypes.c_char_p, ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double)
    ]
    lib.run_converged_expense_forecast_v2.restype = ctypes.c_int
    buffer_size = days + 1
    cal_ptr = (ctypes.c_double * buffer_size)(*deterministic_calendar)
    expected_out = (ctypes.c_double * buffer_size)()
    res = lib.run_converged_expense_forecast_v2(
        E0, mu, buffer_size, paths, account_id.encode('utf-8'),
        cal_ptr, expected_out
    )
    if res != 0: raise RuntimeError("Expense Engine Failed.")
    return pd.DataFrame([list(expected_out)])
