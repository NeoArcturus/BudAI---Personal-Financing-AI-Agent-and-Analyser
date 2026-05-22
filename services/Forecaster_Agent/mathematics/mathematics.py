import ctypes

import os

import subprocess

import pandas as pd

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
            # Test if the existing .so is valid for this OS/Arch
            test_lib = ctypes.CDLL(output_so)
            logger.debug("Existing hybrid_forecaster.so is valid.")
        except Exception as e:
            logger.warning(f"Existing .so is invalid ({e}). Re-compiling...")
            try:
                os.remove(output_so)
            except:
                pass
            should_compile = True

    if should_compile:
        logger.info(f"Compiling C++ forecaster engine to {output_so}...")
        subprocess.run([
            "g++", "-O3", "-shared", "-fPIC", "-std=c++17",
            f"-I{algorithm_dir}", "-o", output_so, algo_cpp, hybrid_cpp
        ], check=True)
        logger.info("Compilation successful.")

    return output_so

def run_hybrid_engine(S0: float, mu: float, params: dict, days: int, paths: int, account_id: str) -> pd.DataFrame:

    output_so = _ensure_compiled()

    hybrid_forecaster_lib = ctypes.CDLL(output_so)

    hybrid_forecaster_lib.run_hybrid_forecast_v2.argtypes = [

        ctypes.c_double, ctypes.c_double,

        ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,

        ctypes.c_double, ctypes.c_double, ctypes.c_double,

        ctypes.c_int, ctypes.c_int, ctypes.c_char_p,

        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)

    ]

    hybrid_forecaster_lib.run_hybrid_forecast_v2.restype = ctypes.c_int

    buffer_size = days + 1

    mean_out = (ctypes.c_double * buffer_size)()

    careless_out = (ctypes.c_double * buffer_size)()

    optimal_out = (ctypes.c_double * buffer_size)()

    res = hybrid_forecaster_lib.run_hybrid_forecast_v2(

        S0, mu,

        params.get('kappa', 2.0), params.get('theta', 0.04), params.get('xi', 0.1), params.get('rho', -0.5),

        params.get('lambda', 0.1), params.get('mu_J', -0.05), params.get('sigma_J', 0.1),

        buffer_size, paths, account_id.encode('utf-8'),

        mean_out, careless_out, optimal_out

    )

    if res != 0:

        raise RuntimeError("Balance Engine Failed.")

    return pd.DataFrame([list(mean_out), list(careless_out), list(optimal_out)])

def run_converged_expense_engine(E0: float, mu: float, days: int, paths: int, account_id: str) -> pd.DataFrame:

    output_so = _ensure_compiled()

    hybrid_forecaster_lib = ctypes.CDLL(output_so)

    hybrid_forecaster_lib.run_converged_expense_forecast.argtypes = [

        ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_char_p,

        ctypes.POINTER(ctypes.c_double)

    ]

    hybrid_forecaster_lib.run_converged_expense_forecast.restype = ctypes.c_int

    buffer_size = days + 1

    expected_out = (ctypes.c_double * buffer_size)()

    res = hybrid_forecaster_lib.run_converged_expense_forecast(

        E0, mu, buffer_size, paths, account_id.encode('utf-8'),

        expected_out

    )

    if res != 0:

        raise RuntimeError("Expense Engine Failed.")

    return pd.DataFrame([list(expected_out)])

