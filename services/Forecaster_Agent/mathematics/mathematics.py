import ctypes
import os
import subprocess
import pandas as pd


def _ensure_compiled():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    algorithm_dir = os.path.join(base_dir, "algorithm")
    algo_cpp = os.path.join(algorithm_dir, "algorithm.cpp")
    hybrid_cpp = os.path.join(algorithm_dir, "hybrid_algorithm.cpp")
    output_so = os.path.join(base_dir, "hybrid_forecaster.so")

    if not os.path.exists(output_so):
        subprocess.run([
            "g++", "-O3", "-shared", "-fPIC", "-std=c++17",
            f"-I{algorithm_dir}", "-o", output_so, algo_cpp, hybrid_cpp
        ], check=True)
    return output_so


def run_hybrid_engine(S0: float, mu: float, days: int, paths: int, account_id: str) -> pd.DataFrame:
    output_so = _ensure_compiled()
    hybrid_forecaster_lib = ctypes.CDLL(output_so)

    hybrid_forecaster_lib.run_hybrid_forecast.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(
            ctypes.c_double), ctypes.POINTER(ctypes.c_double)
    ]
    hybrid_forecaster_lib.run_hybrid_forecast.restype = ctypes.c_int

    buffer_size = days + 1
    mean_out = (ctypes.c_double * buffer_size)()
    careless_out = (ctypes.c_double * buffer_size)()
    optimal_out = (ctypes.c_double * buffer_size)()

    res = hybrid_forecaster_lib.run_hybrid_forecast(
        S0, mu, buffer_size, paths, account_id.encode('utf-8'),
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
