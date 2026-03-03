import ctypes
import os
import subprocess


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
    return output_so, base_dir


def run_hybrid_engine(S0: float, mu: float, days: int, paths: int, account_id: str) -> str:
    output_so, base_dir = _ensure_compiled()
    hybrid_forecaster_lib = ctypes.CDLL(output_so)

    hybrid_forecaster_lib.run_hybrid_forecast.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p
    ]
    hybrid_forecaster_lib.run_hybrid_forecast.restype = ctypes.c_int

    root_dir = os.path.abspath(os.path.join(base_dir, '..', '..'))
    csv_dir = os.path.join(root_dir, "saved_media", "csvs")
    os.makedirs(csv_dir, exist_ok=True)

    target_csv_path = os.path.join(csv_dir, "hybrid_paths.csv")

    result = hybrid_forecaster_lib.run_hybrid_forecast(
        S0, mu, days, paths, account_id.encode(
            'utf-8'), target_csv_path.encode('utf-8')
    )
    if result == 0:
        return target_csv_path
    raise RuntimeError("Balance Engine Failed.")


def run_converged_expense_engine(E0: float, mu: float, days: int, paths: int, account_id: str) -> str:
    output_so, base_dir = _ensure_compiled()
    hybrid_forecaster_lib = ctypes.CDLL(output_so)

    hybrid_forecaster_lib.run_converged_expense_forecast.argtypes = [
        ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p
    ]
    hybrid_forecaster_lib.run_converged_expense_forecast.restype = ctypes.c_int

    root_dir = os.path.abspath(os.path.join(base_dir, '..', '..'))
    csv_dir = os.path.join(root_dir, "saved_media", "csvs")
    os.makedirs(csv_dir, exist_ok=True)

    target_csv_path = os.path.join(csv_dir, "converged_expense.csv")

    result = hybrid_forecaster_lib.run_converged_expense_forecast(
        E0, mu, days, paths, account_id.encode(
            'utf-8'), target_csv_path.encode('utf-8')
    )
    if result == 0:
        return target_csv_path
    raise RuntimeError("Expense Engine Failed.")
