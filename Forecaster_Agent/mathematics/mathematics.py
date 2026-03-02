import ctypes
import os
import pandas as pd
import subprocess
import time


def run_hybrid_engine(S0: float, mu: float, days: int, paths: int, account_id: str) -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))

    algorithm_dir = os.path.join(base_dir, "algorithm")
    algo_cpp = os.path.join(algorithm_dir, "algorithm.cpp")
    hybrid_cpp = os.path.join(algorithm_dir, "hybrid_algorithm.cpp")
    output_so = os.path.join(base_dir, "hybrid_forecaster.so")

    if not os.path.exists(output_so):
        print("Compiling Hybrid C++ Engine...")
        subprocess.run([
            "g++", "-O3", "-shared", "-fPIC", "-std=c++17",
            f"-I{algorithm_dir}",
            "-o", output_so,
            algo_cpp,
            hybrid_cpp
        ], check=True)

    hybrid_forecaster_lib = ctypes.CDLL(output_so)

    hybrid_forecaster_lib.run_hybrid_forecast.argtypes = [
        ctypes.c_double,
        ctypes.c_double,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_char_p
    ]
    hybrid_forecaster_lib.run_hybrid_forecast.restype = ctypes.c_int

    print(
        f"Triggering Hybrid C++ engine for {paths} paths over {days} days...")
    start_time = time.time()

    acc_id_bytes = account_id.encode('utf-8')

    target_csv_path = os.path.join(base_dir, "hybrid_paths.csv")
    target_csv_bytes = target_csv_path.encode('utf-8')

    result = hybrid_forecaster_lib.run_hybrid_forecast(
        S0, mu, days, paths, acc_id_bytes, target_csv_bytes)

    execution_time = time.time() - start_time

    if result == 0:
        print(
            f"Simulation completed successfully in {execution_time:.4f} seconds!")

        target_csv = "Forecaster_Agent/hybrid_paths.csv"
        if not os.path.exists(target_csv):
            target_csv = "hybrid_paths.csv"

        return os.path.abspath(target_csv)
    else:
        raise RuntimeError(f"C++ Engine failed with return code: {result}")
