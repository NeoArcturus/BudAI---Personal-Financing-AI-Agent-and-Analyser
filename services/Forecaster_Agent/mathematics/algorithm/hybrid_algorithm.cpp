#include "algorithm.h"
#include <string>
#include <functional>
#include <vector>
#include <algorithm>
#include <cmath>

extern "C"
{
    int run_hybrid_forecast_v3(
        double S0, double mu, 
        double kappa, double theta, double xi, double rho,
        double lambda, double mu_J, double sigma_J,
        int buffer_size, int paths, const char *account_id, 
        const double *deterministic_calendar,
        double *mean_out, double *careless_out, double *optimal_out
    ) {
        HestonParams hp = {S0, theta, kappa, theta, xi, rho};
        JumpParams jp = {lambda, mu_J, sigma_J};
        double dt = 1.0;
        int days = buffer_size - 1;

        std::hash<std::string> hasher;
        int seed = static_cast<int>(hasher(std::string(account_id)));

        std::vector<std::vector<double>> final_paths = heston_jump_mc(hp, jp, mu, dt, days, paths, seed, deterministic_calendar);

        if (final_paths.empty()) return 1;

        for (int j = 0; j < buffer_size; ++j)
        {
            std::vector<double> day_vals(paths);
            double sum = 0.0;
            for (int i = 0; i < paths; ++i)
            {
                day_vals[i] = final_paths[i][j];
                sum += final_paths[i][j];
            }

            mean_out[j] = sum / paths;

            int p05_idx = static_cast<int>(0.05 * paths);
            int p95_idx = static_cast<int>(0.95 * paths);

            std::nth_element(day_vals.begin(), day_vals.begin() + p05_idx, day_vals.end());
            careless_out[j] = day_vals[p05_idx];

            std::nth_element(day_vals.begin(), day_vals.begin() + p95_idx, day_vals.end());
            optimal_out[j] = day_vals[p95_idx];
        }

        return 0;
    }

    int run_converged_expense_forecast_v2(
        double E0, double mu, int buffer_size, int paths, 
        const char *account_id, const double *deterministic_calendar,
        double *expected_out
    ) {
        HestonParams hp = {E0, 0.04, 2.0, 0.04, 0.1, -0.5};
        JumpParams jp = {0.1, -0.05, 0.1};
        double dt = 1.0;
        int days = buffer_size - 1;

        std::hash<std::string> hasher;
        int seed = static_cast<int>(hasher(std::string(account_id)));

        std::vector<std::vector<double>> final_paths = heston_jump_mc(hp, jp, mu, dt, days, paths, seed, deterministic_calendar);
        if (final_paths.empty()) return 1;

        double historical_expected_total = 0.0;
        for (int t = 1; t <= days; ++t) {
            historical_expected_total += E0 * exp(mu * t);
        }

        int best_path_idx = 0;
        double smallest_difference = 1e18;

        for (int i = 0; i < paths; ++i) {
            double path_sum = 0.0;
            for (int j = 1; j < buffer_size; ++j) {
                path_sum += final_paths[i][j];
            }
            double diff = std::abs(path_sum - historical_expected_total);
            if (diff < smallest_difference) {
                smallest_difference = diff;
                best_path_idx = i;
            }
        }

        for (int j = 0; j < buffer_size; ++j) {
            expected_out[j] = final_paths[best_path_idx][j];
        }

        return 0;
    }
}
