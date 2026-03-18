#include "algorithm.h"
#include <string>
#include <functional>
#include <vector>
#include <algorithm>
#include <cmath>

extern "C"
{
    int run_hybrid_forecast(double S0, double mu, int buffer_size, int paths, const char *account_id, double *mean_out, double *careless_out, double *optimal_out)
    {
        HestonParams hp = {S0, 0.04, 2.0, 0.04, 0.1, -0.5};
        JumpParams jp = {0.1, -0.05, 0.1};
        double dt = 1.0;

        std::hash<std::string> hasher;
        int seed = static_cast<int>(hasher(std::string(account_id)));

        std::vector<std::vector<double>> final_paths = heston_jump_mc(hp, jp, mu, dt, buffer_size - 1, paths, seed);

        if (final_paths.empty())
            return 1;

        int path_len = final_paths[0].size();
        int days_to_copy = std::min(buffer_size, path_len);

        for (int j = 0; j < days_to_copy; ++j)
        {
            std::vector<double> day_vals(paths);
            double sum = 0.0;
            for (int i = 0; i < paths; ++i)
            {
                day_vals[i] = final_paths[i][j];
                sum += final_paths[i][j];
            }

            mean_out[j] = sum / paths;

            std::sort(day_vals.begin(), day_vals.end());
            int p05_idx = std::max(0, std::min(static_cast<int>(0.05 * paths), paths - 1));
            int p95_idx = std::max(0, std::min(static_cast<int>(0.95 * paths), paths - 1));

            careless_out[j] = day_vals[p05_idx];
            optimal_out[j] = day_vals[p95_idx];
        }

        return 0;
    }

    int run_converged_expense_forecast(double E0, double mu, int buffer_size, int paths, const char *account_id, double *expected_out)
    {
        HestonParams hp = {E0, 0.04, 2.0, 0.04, 0.1, -0.5};
        JumpParams jp = {0.1, -0.05, 0.1};
        double dt = 1.0;
        int sim_days = buffer_size - 1;

        std::hash<std::string> hasher;
        int seed = static_cast<int>(hasher(std::string(account_id)));

        std::vector<std::vector<double>> final_paths = heston_jump_mc(hp, jp, mu, dt, sim_days, paths, seed);
        if (final_paths.empty())
            return 1;

        double historical_expected_total = 0.0;
        for (int t = 1; t <= sim_days; ++t)
        {
            historical_expected_total += E0 * exp(mu * t);
        }

        double target_error = 0.0;
        int best_path_idx = 0;
        double smallest_difference = 1e9;

        for (size_t i = 0; i < final_paths.size(); ++i)
        {
            double path_sum = 0.0;
            for (size_t j = 1; j <= sim_days && j < final_paths[i].size(); ++j)
            {
                path_sum += final_paths[i][j];
            }

            double current_error = std::abs(path_sum - historical_expected_total) / historical_expected_total;
            double diff_from_target = std::abs(current_error - target_error);

            if (diff_from_target < smallest_difference)
            {
                smallest_difference = diff_from_target;
                best_path_idx = i;
            }
        }

        int path_len = final_paths[best_path_idx].size();
        int days_to_copy = std::min(buffer_size, path_len);

        for (int j = 0; j < days_to_copy; ++j)
        {
            expected_out[j] = final_paths[best_path_idx][j];
        }

        return 0;
    }
}