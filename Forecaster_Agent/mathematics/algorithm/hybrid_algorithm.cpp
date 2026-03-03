#include "algorithm.h"
#include <fstream>
#include <string>
#include <functional>

extern "C"
{
    int run_hybrid_forecast(double S0, double mu, int days, int paths, const char *account_id, const char *output_file_path)
    {

        HestonParams hp = {S0, 0.04, 2.0, 0.04, 0.1, -0.5};
        JumpParams jp = {0.1, -0.05, 0.1};
        double dt = 1.0;

        std::hash<std::string> hasher;
        int seed = static_cast<int>(hasher(std::string(account_id)));

        std::vector<std::vector<double>> final_paths = heston_jump_mc(hp, jp, mu, dt, days, paths, seed);

        std::ofstream file(output_file_path);
        if (!file.is_open())
        {
            return 1;
        }

        for (size_t i = 0; i < final_paths.size(); ++i)
        {
            for (size_t j = 0; j < final_paths[i].size(); ++j)
            {
                file << final_paths[i][j] << (j == final_paths[i].size() - 1 ? "" : ",");
            }
            file << "\n";
        }
        file.close();

        return 0;
    }

    int run_converged_expense_forecast(double E0, double mu, int days, int paths, const char *account_id, const char *output_path)
    {
        HestonParams hp = {E0, 0.04, 2.0, 0.04, 0.1, -0.5};
        JumpParams jp = {0.1, -0.05, 0.1};
        double dt = 1.0;

        std::hash<std::string> hasher;
        int seed = static_cast<int>(hasher(std::string(account_id)));

        std::vector<std::vector<double>> final_paths = heston_jump_mc(hp, jp, mu, dt, days, paths, seed);

        double historical_expected_total = 0.0;
        for (int t = 1; t <= days; ++t)
        {
            historical_expected_total += E0 * exp(mu * t);
        }

        double target_error = 0.0;
        int best_path_idx = 0;
        double smallest_difference = 1e9;

        for (size_t i = 0; i < final_paths.size(); ++i)
        {
            double path_sum = 0.0;
            for (size_t j = 1; j <= days; ++j)
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

        std::ofstream file(output_path);
        if (!file.is_open())
            return 1;

        for (size_t j = 0; j < final_paths[best_path_idx].size(); ++j)
        {
            file << final_paths[best_path_idx][j] << (j == final_paths[best_path_idx].size() - 1 ? "" : ",");
        }
        file << "\n";
        file.close();

        return 0;
    }
}