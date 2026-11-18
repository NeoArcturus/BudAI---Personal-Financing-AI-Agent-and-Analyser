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

        std::vector<double> final_paths = heston_jump_mc(hp, jp, mu, dt, days, paths, seed, deterministic_calendar);

        if (final_paths.empty()) return 1;

        std::vector<double> day_vals(paths);
        const int p05_idx = static_cast<int>(0.05 * paths);
        const int p95_idx = static_cast<int>(0.95 * paths);

        for (int j = 0; j < buffer_size; ++j)
        {
            double sum = 0.0;
            for (int i = 0; i < paths; ++i)
            {
                double val = final_paths[static_cast<size_t>(i) * buffer_size + j];
                day_vals[i] = val;
                sum += val;
            }

            mean_out[j] = sum / paths;

            std::nth_element(day_vals.begin(), day_vals.begin() + p05_idx, day_vals.end());
            careless_out[j] = day_vals[p05_idx];

            std::nth_element(day_vals.begin() + p05_idx, day_vals.begin() + p95_idx, day_vals.end());
            optimal_out[j] = day_vals[p95_idx];
        }

        return 0;
    }
}
