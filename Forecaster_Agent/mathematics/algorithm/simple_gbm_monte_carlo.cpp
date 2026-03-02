#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <string>
#include <functional>
#include <thread>

using namespace std;

void worker_task(double S0, double mu, double sigma, int days, int num_sims,
                 size_t base_seed, int start_idx, vector<vector<double>> *all_results)
{
    double dt = 1.0;
    mt19937 gen(static_cast<unsigned int>(base_seed + start_idx));
    normal_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < num_sims; ++i)
    {
        int global_idx = start_idx + i;
        (*all_results)[global_idx][0] = S0;
        double current_S = S0;
        for (int t = 1; t <= days; ++t)
        {
            double Z = dist(gen);
            current_S *= exp((mu - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * Z);
            (*all_results)[global_idx][t] = current_S;
        }
    }
}

extern "C"
{
    int run_simple_gbm_forecast(double S0, double mu, double sigma, int days, int paths, const char *account_id)
    {
        vector<vector<double>> all_results(paths, vector<double>(days + 1));
        size_t base_seed = hash<string>{}(string(account_id));

        unsigned int num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0)
            num_threads = 4;

        vector<thread> threads;
        int sims_per_thread = paths / num_threads;
        int current_start = 0;

        for (unsigned int i = 0; i < num_threads; ++i)
        {
            int count = (i == (num_threads - 1))
                            ? (paths - current_start)
                            : sims_per_thread;

            threads.emplace_back(worker_task, S0, mu, sigma, days, count, base_seed, current_start, &all_results);
            current_start += count;
        }

        for (auto &thread : threads)
            thread.join();

        ofstream file("all_paths_for_" + to_string(days) + "_days.csv");
        for (int p = 0; p < paths; ++p)
        {
            double St = S0;
            file << St;

            for (int t = 1; t <= days; ++t)
            {
                file << all_results[p][t] << (t == days ? "" : ",");
            }
            file << "\n";
        }

        return 0;
    }
}