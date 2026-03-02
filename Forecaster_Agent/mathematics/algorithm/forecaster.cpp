#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <string>
#include <algorithm>
#include <numeric>
#include <functional>

using namespace std;

int main(int argc, char *argv[])
{
    if (argc < 7)
        return 1;

    double S0 = atof(argv[1]);
    double mu = atof(argv[2]);
    double sigma = atof(argv[3]);
    int days = atoi(argv[4]);
    int paths = atoi(argv[5]);
    string account_id = argv[6];

    double dt = 1.0;
    vector<vector<double>> all_results(paths, vector<double>(days + 1));

    size_t seed = hash<string>{}(account_id);
    mt19937 gen(static_cast<unsigned int>(seed));
    normal_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < paths; ++i)
    {
        all_results[i][0] = S0;
        double current_S = S0;
        for (int t = 1; t <= days; ++t)
        {
            double Z = dist(gen);
            current_S *= exp((mu - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * Z);
            all_results[i][t] = current_S;
        }
    }

    vector<double> mean_trajectory(days + 1, 0.0);
    for (int t = 0; t <= days; ++t)
    {
        double sum = 0;
        for (int i = 0; i < paths; ++i)
            sum += all_results[i][t];
        mean_trajectory[t] = sum / paths;
    }

    int path2_idx = 0;
    double min_dist = -1.0;
    for (int i = 0; i < paths; ++i)
    {
        double d = 0;
        for (int t = 0; t <= days; ++t)
            d += pow(all_results[i][t] - mean_trajectory[t], 2);
        if (min_dist < 0 || d < min_dist)
        {
            min_dist = d;
            path2_idx = i;
        }
    }

    vector<pair<double, int>> final_states;
    for (int i = 0; i < paths; ++i)
        final_states.push_back({all_results[i][days], i});
    sort(final_states.begin(), final_states.end());

    int path1_idx = final_states[static_cast<int>(0.05 * paths)].second;

    double threshold = 0.05;
    vector<int> safe_indices;
    for (int i = 0; i < paths; ++i)
    {
        bool safe = true;
        for (int t = 0; t <= days; ++t)
        {
            if (all_results[i][t] < threshold)
            {
                safe = false;
                break;
            }
        }
        if (safe)
            safe_indices.push_back(i);
    }

    int path3_idx;
    if (safe_indices.empty())
    {
        path3_idx = final_states.back().second;
    }
    else
    {
        double min_var = -1.0;
        for (int idx : safe_indices)
        {
            double sum = accumulate(all_results[idx].begin(), all_results[idx].end(), 0.0);
            double mean = sum / (days + 1);
            double var = 0;
            for (double val : all_results[idx])
                var += pow(val - mean, 2);
            if (min_var < 0 || var < min_var)
            {
                min_var = var;
                path3_idx = idx;
            }
        }
    }

    ofstream file("all_paths_for_" + to_string(days) + "_days.csv");
    int selected[] = {path1_idx, path2_idx, path3_idx};
    for (int idx : selected)
    {
        for (int t = 0; t <= days; ++t)
        {
            file << all_results[idx][t] << (t == days ? "" : ",");
        }
        file << "\n";
    }
    file.close();
    cout << "\nSuccessfully generated 3 paths for the current balance of £" << S0 << " over " << days << " days";

    return 0;
}