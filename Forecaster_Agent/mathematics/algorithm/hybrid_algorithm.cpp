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
}