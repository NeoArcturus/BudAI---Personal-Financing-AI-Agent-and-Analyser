#include "algorithm.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <thread>

using namespace std;

const double PI = 3.14159265358979323846;

void fft(vector<complex<double>> &x)
{
    int n = x.size();
    if (n <= 1)
        return;

    vector<complex<double>> even(n / 2), odd(n / 2);
    for (int i = 0; i < n / 2; ++i)
    {
        even[i] = x[i * 2];
        odd[i] = x[i * 2 + 1];
    }

    fft(even);
    fft(odd);

    for (int i = 0; i < n / 2; ++i)
    {
        complex<double> t = polar(1.0, -2.0 * PI * i / n) * odd[i];
        x[i] = even[i] + t;
        x[i + n / 2] = even[i] - t;
    }
}

complex<double> bates_characteristic_function(double u, double T, double r, HestonParams hp, JumpParams jp)
{
    complex<double> i_comp(0.0, 1.0);
    complex<double> cu(u, 0.0);

    complex<double> d = sqrt(pow(hp.kappa - hp.rho * hp.xi * cu * i_comp, 2.0) + pow(hp.xi, 2.0) * (cu * i_comp + pow(cu, 2.0)));
    complex<double> g = (hp.kappa - hp.rho * hp.xi * cu * i_comp - d) / (hp.kappa - hp.rho * hp.xi * cu * i_comp + d);

    complex<double> C = r * cu * i_comp * T + (hp.kappa * hp.theta / pow(hp.xi, 2.0)) * ((hp.kappa - hp.rho * hp.xi * cu * i_comp - d) * T - 2.0 * log((1.0 - g * exp(-d * T)) / (1.0 - g)));
    complex<double> D = ((hp.kappa - hp.rho * hp.xi * cu * i_comp - d) / pow(hp.xi, 2.0)) * ((1.0 - exp(-d * T)) / (1.0 - g * exp(-d * T)));

    double expected_jump = exp(jp.mu_J + 0.5 * pow(jp.sigma_J, 2.0)) - 1.0;
    complex<double> jump_comp = jp.lambda * T * (exp(cu * i_comp * jp.mu_J - 0.5 * pow(jp.sigma_J * cu, 2.0)) - 1.0 - cu * i_comp * expected_jump);

    return exp(C + D * hp.v0 + i_comp * cu * log(hp.S0) + jump_comp);
}

vector<vector<double>> heston_jump_mc(HestonParams hp, JumpParams jp, double mu, double dt, int days, int paths, int seed)
{
    vector<vector<double>> results(paths, vector<double>(days + 1, 0.0));
    unsigned int num_threads = thread::hardware_concurrency();
    if (num_threads == 0)
        num_threads = 4;

    vector<thread> threads;
    int chunk_size = paths / num_threads;

    auto worker = [&](int start_idx, int end_idx, int thread_seed)
    {
        mt19937 gen(thread_seed);
        normal_distribution<double> norm(0.0, 1.0);
        uniform_real_distribution<double> unif(0.0, 1.0);

        double expected_jump = exp(jp.mu_J + 0.5 * pow(jp.sigma_J, 2.0)) - 1.0;

        for (int p = start_idx; p < end_idx; ++p)
        {
            double S = hp.S0;
            double v = hp.v0;

            results[p][0] = S;

            for (int t = 0; t < days; ++t)
            {
                double Z1 = norm(gen);
                double Z2 = norm(gen);

                double Z_S = Z1;
                double Z_v = hp.rho * Z1 + sqrt(1.0 - hp.rho * hp.rho) * Z2;

                double v_next = max(0.0, v + hp.kappa * (hp.theta - max(0.0, v)) * dt + hp.xi * sqrt(max(0.0, v)) * sqrt(dt) * Z_v);

                double jump_multiplier = 0.0;
                if (unif(gen) < jp.lambda * dt)
                {
                    jump_multiplier = exp(norm(gen) * jp.sigma_J + jp.mu_J) - 1.0;
                }

                double drift = (mu - jp.lambda * expected_jump - 0.5 * max(0.0, v)) * dt;
                double diffusion = sqrt(max(0.0, v) * dt) * Z_S;

                S = S * exp(drift + diffusion) + S * jump_multiplier;
                v = v_next;

                // FIX 3: Record the balance for each daily step
                results[p][t + 1] = S;
            }
        }
    };

    for (unsigned int i = 0; i < num_threads; ++i)
    {
        int start_idx = i * chunk_size;
        int end_idx = (i == num_threads - 1) ? paths : start_idx + chunk_size;
        threads.emplace_back(worker, start_idx, end_idx, seed + i);
    }

    for (auto &t : threads)
    {
        t.join();
    }

    return results;
}