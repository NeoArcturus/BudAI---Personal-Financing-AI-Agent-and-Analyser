#ifndef CALCULATION_H
#define CALCULATION_H

#include <vector>
#include <complex>

struct HestonParams
{
    double S0;
    double v0;
    double kappa;
    double theta;
    double xi;
    double rho;
};

struct JumpParams
{
    double lambda;
    double mu_J;
    double sigma_J;
};

void fft(std::vector<std::complex<double>> &x);

std::complex<double> bates_characteristic_function(double u, double T, double r, HestonParams hp, JumpParams jp);

std::vector<std::vector<double>> heston_jump_mc(HestonParams hp, JumpParams jp, double mu, double dt, int days, int paths, int seed);

#endif