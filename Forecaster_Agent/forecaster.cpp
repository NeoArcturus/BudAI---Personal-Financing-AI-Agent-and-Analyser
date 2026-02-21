#include <iostream>
#include <random>
#include <cmath>
#include <fstream>
#include <string>

using namespace std;

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        cerr << "Illegal number of arguments given! Usage: ./forecaster <S0> <mu> <sigma> <days> <paths>\n";
    }

    double S0 = stod(argv[1]);
    double mu = stod(argv[2]);
    double sigma = stod(argv[3]);
    int days = stoi(argv[4]);
    int num_paths = stoi(argv[5]);

    double dt = 1.0;

    random_device rd;
    mt19937 gen(rd());
    normal_distribution<> d(0.0, 1.0);

    ofstream file("all_paths.csv");

    for (int p = 0; p < num_paths; ++p)
    {
        double St = S0;
        file << St;

        for (int t = 1; t <= days; ++t)
        {
            double Z = d(gen);

            St = St * exp((mu - 0.5 * sigma * sigma) * dt + sigma * sqrt(dt) * Z);
            file << ", " << St;
        }
        file << "\n";
    }

    file.close();
    cout << "Successfully generated " << num_paths << " paths over " << days << " days, with last check account balance: £" << S0 << "\n";

    return 0;
}