#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include "poisson2d_serial.h"

int main(int argc, char **argv) {

    int size_x_logical = 10;
    int size_y_logical = 10;
    int iter_max = 1000;

    const char* pch;
    for ( int n = 1; n < argc; n++ ) 
    {
        pch = strstr(argv[n], "-sizex=");

        if(pch != NULL) {
            size_x_logical = atoi ( argv[n] + 7 ); continue;
        }
        pch = strstr(argv[n], "-sizey=");

        if(pch != NULL) {
            size_y_logical = atoi ( argv[n] + 7 ); continue;
        }
        pch = strstr(argv[n], "-iters=");

        if(pch != NULL) {
            iter_max = atoi ( argv[n] + 7 ); continue;
        }
    }

    int size_x = size_x_logical + 2; // Add padding for halo regions
    int size_y = size_y_logical + 2;

    // Allocate memory
    std::vector<double> u(size_x * size_y, 0.0);
    std::vector<double> u_new(size_x * size_y, 0.0);
    std::vector<double> f(size_x * size_y, 1.0); // Source term

    // Call the Poisson solver
    poisson2d_serial(u.data(), u_new.data(), f.data(), size_x, size_y, iter_max);

    // Print a value to check the result
    std::cout << "u[(size_y/2) * size_x + (size_x/2)] = " << u[(size_y/2) * size_x + (size_x/2)] << std::endl;

    return 0;
}