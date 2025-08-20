#ifndef POISSON2D_SERIAL_H
#define POISSON2D_SERIAL_H

void poisson2d_serial(double *u, double *u_new, double *f, int size_x, int size_y, int iter_max) {
    double hx = 1.0 / (size_x - 1);
    double hy = 1.0 / (size_y - 1);
    double hx2 = hx * hx;
    double hy2 = hy * hy;

    for (int iter = 0; iter < iter_max; ++iter) {
        for (int i = 1; i < size_y - 1; ++i) {
            for (int j = 1; j < size_x - 1; ++j) {
                u_new[i * size_x + j] = 0.5 * ((u[(i + 1) * size_x + j] + u[(i - 1) * size_x + j]) / hy2 +
                                             (u[i * size_x + j + 1] + u[i * size_x + j - 1]) / hx2 -
                                             f[i * size_x + j]) / (1.0 / hx2 + 1.0 / hy2);
            }
        }

        // Copy u_new to u
        for (int i = 1; i < size_y - 1; ++i) {
            for (int j = 1; j < size_x - 1; ++j) {
                u[i * size_x + j] = u_new[i * size_x + j];
            }
        }
    }
}
#endif // POISSON2D_SERIAL_H