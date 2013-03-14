#pragma once

#include "utils.h"
#include "matrix.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// The spmv function computes y = a*A*x + b*y where A is a sparse matrix with 4x4 blocks, x and y are vectors.

void spmv(Context *ctx, const Matrix *A, const double *x, double *y);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
