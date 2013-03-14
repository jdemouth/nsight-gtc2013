#pragma once

#include "utils.h"
#include "matrix.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void axpby(Context *ctx, int n, double a, const double *x, 
                                        double b, const double *y, 
                                                        double *z);

// --------------------------------------------------------------------------------------------------------------------

void axpbypcz(Context *ctx, int n, double a, const double *x, 
                                           double b, const double *y, 
                                           double c, const double *z,
                                                           double *w);

// --------------------------------------------------------------------------------------------------------------------

void dot(Context *ctx, int n, const double *x, const double *y, double *res, double *wk = NULL);

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
