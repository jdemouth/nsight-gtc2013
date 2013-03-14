#pragma once

#include "utils.h"
#include "matrix.h"
#include "solver.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Jacobi : public Solver
{
  // The relaxation weight.
  double m_omega;
  // The inverted diagonal of A.
  double *m_inv_diag;
  // The temporary vector needed at each iteration.
  double *m_y, *m_r;

public:
  // Create an uninitialized solver.
  Jacobi(double omega = 1.0);

  // Destructor.
  ~Jacobi();

  // Setup the Jacobi solver.
  void setup(Context *ctx, const Matrix *A);
  // Smooth x by running one iteration of Jacobi.
  void smooth(Context *ctx, const Matrix *A, double *x, const double *b);
  // Smooth x by running one iteration of Jacobi.
  void solve(Context *ctx, const Matrix *A, double *x, const double *b);

private:
  // Invert the diagonal of A.
  void invert_diag(Context *ctx, const Matrix *A);
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
