#pragma once

#include "matrix.h"
#include "utils.h"
#include "solver.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Bicgstab : public Solver
{
  // The preconditioner.
  Solver *m_preconditioner;
  
  // Temporary storage.
  double *m_y;
  double *m_p, *m_r, *m_r_star, *m_s;
  double *m_Mp, *m_AMp;
  double *m_Ms, *m_AMs;
  double *m_scal0, *m_scal1;

public:
  // Create an uninitialized solver.
  Bicgstab();

  // Destructor.
  ~Bicgstab();

  // Setup the Jacobi solver.
  void setup(Context *ctx, const Matrix *A);
  // Smooth x by running one iteration of Jacobi.
  void solve(Context *ctx, const Matrix *A, double *x, const double *b);
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
