#pragma once

#include "utils.h"
#include "matrix.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Solver
{
protected:
  // Iterations.
  int m_num_max_iters;
  // Tolerance.
  double m_tolerance;
  // The norm.
  double m_nrm[4];
  // Temporary storage.
  double *m_wk0, *m_vec0;
  // Event to read from async memcpys.
  cudaEvent_t m_e0;

public:
  // Constructor.
  Solver();

  // Destructor.
  virtual ~Solver();

  // Virtual functions.
  virtual void setup(Context *ctx, const Matrix *A);
  // Smooth x by running one iteration of Jacobi.
  virtual void smooth(Context *ctx, const Matrix *A, double *x, const double *b);
  // Smooth x by running one iteration of Jacobi.
  virtual void solve(Context *ctx, const Matrix *A, double *x, const double *b) = 0;

protected:
  // Compute the norm and determine convergence.
  bool converged(Context *ctx, int n, const double *r);
  // Get a scalar value.
  void get_from_device(Context *ctx, const double *src_d, double *dst_h, size_t sz, bool blocking = true);
  // Print the last norm computed.
  void print_norm(const char *msg);
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
