#include "bicgstab.h"
#include <algorithm>
#include "utils.h"
#include "spmv.h"
#include "blas.h"
#include "jacobi.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Bicgstab::Bicgstab() :
  m_y(NULL),
  m_p(NULL),
  m_r(NULL),
  m_r_star(NULL),
  m_s(NULL),
  m_Mp(NULL),
  m_AMp(NULL),
  m_Ms(NULL),
  m_AMs(NULL),
  m_scal0(NULL),
  m_scal1(NULL)
{
  m_preconditioner = new Jacobi(1.0);
}

// --------------------------------------------------------------------------------------------------------------------

Bicgstab::~Bicgstab()
{
  CUDA_SAFE_CALL( cudaFree(m_y) );
  CUDA_SAFE_CALL( cudaFree(m_p) );
  CUDA_SAFE_CALL( cudaFree(m_r) );
  CUDA_SAFE_CALL( cudaFree(m_r_star) );
  CUDA_SAFE_CALL( cudaFree(m_s) );
  CUDA_SAFE_CALL( cudaFree(m_Mp) );
  CUDA_SAFE_CALL( cudaFree(m_AMp) );
  CUDA_SAFE_CALL( cudaFree(m_Ms) );
  CUDA_SAFE_CALL( cudaFree(m_AMs) );
  CUDA_SAFE_CALL( cudaFree(m_scal1) );

  delete m_preconditioner;
}

// --------------------------------------------------------------------------------------------------------------------

void Bicgstab::setup(Context *ctx, const Matrix *A)
{
  Solver::setup(ctx, A);

  const size_t sz = 16*A->get_num_rows()*sizeof(double);
  
  CUDA_SAFE_CALL( cudaMalloc((void**) &m_y, sz) );
  CUDA_SAFE_CALL( cudaMalloc((void**) &m_p, sz) );
  CUDA_SAFE_CALL( cudaMalloc((void**) &m_r, sz) );
  CUDA_SAFE_CALL( cudaMalloc((void**) &m_r_star, sz) );
  CUDA_SAFE_CALL( cudaMalloc((void**) &m_s, sz) );
  CUDA_SAFE_CALL( cudaMalloc((void**) &m_Mp, sz) );
  CUDA_SAFE_CALL( cudaMalloc((void**) &m_AMp, sz) );
  CUDA_SAFE_CALL( cudaMalloc((void**) &m_Ms, sz) );
  CUDA_SAFE_CALL( cudaMalloc((void**) &m_AMs, sz) );

  CUDA_SAFE_CALL( cudaMalloc((void**) &m_scal0, sizeof(double)) );
  CUDA_SAFE_CALL( cudaMalloc((void**) &m_scal1, sizeof(double)) );

  m_preconditioner->setup(ctx, A);
}

// --------------------------------------------------------------------------------------------------------------------

void Bicgstab::solve(Context *ctx, const Matrix *A, double *x, const double *b)
{
  const int n = A->get_num_rows();
  // y = A*x
  spmv(ctx, A, x, m_y);
  // r = b - A*x
  axpby(ctx, n, 1.0, b, -1.0, m_y, m_r);

  // p = r
  CUDA_SAFE_CALL( cudaMemcpyAsync(m_p, m_r, 4*n*sizeof(double), cudaMemcpyDeviceToDevice, ctx->get_stream(0)) );
  // r_start = r
  CUDA_SAFE_CALL( cudaMemcpyAsync(m_r_star, m_r, 4*n*sizeof(double), cudaMemcpyDeviceToDevice, ctx->get_stream(0)) );

  // r_r_star_old = dot(r_star, r).
  double r_r_star_old;
  dot(ctx, n, m_r_star, m_r, m_scal0);
  get_from_device(ctx, m_scal0, &r_r_star_old, sizeof(double), false);
  
  bool done = converged(ctx, n, m_r);
  print_norm("INIT. RESID.: ");
  printf("\n#############################################################################################\n\n");
  
  // Iterate.
  for( int iter = 0 ; iter < m_num_max_iters && !done ; ++iter )
  {
    // Mp = M*p
    CUDA_SAFE_CALL( cudaMemsetAsync(m_Mp, 0, 4*n*sizeof(double), ctx->get_stream(0)) );
    m_preconditioner->smooth(ctx, A, m_Mp, m_p);
    
    // AMp = A*Mp
    spmv(ctx, A, m_Mp, m_AMp);
    
    // alpha = (r_j, r_star) / (A*M*p, r_star)
    double alpha = 0.0;
    dot(ctx, n, m_AMp, m_r_star, m_scal0, m_wk0);
    get_from_device(ctx, m_scal0, &alpha, sizeof(double));
    alpha = r_r_star_old / alpha;
        
    // s_j = r_j - alpha * AMp
    axpby(ctx, n, 1.0, m_r, -alpha, m_AMp, m_s);
    if( converged(ctx, n, m_s) )
    {
      // x += alpha*M*p_j
      axpby(ctx, n, alpha, m_Mp, 1.0, x, x);
      break;
    }

    // Ms = M*s_j
    CUDA_SAFE_CALL( cudaMemsetAsync(m_Ms, 0, 4*n*sizeof(double), ctx->get_stream(0)) );
    m_preconditioner->smooth(ctx, A, m_Ms, m_s);
        
    // AMs = A*Ms
    spmv(ctx, A, m_Ms, m_AMs);

    // omega = (AMs, s) / (AMs, AMs)
    double d0 = 0.0;
    dot(ctx, n, m_AMs, m_s, m_scal0, m_wk0);
    get_from_device(ctx, m_scal0, &d0, sizeof(double), false);
    
    double d1 = 0.0;
    dot(ctx, n, m_AMs, m_AMs, m_scal1, m_wk0);
    get_from_device(ctx, m_scal1, &d1, sizeof(double));
    
    // Compute omega.
    double omega = d0 / d1;
        
    // x_{j+1} = x_j + alpha*M*p_j + omega*M*s_j
    axpbypcz(ctx, n, 1.0, x, alpha, m_Mp, omega, m_Ms, x);
    // r_{j+1} = s_j - omega*A*M*s
    axpby(ctx, n, 1.0, m_s, -omega, m_AMs, m_r);

    // beta_j = (r_{j+1}, r_star) / (r_j, r_star) * (alpha/omega)
    double r_r_star_new = 0.0;
    dot(ctx, n, m_r_star, m_r, m_scal0, m_wk0);
    get_from_device(ctx, m_scal0, &r_r_star_new, sizeof(double));
    
    double beta = (r_r_star_new / r_r_star_old) * (alpha / omega);
    r_r_star_old = r_r_star_new;

    // p_{j+1} = r_{j+1} + beta*(p_j - omega*A*M*p)
    axpbypcz(ctx, n, 1.0, m_r, beta, m_p, -beta*omega, m_AMp, m_p);

    // Output the result of the iteration.
    done = converged(ctx, n, m_r);
    char buffer[64];
    sprintf(buffer, "ITERATION %2d: ", iter);
    print_norm(buffer);
  }

  printf("\n#############################################################################################\n\n");
  print_norm("FINAL RESID.: ");
  printf("\n#############################################################################################\n\n");
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
