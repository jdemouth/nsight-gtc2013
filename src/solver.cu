#include "solver.h"
#include <algorithm>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int BLOCK_SIZE >
__global__ 
void reduce_l2_norm_kernel( int n, const double *x, double *res )
{
  const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

  const int warp_id = warpid();
  const int lane_id = laneid();

  __shared__ volatile double smem[BLOCK_SIZE+4*NUM_WARPS];
  double my_x = 0.0;
  for( int idx = threadIdx.x, n_4 = 4*n ; idx < n_4 ; idx += BLOCK_SIZE )
    my_x += x[idx];
  smem[threadIdx.x] = my_x;
  
  for( int offset = WARP_SIZE / 2 ; offset > 2 ; offset >>= 1 )
    if( lane_id < offset )
      smem[threadIdx.x] = my_x += smem[threadIdx.x+offset];
  if( lane_id < 4 )
    smem[BLOCK_SIZE + 4*warp_id + lane_id] = my_x;
  __syncthreads();

  if( threadIdx.x < 4*NUM_WARPS/2 )
    my_x = smem[BLOCK_SIZE + threadIdx.x];
  for( int offset = 4*NUM_WARPS/2 ; offset > 2 ; offset >>= 1 )
    if( threadIdx.x < offset )
      smem[BLOCK_SIZE + threadIdx.x] = my_x += smem[BLOCK_SIZE + threadIdx.x + offset];
  if( threadIdx.x < 4 )
    res[threadIdx.x] = sqrt(my_x);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int BLOCK_SIZE >
__global__
void l2_norm_kernel_v0( int n, const double *x, double *nrm )
{
  int idx = blockIdx.x*BLOCK_SIZE + threadIdx.x;

  // SMEM for the reduction.
  __shared__ volatile double smem[4][BLOCK_SIZE];

  // My norm.
  double my_nrm[4];

  // Initialize to 0.
  for( int k = 0 ; k < 4 ; ++k )
    my_nrm[k] = 0.0;

  // Iterate over the elements.
  for( ; idx < n ; idx += BLOCK_SIZE*gridDim.x )
  {
    double my_x[4];

    // Load my elements.
    for( int k = 0 ; k < 4 ; ++k )
      my_x[k] = x[4*idx + k];

    // Update my term.
    for( int k = 0 ; k < 4 ; ++k )
      my_nrm[k] += my_x[k] * my_x[k];
  }

  // Threads store their elements in SMEM.
  for( int k = 0 ; k < 4 ; ++k )
    smem[k][threadIdx.x] = my_nrm[k];
  __syncthreads();

  // Reduce in the block.
  for( int offset = BLOCK_SIZE / 2 ; offset > 0 ; offset >>= 1 )
  {
    if( threadIdx.x < offset )
    {
      smem[0][threadIdx.x] = my_nrm[0] += smem[0][threadIdx.x + offset];
      smem[1][threadIdx.x] = my_nrm[1] += smem[1][threadIdx.x + offset];
      smem[2][threadIdx.x] = my_nrm[2] += smem[2][threadIdx.x + offset];
      smem[3][threadIdx.x] = my_nrm[3] += smem[3][threadIdx.x + offset];
    }
    __syncthreads();
  }

  // Store the results.
  if( threadIdx.x == 0 )
  {
    nrm[4*blockIdx.x + 0] = my_nrm[0];
    nrm[4*blockIdx.x + 1] = my_nrm[1];
    nrm[4*blockIdx.x + 2] = my_nrm[2];
    nrm[4*blockIdx.x + 3] = my_nrm[3];
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum { BLOCK_SIZE = 256 };

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Solver::Solver() : 
  m_num_max_iters(100), 
  m_tolerance(1.0e-4),
  m_wk0(NULL),
  m_vec0(NULL)
{
  CUDA_SAFE_CALL( cudaEventCreate(&m_e0) );
}

// --------------------------------------------------------------------------------------------------------------------

Solver::~Solver()
{
  CUDA_SAFE_CALL( cudaFree(m_wk0) );
  CUDA_SAFE_CALL( cudaFree(m_vec0) );

  CUDA_SAFE_CALL( cudaEventDestroy(m_e0) );
}

// --------------------------------------------------------------------------------------------------------------------

bool Solver::converged(Context *ctx, int n, const double *r)
{
  int grid_size = std::min( (int) MAX_GRID_SIZE, (n+BLOCK_SIZE-1) / BLOCK_SIZE );
  l2_norm_kernel_v0<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, ctx->get_stream(0)>>>(n, r, m_wk0);
  CUDA_SAFE_CALL( cudaGetLastError() );

  reduce_l2_norm_kernel<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, ctx->get_stream(0)>>>(grid_size, m_wk0, m_vec0);
  CUDA_SAFE_CALL( cudaGetLastError() );
  get_from_device(ctx, m_vec0, m_nrm, 4*sizeof(double));
  
  bool ok = true;
  for( int k = 0 ; ok && k < 4 ; ++k )
    ok = m_nrm[k] <= m_tolerance;
  return ok;
}

// --------------------------------------------------------------------------------------------------------------------

void Solver::get_from_device(Context *ctx, const double *src_d, double *dst_h, size_t sz, bool blocking)
{
  CUDA_SAFE_CALL( cudaMemcpyAsync(dst_h, src_d, sz, cudaMemcpyDeviceToHost, ctx->get_stream(0)) );
  if( !blocking )
    return;
  CUDA_SAFE_CALL( cudaEventRecord(m_e0, ctx->get_stream(0)) );
  CUDA_SAFE_CALL( cudaEventSynchronize(m_e0) );
}

// --------------------------------------------------------------------------------------------------------------------

void Solver::print_norm(const char *msg)
{
  printf( "** %s [ %15.6e %15.6e %15.6e %15.6e ]      **\n", msg, m_nrm[0], m_nrm[1], m_nrm[2], m_nrm[3] );
}

// --------------------------------------------------------------------------------------------------------------------

void Solver::setup(Context *ctx, const Matrix *A)
{
  const int n = 4*A->get_num_rows();
  
  CUDA_SAFE_CALL( cudaMalloc((void**) &m_wk0, n*sizeof(double)) );
  CUDA_SAFE_CALL( cudaMalloc((void**) &m_vec0, 4*sizeof(double)) );
}

// --------------------------------------------------------------------------------------------------------------------

void Solver::smooth(Context *ctx, const Matrix *A, double *x, const double *b)
{
  this->solve(ctx, A, x, b);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
