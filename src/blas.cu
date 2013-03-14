#include "blas.h"
#include <algorithm>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int BLOCK_SIZE >
__global__ 
void axpby_kernel( const int n, 
                   const double a, 
                   const double *x, 
                   const double b, 
                   const double *y,
                         double *z)
{
  // One thread per row.
  int idx = blockIdx.x*BLOCK_SIZE + threadIdx.x;

  // Iterate over the rows of the matrix.
  for( ; idx < n ; idx += BLOCK_SIZE*gridDim.x )
  {
    // Load x and y.
    double my_x = x[idx];
    double my_y = y[idx];

    //if( idx < 4 )
    //  printf( "axpby: [ %12.8f %12.8f ]\n", my_x, my_y );

    // Store the results.
    z[idx] = a*my_x + b*my_y;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int BLOCK_SIZE >
__global__ 
void axpbypcz_kernel( const int n, 
                      const double a, 
                      const double *x, 
                      const double b, 
                      const double *y,
                      const double c,
                      const double *z,
                            double *w)
{
  // One thread per row.
  int idx = blockIdx.x*BLOCK_SIZE + threadIdx.x;

  // Iterate over the rows of the matrix.
  for( ; idx < n ; idx += BLOCK_SIZE*gridDim.x )
  {
    // Load x and y.
    double my_x = x[idx];
    double my_y = y[idx];
    double my_z = z[idx];

    // Store the results.
    w[idx] = a*my_x + b*my_y + c*my_z;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int BLOCK_SIZE >
__global__ 
void dot_kernel_v0( const int n, const double *__restrict x, const double *__restrict y, double *res )
{
  int idx = blockIdx.x*BLOCK_SIZE + threadIdx.x;

  // Shared memory to compute the block reduction.
  __shared__ volatile double smem[BLOCK_SIZE];

  // My dot values.
  double my_x, my_y, my_res = 0.0;

  // Serial reduction.
  for( ; idx < n ; idx += BLOCK_SIZE*gridDim.x )
  {
    // Load x and y.
    my_x = x[idx];
    my_y = y[idx];

    // Update my local value.
    my_res += my_x*my_y;
  }

  // Store the result in SMEM.
  smem[threadIdx.x] = my_res;

  // Make sure all threads have written their values.
  __syncthreads();

  // Block-wide reduction.
  for( int offset = BLOCK_SIZE / 2 ; offset > 0 ; offset /= 2 )
  {
    if( threadIdx.x < offset )
      smem[threadIdx.x] = my_res += smem[threadIdx.x + offset];
    __syncthreads();
  }

  // Store the result.
  if( threadIdx.x == 0 )
    res[blockIdx.x] = my_res;
}

// --------------------------------------------------------------------------------------------------------------------

template< int BLOCK_SIZE >
__global__ 
void dot_kernel_v1( const int n, const double *__restrict x, const double *__restrict y, double *res )
{
  const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

  int idx = blockIdx.x*BLOCK_SIZE + threadIdx.x;

  double my_x, my_y, my_res = 0.0;

  // Thread coordinates in the block.
  const int warp_id = warpid();
  const int lane_id = laneid();

  // Shared memory to compute the block reduction.
  __shared__ volatile double smem[BLOCK_SIZE + NUM_WARPS];

  // Serial reduction.
  for( ; idx < n ; idx += BLOCK_SIZE*gridDim.x )
  {
    // Load x and y.
    my_x = x[idx];
    my_y = y[idx];

    // Update my local value.
    my_res += my_x*my_y;
  }

  // Store the result in SMEM.
  smem[threadIdx.x] = my_res;
  for( int offset = WARP_SIZE / 2 ; offset > 0 ; offset >>= 1 )
    if( lane_id < offset )
      smem[threadIdx.x] = my_res += smem[threadIdx.x+offset];

  // Make sure all threads have written their values.
  if( lane_id == 0 )
    smem[BLOCK_SIZE+warp_id] = my_res;
  __syncthreads();

  // First warp reduction.
  if( threadIdx.x < NUM_WARPS )
    my_res = smem[BLOCK_SIZE + threadIdx.x];
  for( int offset = NUM_WARPS / 2 ; offset > 0 ; offset >>= 1 )
    if( threadIdx.x < offset )
      smem[BLOCK_SIZE + threadIdx.x] = my_res += smem[BLOCK_SIZE + threadIdx.x + offset];

  // Store the result.
  if( threadIdx.x == 0 )
    res[blockIdx.x] = my_res;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int BLOCK_SIZE >
__global__ 
void reduce_kernel( int n, const double *x, double *res )
{
  const int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;

  const int warp_id = warpid();
  const int lane_id = laneid();

  __shared__ volatile double smem[BLOCK_SIZE + NUM_WARPS];
  double my_x = 0.0;
  if( threadIdx.x < n )
    my_x = x[threadIdx.x];
  smem[threadIdx.x] = my_x;

  for( int offset = WARP_SIZE / 2 ; offset > 0 ; offset >>= 1 )
    if( lane_id < offset )
      smem[threadIdx.x] = my_x += smem[threadIdx.x+offset];
  if( lane_id == 0 )
    smem[BLOCK_SIZE + warp_id] = my_x;
  __syncthreads();

  if( threadIdx.x < NUM_WARPS / 2 )
    my_x = smem[BLOCK_SIZE + threadIdx.x];
  for( int offset = NUM_WARPS / 2 ; offset > 0 ; offset >>= 1 )
    if( threadIdx.x < offset )
      smem[BLOCK_SIZE+threadIdx.x] = my_x += smem[BLOCK_SIZE + threadIdx.x+offset];
  if( threadIdx.x == 0 )
    res[0] = my_x;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum { BLOCK_SIZE = 256 };

void axpby(Context *ctx, int n, double a, const double *x, 
                                double b, const double *y,
                                                double *z)
{
  const int grid_size = std::min( (int) MAX_GRID_SIZE, (4*n+BLOCK_SIZE-1) / BLOCK_SIZE );
  axpby_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, ctx->get_stream(0)>>>(4*n, a, x, b, y, z);
  CUDA_SAFE_CALL( cudaGetLastError() );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void axpbypcz(Context *ctx, int n, double a, const double *x, 
                                   double b, const double *y, 
                                   double c, const double *z,
                                                   double *w)
{
  const int grid_size = std::min( (int) MAX_GRID_SIZE, (4*n+BLOCK_SIZE-1) / BLOCK_SIZE );
  axpbypcz_kernel<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, ctx->get_stream(0)>>>(4*n, a, x, b, y, c, z, w);
  CUDA_SAFE_CALL( cudaGetLastError() );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void dot( Context *ctx, int n, const double *x, const double *y, double *res, double *wk )
{
  bool local_wk = wk == NULL;
  if( local_wk )
  {
    CUDA_SAFE_CALL( cudaMalloc((void**) &wk, BLOCK_SIZE*sizeof(double)) );
  }

  int grid_size = 0;
  switch(ctx->dot)
  {
  case 0:
    grid_size = std::min( (int) BLOCK_SIZE, (4*n+BLOCK_SIZE-1) / BLOCK_SIZE );
    dot_kernel_v0<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, ctx->get_stream(0)>>>(4*n, x, y, wk);
    CUDA_SAFE_CALL( cudaGetLastError() );

    reduce_kernel<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, ctx->get_stream(0)>>>(grid_size, wk, res);
    CUDA_SAFE_CALL( cudaGetLastError() );
    break;
  case 1:
    grid_size = std::min( (int) BLOCK_SIZE, (4*n+BLOCK_SIZE-1) / BLOCK_SIZE );
    dot_kernel_v1<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, ctx->get_stream(0)>>>(4*n, x, y, wk);
    CUDA_SAFE_CALL( cudaGetLastError() );

    reduce_kernel<BLOCK_SIZE><<<1, BLOCK_SIZE, 0, ctx->get_stream(0)>>>(grid_size, wk, res);
    CUDA_SAFE_CALL( cudaGetLastError() );
    break;
  default:
    std::fprintf(stderr, "Invalid version for dot kernel=%d, valid values=[0,1]\n", ctx->dot);
    std::exit(1);
  }

  if( local_wk )
  {
    CUDA_SAFE_CALL( cudaFree(wk) );
    wk = NULL;
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
