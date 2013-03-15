#include "spmv.h"
#include <algorithm>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int BLOCK_SIZE >
__global__ 
  void spmv_kernel_v0( const int A_num_rows, 
                                const int A_num_vals,
                                const int *A_rows,
                                const int *A_cols,
                                const double *A_vals,
                                const double *A_diag,
                                const double *x,
                                      double *y )
{
  // One thread per row.
  int row = blockIdx.x*BLOCK_SIZE + threadIdx.x;

  // Iterate over the rows of the matrix.
  for( ; row < A_num_rows ; row += BLOCK_SIZE*gridDim.x )
  {
    double my_A[16], my_x[4], my_y[4];

    // Load the diagonal.
    for( int k = 0 ; k < 4 ; ++k )
    {
      my_A[4*k+0] = A_diag[4*k*A_num_rows + 4*row+0];
      my_A[4*k+1] = A_diag[4*k*A_num_rows + 4*row+1];
      my_A[4*k+2] = A_diag[4*k*A_num_rows + 4*row+2];
      my_A[4*k+3] = A_diag[4*k*A_num_rows + 4*row+3];
    }

    // Load the x diagonal.
    for( int k = 0 ; k < 4 ; ++k )
      my_x[k] = x[4*row+k];

    // Compute the product.
    for( int k = 0 ; k < 4 ; ++k )
    {
      my_y[k]  = my_A[4*k+0] * my_x[0];
      my_y[k] += my_A[4*k+1] * my_x[1];
      my_y[k] += my_A[4*k+2] * my_x[2];
      my_y[k] += my_A[4*k+3] * my_x[3];
    }

    // Each thread iterates over its row.
    for( int it = A_rows[row], end = A_rows[row+1] ; it < end ; ++it )
    {
      const int col = A_cols[it];

      // Load the matrix block.
      for( int k = 0 ; k < 4 ; ++k )
      {
        my_A[4*k+0] = A_vals[4*k*A_num_vals + 4*it + 0];
        my_A[4*k+1] = A_vals[4*k*A_num_vals + 4*it + 1];
        my_A[4*k+2] = A_vals[4*k*A_num_vals + 4*it + 2];
        my_A[4*k+3] = A_vals[4*k*A_num_vals + 4*it + 3];
      }

      // Load the x block.
      for( int k = 0 ; k < 4 ; ++k )
        my_x[k] = x[4*col+k];

      // Run the product.
      for( int k = 0 ; k < 4 ; ++k )
      {
        my_y[k] += my_A[4*k+0] * my_x[0];
        my_y[k] += my_A[4*k+1] * my_x[1];
        my_y[k] += my_A[4*k+2] * my_x[2];
        my_y[k] += my_A[4*k+3] * my_x[3];
      }
    }

    // Store the result to GMEM.
    for( int k = 0 ; k < 4 ; ++k )
      y[4*row+k] = my_y[k];
  }
}

// --------------------------------------------------------------------------------------------------------------------

template< int BLOCK_SIZE >
__global__ __launch_bounds__(BLOCK_SIZE, 6)
void spmv_kernel_v1( const int A_num_rows, 
                                const int A_num_vals,
                                const int *A_rows,
                                const int *A_cols,
                                const double *A_vals,
                                const double *A_diag,
                                const double *x,
                                      double *y )
{
  // One thread per row.
  int row = blockIdx.x*BLOCK_SIZE + threadIdx.x;

  // Iterate over the rows of the matrix.
  for( ; row < A_num_rows ; row += BLOCK_SIZE*gridDim.x )
  {
    double my_A[16], my_x[4], my_y[4];

    // Load the diagonal.
    for( int k = 0 ; k < 4 ; ++k )
    {
      my_A[4*k+0] = A_diag[4*k*A_num_rows + 4*row+0];
      my_A[4*k+1] = A_diag[4*k*A_num_rows + 4*row+1];
      my_A[4*k+2] = A_diag[4*k*A_num_rows + 4*row+2];
      my_A[4*k+3] = A_diag[4*k*A_num_rows + 4*row+3];
    }

    // Load the x diagonal.
    for( int k = 0 ; k < 4 ; ++k )
      my_x[k] = x[4*row+k];

    // Compute the product.
    for( int k = 0 ; k < 4 ; ++k )
    {
      my_y[k]  = my_A[4*k+0] * my_x[0];
      my_y[k] += my_A[4*k+1] * my_x[1];
      my_y[k] += my_A[4*k+2] * my_x[2];
      my_y[k] += my_A[4*k+3] * my_x[3];
    }

    // Each thread iterates over its row.
    for( int it = A_rows[row], end = A_rows[row+1] ; it < end ; ++it )
    {
      const int col = A_cols[it];

      // Load the matrix block.
      for( int k = 0 ; k < 4 ; ++k )
      {
        my_A[4*k+0] = __ldg(&A_vals[4*k*A_num_vals + 4*it + 0]);
        my_A[4*k+1] = __ldg(&A_vals[4*k*A_num_vals + 4*it + 1]);
        my_A[4*k+2] = __ldg(&A_vals[4*k*A_num_vals + 4*it + 2]);
        my_A[4*k+3] = __ldg(&A_vals[4*k*A_num_vals + 4*it + 3]);
      }

      // Load the x block.
      for( int k = 0 ; k < 4 ; ++k )
        my_x[k] = x[4*col+k];

      // Run the product.
      for( int k = 0 ; k < 4 ; ++k )
      {
        my_y[k] += my_A[4*k+0] * my_x[0];
        my_y[k] += my_A[4*k+1] * my_x[1];
        my_y[k] += my_A[4*k+2] * my_x[2];
        my_y[k] += my_A[4*k+3] * my_x[3];
      }
    }

    // Store the result to GMEM.
    for( int k = 0 ; k < 4 ; ++k )
      y[4*row+k] = my_y[k];
  }
}

// --------------------------------------------------------------------------------------------------------------------

template< int BLOCK_SIZE >
__global__ void spmv_kernel_v2( const int A_num_rows, 
                                const int A_num_vals,
                                const int *A_rows,
                                const int *A_cols,
                                const double *A_vals,
                                const double *A_diag,
                                const double *x,
                                      double *y )
{
  // One thread per row.
  int row = blockIdx.x*BLOCK_SIZE + threadIdx.x;

  // Iterate over the rows of the matrix.
  for( ; row < A_num_rows ; row += BLOCK_SIZE*gridDim.x )
  {
    double my_A[16], my_x[4], my_y[4];

    // Load the diagonal.
    for( int k = 0 ; k < 4 ; ++k )
    {
      my_A[4*k+0] = A_diag[4*k*A_num_rows + 4*row+0];
      my_A[4*k+1] = A_diag[4*k*A_num_rows + 4*row+1];
      my_A[4*k+2] = A_diag[4*k*A_num_rows + 4*row+2];
      my_A[4*k+3] = A_diag[4*k*A_num_rows + 4*row+3];
    }

    // Load the x diagonal.
    for( int k = 0 ; k < 4 ; ++k )
      my_x[k] = __ldg(&x[4*row+k]);

    // Compute the product.
    for( int k = 0 ; k < 4 ; ++k )
    {
      my_y[k]  = my_A[4*k+0] * my_x[0];
      my_y[k] += my_A[4*k+1] * my_x[1];
      my_y[k] += my_A[4*k+2] * my_x[2];
      my_y[k] += my_A[4*k+3] * my_x[3];
    }

    // Each thread iterates over its row.
    for( int it = A_rows[row], end = A_rows[row+1] ; it < end ; ++it )
    {
      const int col = A_cols[it];

      // Load the matrix block.
      for( int k = 0 ; k < 4 ; ++k )
      {
        my_A[4*k+0] = A_vals[4*k*A_num_vals + 4*it + 0];
        my_A[4*k+1] = A_vals[4*k*A_num_vals + 4*it + 1];
        my_A[4*k+2] = A_vals[4*k*A_num_vals + 4*it + 2];
        my_A[4*k+3] = A_vals[4*k*A_num_vals + 4*it + 3];
      }

      // Load the x block.
      for( int k = 0 ; k < 4 ; ++k )
        my_x[k] = __ldg(&x[4*col+k]);

      // Run the product.
      for( int k = 0 ; k < 4 ; ++k )
      {
        my_y[k] += my_A[4*k+0] * my_x[0];
        my_y[k] += my_A[4*k+1] * my_x[1];
        my_y[k] += my_A[4*k+2] * my_x[2];
        my_y[k] += my_A[4*k+3] * my_x[3];
      }
    }

    // Store the result to GMEM.
    for( int k = 0 ; k < 4 ; ++k )
      y[4*row+k] = my_y[k];
  }
}

// --------------------------------------------------------------------------------------------------------------------

template< int BLOCK_SIZE >
__global__ __launch_bounds__(BLOCK_SIZE) 
  void spmv_kernel_v3( const int A_num_rows, 
                                const int A_num_vals, 
                                const int *A_rows,
                                const int *A_cols,
                                const double *A_vals,
                                const double *A_diag,
                                const double *x,
                                      double */*__restrict*/ y )
{
  const int NUM_ROWS_PER_BLOCK = BLOCK_SIZE / 4;

  // Shared memory to run the reduction.
  __shared__ volatile double smem[4][BLOCK_SIZE];

  // Lane in the quad of threads.
  const int lane_id_mod_4 = laneid() % 4;

  // One thread per row.
  int row = blockIdx.x*NUM_ROWS_PER_BLOCK + threadIdx.x/4;

  // Iterate over the rows of the matrix.
  for( ; row < A_num_rows ; row += NUM_ROWS_PER_BLOCK*gridDim.x )
  {
    double my_A[4], my_x, my_y[4];

    // Load the diagonal.
    for( int k = 0 ; k < 4 ; ++k )
      my_A[k] = A_diag[4*k*A_num_rows + 4*row + lane_id_mod_4];

    // Load my x.
    my_x = __ldg(&x[4*row+lane_id_mod_4]);

    // Compute the product.
    my_y[0] = my_A[0]*my_x;
    my_y[1] = my_A[1]*my_x;
    my_y[2] = my_A[2]*my_x;
    my_y[3] = my_A[3]*my_x;

    // Each thread iterates over its row.
    for( int it = A_rows[row], end = A_rows[row+1] ; it < end ; ++it )
    {
      const int col = A_cols[it];

      // Load the matrix block.
      for( int k = 0 ; k < 4 ; ++k )
        my_A[k] = A_vals[4*k*A_num_vals + 4*it + lane_id_mod_4];

      // Load the x block.
      my_x = __ldg(&x[4*col + lane_id_mod_4]);

      // Run the product.
      my_y[0] += my_A[0]*my_x;
      my_y[1] += my_A[1]*my_x;
      my_y[2] += my_A[2]*my_x;
      my_y[3] += my_A[3]*my_x;
    }

    // Store the values to SMEM.
    smem[0][threadIdx.x] = my_y[0];
    smem[1][threadIdx.x] = my_y[1];
    smem[2][threadIdx.x] = my_y[2];
    smem[3][threadIdx.x] = my_y[3];

    // Root of the segment.
    const int offset = threadIdx.x - lane_id_mod_4; 

    // Perform a reduction.
    smem[lane_id_mod_4][threadIdx.x] += smem[lane_id_mod_4][offset + ((lane_id_mod_4+1)&3)];
    smem[lane_id_mod_4][threadIdx.x] += smem[lane_id_mod_4][offset + ((lane_id_mod_4+2)&3)];
    smem[lane_id_mod_4][threadIdx.x] += smem[lane_id_mod_4][offset + ((lane_id_mod_4+3)&3)];

    // Store the result to GMEM.
    y[4*row + lane_id_mod_4] = smem[lane_id_mod_4][threadIdx.x];
  }
}

// --------------------------------------------------------------------------------------------------------------------

template< int BLOCK_SIZE >
__global__ __launch_bounds__(BLOCK_SIZE, 8) 
void spmv_kernel_v4( const int A_num_rows, 
                     const int A_num_vals, 
                     const int *A_rows,
                     const int *A_cols,
                     const double *A_vals,
                     const double *A_diag,
                     const double *x,
                           double */*__restrict*/ y )
{
  const int NUM_ROWS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

  // Shared memory to run the reduction.
  __shared__ volatile double smem[BLOCK_SIZE];
  // Shared memory to broadcast column ids.
  __shared__ volatile int scols[BLOCK_SIZE];

  // Lane in the warp.
  const int warp_id = warpid();
  const int lane_id = laneid();

  // Lane in the quad of threads.
  const int lane_id_mod_4 = lane_id % 4;

  // One thread per row.
  int row = blockIdx.x*NUM_ROWS_PER_BLOCK + threadIdx.x/WARP_SIZE;

  // Iterate over the rows of the matrix.
  for( ; row < A_num_rows ; row += NUM_ROWS_PER_BLOCK*gridDim.x )
  {
    double my_A[4], my_y[4], my_x = 0.0;

    // Load the diagonal.
    if( lane_id < 4 )
    {
      #pragma unroll
      for( int k = 0 ; k < 4 ; ++k )
        my_A[k] = A_diag[4*k*A_num_rows + 4*row+lane_id_mod_4];
      my_x = __ldg(&x[4*row+lane_id_mod_4]);
    }

    // Compute the product.
    my_y[0] = my_A[0]*my_x;
    my_y[1] = my_A[1]*my_x;
    my_y[2] = my_A[2]*my_x;
    my_y[3] = my_A[3]*my_x; 

    // Each thread iterates over its row.
    for( int it = A_rows[row]+lane_id/4, end = A_rows[row+1] ; it < end ; it += WARP_SIZE/4 )
    {
      // Load the column.
      int col = A_cols[it];
      
      // Load the matrix block and x.
      #pragma unroll
      for( int k = 0 ; k < 4 ; ++k )
        my_A[k] = A_vals[4*k*A_num_vals + 4*it + lane_id_mod_4];
      my_x = __ldg(&x[4*col+lane_id_mod_4]);
      
      // Run the product.
      my_y[0] += my_A[0]*my_x;
      my_y[1] += my_A[1]*my_x;
      my_y[2] += my_A[2]*my_x;
      my_y[3] += my_A[3]*my_x;
    }

    // Run the reduction.
    #pragma unroll
    for( int mask = WARP_SIZE/2 ; mask > 0 ; mask >>= 1 )
    {
      my_y[0] += shfl_xor(my_y[0], mask);
      my_y[1] += shfl_xor(my_y[1], mask);
      my_y[2] += shfl_xor(my_y[2], mask);
      my_y[3] += shfl_xor(my_y[3], mask);
    }

    if( lane_id == 0 )
    {
      smem[threadIdx.x+0] = my_y[0];
      smem[threadIdx.x+1] = my_y[1];
      smem[threadIdx.x+2] = my_y[2];
      smem[threadIdx.x+3] = my_y[3];
    }
    
    // Store the result to GMEM.
    if( lane_id < 4 )
      y[4*row + lane_id_mod_4] = smem[threadIdx.x];
  }
}

// --------------------------------------------------------------------------------------------------------------------

template< int BLOCK_SIZE >
__global__ __launch_bounds__(BLOCK_SIZE, 8) 
void spmv_kernel_v5( const int A_num_rows, 
                     const int A_num_vals, 
                     const int *A_rows,
                     const int *A_cols,
                     const double *A_vals,
                     const double *A_diag,
                     const double *x,
                           double */*__restrict*/ y )
{
  const int NUM_ROWS_PER_BLOCK = BLOCK_SIZE / 16;

  // Shared memory to run the reduction.
  __shared__ volatile double smem[BLOCK_SIZE];
  // Shared memory to broadcast column ids.
  __shared__ volatile int scols[BLOCK_SIZE];

  // Lane in the warp.
  const int warp_id = warpid();
  const int lane_id = laneid();

  // Lane in the halfwarp of threads.
  const int lane_id_mod_16 = lane_id % 16;

  // Lane in the quad of threads.
  const int lane_id_mod_4 = lane_id % 4;

  // One thread per row.
  int row = blockIdx.x*NUM_ROWS_PER_BLOCK + threadIdx.x/16;

  // Iterate over the rows of the matrix.
  for( ; row < A_num_rows ; row += NUM_ROWS_PER_BLOCK*gridDim.x )
  {
    double my_A[4], my_y[4], my_x = 0.0;

    // Load the diagonal.
    if( lane_id_mod_16 < 4 )
    {
      #pragma unroll
      for( int k = 0 ; k < 4 ; ++k )
        my_A[k] = A_diag[4*k*A_num_rows + 4*row + lane_id_mod_4];
      my_x = __ldg(&x[4*row+lane_id_mod_4]);
    }

    // Compute the product.
    my_y[0] = my_A[0]*my_x;
    my_y[1] = my_A[1]*my_x;
    my_y[2] = my_A[2]*my_x;
    my_y[3] = my_A[3]*my_x; 

    // Each thread iterates over its row.
    for( int it = A_rows[row]+lane_id_mod_16/4, end = A_rows[row+1] ; it < end ; it += 4 )
    {
      // Load the column.
      int col = A_cols[it];
      
      // Load the matrix block and x.
      #pragma unroll
      for( int k = 0 ; k < 4 ; ++k )
        my_A[k] = A_vals[4*k*A_num_vals + 4*it + lane_id_mod_4];
      my_x = __ldg(&x[4*col + lane_id_mod_4]);
      
      // Run the product.
      my_y[0] += my_A[0]*my_x;
      my_y[1] += my_A[1]*my_x;
      my_y[2] += my_A[2]*my_x;
      my_y[3] += my_A[3]*my_x;
    }

    // Run the reduction.
    #pragma unroll
    for( int mask = 8 ; mask > 0 ; mask >>= 1 )
    {
      my_y[0] += shfl_xor(my_y[0], mask);
      my_y[1] += shfl_xor(my_y[1], mask);
      my_y[2] += shfl_xor(my_y[2], mask);
      my_y[3] += shfl_xor(my_y[3], mask);
    }

    if( lane_id_mod_16 == 0 )
    {
      smem[threadIdx.x+0] = my_y[0];
      smem[threadIdx.x+1] = my_y[1];
      smem[threadIdx.x+2] = my_y[2];
      smem[threadIdx.x+3] = my_y[3];
    }
    
    // Store the result to GMEM.
    if( lane_id_mod_16 < 4 )
      y[4*row + lane_id_mod_4] = smem[threadIdx.x];
  }
}

// --------------------------------------------------------------------------------------------------------------------

template< int BLOCK_SIZE >
__global__ __launch_bounds__(BLOCK_SIZE, 8) 
void spmv_kernel_v6( const int A_num_rows, 
                     const int A_num_vals, 
                     const int *A_rows,
                     const int *A_cols,
                     const double *A_vals,
                     const double *A_diag,
                     const double *x,
                           double */*__restrict*/ y )
{
  const int NUM_ROWS_PER_BLOCK = BLOCK_SIZE / 16;

  // Shared memory to run the reduction.
  __shared__ volatile double smem[BLOCK_SIZE];

  // Lane in the warp.
  const int warp_id = warpid();
  const int lane_id = laneid();

  // Lane in the halfwarp of threads.
  const int lane_id_mod_16 = lane_id % 16;

  // Lane in the quad of threads.
  const int lane_id_mod_4 = lane_id % 4;

  // One thread per row.
  int row = blockIdx.x*NUM_ROWS_PER_BLOCK + threadIdx.x/16;

  // Iterate over the rows of the matrix.
  for( ; row < A_num_rows ; row += NUM_ROWS_PER_BLOCK*gridDim.x )
  {
    double my_A[4], my_y[4], my_x = 0.0;

    // Load the diagonal.
    if( lane_id_mod_16 < 4 )
    {
      #pragma unroll
      for( int k = 0 ; k < 4 ; ++k )
        my_A[k] = A_diag[4*k*A_num_rows + 4*row + lane_id_mod_4];
      my_x = __ldg(&x[4*row+lane_id_mod_4]);
    }

    // Compute the product.
    my_y[0] = my_A[0]*my_x;
    my_y[1] = my_A[1]*my_x;
    my_y[2] = my_A[2]*my_x;
    my_y[3] = my_A[3]*my_x; 

    // Each thread iterates over its row.
    for( int it = A_rows[row] + lane_id_mod_16/4, end = A_rows[row+1] ; __any(it < end) ; it += 4 )
    {
      if( it >= end )
        continue;

      int col = A_cols[it];
      
      // Load the matrix block and x.
      #pragma unroll
      for( int k = 0 ; k < 4 ; ++k )
        my_A[k] = A_vals[4*k*A_num_vals + 4*it + lane_id_mod_4];
      my_x = __ldg(&x[4*col + lane_id_mod_4]);
      
      // Run the product.
      my_y[0] += my_A[0]*my_x;
      my_y[1] += my_A[1]*my_x;
      my_y[2] += my_A[2]*my_x;
      my_y[3] += my_A[3]*my_x;
    }

    // Run the reduction.
    #pragma unroll
    for( int mask = 8 ; mask > 0 ; mask >>= 1 )
    {
      my_y[0] += shfl_xor(my_y[0], mask);
      my_y[1] += shfl_xor(my_y[1], mask);
      my_y[2] += shfl_xor(my_y[2], mask);
      my_y[3] += shfl_xor(my_y[3], mask);
    }

    if( lane_id_mod_16 == 0 )
    {
      smem[threadIdx.x+0] = my_y[0];
      smem[threadIdx.x+1] = my_y[1];
      smem[threadIdx.x+2] = my_y[2];
      smem[threadIdx.x+3] = my_y[3];
    }
    
    // Store the result to GMEM.
    if( lane_id_mod_16 < 4 )
      y[4*row + lane_id_mod_4] = smem[threadIdx.x];
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum { BLOCK_SIZE = 256, BLOCK_SIZE_V4 = 128 };

// --------------------------------------------------------------------------------------------------------------------

void spmv( Context *ctx, const Matrix *A, const double *x, double *y )
{
  int grid_size = 0;
  switch(ctx->spmv)
  {
  case 0:
    grid_size = std::min( (int) MAX_GRID_SIZE, (A->get_num_rows()+BLOCK_SIZE-1) / BLOCK_SIZE );
    spmv_kernel_v0<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, ctx->get_stream(0)>>>(A->get_num_rows(),
                                                                                 A->get_num_vals()-A->get_num_rows(),
                                                                                 A->get_rows(),
                                                                                 A->get_cols(),
                                                                                 A->get_vals(),
                                                                                 A->get_diag(), 
                                                                                 x,
                                                                                 y);
    CUDA_SAFE_CALL( cudaGetLastError() );
    break;
  case 1:
    grid_size = std::min( (int) MAX_GRID_SIZE, (A->get_num_rows()+BLOCK_SIZE-1) / BLOCK_SIZE );
    spmv_kernel_v1<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, ctx->get_stream(0)>>>(A->get_num_rows(),
                                                                                 A->get_num_vals()-A->get_num_rows(),
                                                                                 A->get_rows(),
                                                                                 A->get_cols(),
                                                                                 A->get_vals(),
                                                                                 A->get_diag(), 
                                                                                 x,
                                                                                 y);
    CUDA_SAFE_CALL( cudaGetLastError() );
    break;
  case 2:
    grid_size = std::min( (int) MAX_GRID_SIZE, (A->get_num_rows()+BLOCK_SIZE-1) / BLOCK_SIZE );
    spmv_kernel_v2<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, ctx->get_stream(0)>>>(A->get_num_rows(),
                                                                                 A->get_num_vals()-A->get_num_rows(),
                                                                                 A->get_rows(),
                                                                                 A->get_cols(),
                                                                                 A->get_vals(),
                                                                                 A->get_diag(), 
                                                                                 x,
                                                                                 y);
    CUDA_SAFE_CALL( cudaGetLastError() );
    break;
  case 3:
    grid_size = std::min( (int) MAX_GRID_SIZE, (A->get_num_rows()+BLOCK_SIZE/4-1) / (BLOCK_SIZE/4) );
    spmv_kernel_v3<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, ctx->get_stream(0)>>>(A->get_num_rows(),
                                                                                 A->get_num_vals()-A->get_num_rows(),
                                                                                 A->get_rows(),
                                                                                 A->get_cols(),
                                                                                 A->get_vals(),
                                                                                 A->get_diag(), 
                                                                                 x,
                                                                                 y);
    CUDA_SAFE_CALL( cudaGetLastError() );
    break;
  case 4:
    grid_size = std::min( (int) MAX_GRID_SIZE, (A->get_num_rows()+BLOCK_SIZE_V4/16-1) / (BLOCK_SIZE_V4/16) );
    spmv_kernel_v4<BLOCK_SIZE_V4><<<grid_size, BLOCK_SIZE_V4, 0, ctx->get_stream(0)>>>(A->get_num_rows(),
                                                                                       A->get_num_vals()-A->get_num_rows(),
                                                                                       A->get_rows(),
                                                                                       A->get_cols(),
                                                                                       A->get_vals(),
                                                                                       A->get_diag(), 
                                                                                       x,
                                                                                       y);
    CUDA_SAFE_CALL( cudaGetLastError() );
    break;
  case 5:
    grid_size = std::min( (int) MAX_GRID_SIZE, (A->get_num_rows()+BLOCK_SIZE_V4/16-1) / (BLOCK_SIZE_V4/16) );
    spmv_kernel_v5<BLOCK_SIZE_V4><<<grid_size, BLOCK_SIZE_V4, 0, ctx->get_stream(0)>>>(A->get_num_rows(),
                                                                                       A->get_num_vals()-A->get_num_rows(),
                                                                                       A->get_rows(),
                                                                                       A->get_cols(),
                                                                                       A->get_vals(),
                                                                                       A->get_diag(), 
                                                                                       x,
                                                                                       y);
    CUDA_SAFE_CALL( cudaGetLastError() );
    break;
  case 6:
    grid_size = std::min( (int) MAX_GRID_SIZE, (A->get_num_rows()+BLOCK_SIZE_V4/16-1) / (BLOCK_SIZE_V4/16) );
    spmv_kernel_v6<BLOCK_SIZE_V4><<<grid_size, BLOCK_SIZE_V4, 0, ctx->get_stream(0)>>>(A->get_num_rows(),
                                                                                       A->get_num_vals()-A->get_num_rows(),
                                                                                       A->get_rows(),
                                                                                       A->get_cols(),
                                                                                       A->get_vals(),
                                                                                       A->get_diag(), 
                                                                                       x,
                                                                                       y);
    CUDA_SAFE_CALL( cudaGetLastError() );
    break;
  default:
    std::fprintf(stderr, "Invalid version for spmv kernel=%d, valid values=[0,1,2,3,4,5,6]\n", ctx->spmv);
    std::exit(1);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


