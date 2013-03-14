#include "jacobi.h"
#include <algorithm>
#include "utils.h"
#include "spmv.h"
#include "blas.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template< int BLOCK_SIZE >
__global__ __launch_bounds__(BLOCK_SIZE)
void jacobi_invert_diag_kernel_v0( const int num_rows, const double *diag, double *inv_diag )
{
  // One thread per diagonal element.
  int row = blockIdx.x*blockDim.x + threadIdx.x; 

  // Iterate over the diagonal elements.
  for( ; row < num_rows ; row += blockDim.x*gridDim.x )
  {
    double tmp[12], mat[16], src[16], inv[16];

    // Load the matrix and transpose the matrix.
    for( int i = 0 ; i < 4 ; ++i )
      for( int j = 0 ; j < 4 ; ++j )
        mat[4*i+j] = diag[4*i*num_rows + 4*row + j];

    // Transpose the matrix.
    for( int k = 0 ; k < 4 ; ++k )
    {
      src[k+ 0] = mat[4*k+0];
      src[k+ 4] = mat[4*k+1];
      src[k+ 8] = mat[4*k+2];
      src[k+12] = mat[4*k+3];
    }

    // Compute the pairs for the first 8 elements (cofactors).
    tmp[ 0] = src[10] * src[15];
    tmp[ 1] = src[11] * src[14];
    tmp[ 2] = src[ 9] * src[15];
    tmp[ 3] = src[11] * src[13];
    tmp[ 4] = src[ 9] * src[14];
    tmp[ 5] = src[10] * src[13];
    tmp[ 6] = src[ 8] * src[15];
    tmp[ 7] = src[11] * src[12];
    tmp[ 8] = src[ 8] * src[14];
    tmp[ 9] = src[10] * src[12];
    tmp[10] = src[ 8] * src[13];
    tmp[11] = src[ 9] * src[12];

    // Compute the 8 first elements (cofactors).
    inv[0]  = tmp[0]*src[5] + tmp[3]*src[6] + tmp[ 4]*src[7];
    inv[0] -= tmp[1]*src[5] + tmp[2]*src[6] + tmp[ 5]*src[7];
    inv[1]  = tmp[1]*src[4] + tmp[6]*src[6] + tmp[ 9]*src[7];
    inv[1] -= tmp[0]*src[4] + tmp[7]*src[6] + tmp[ 8]*src[7];
    inv[2]  = tmp[2]*src[4] + tmp[7]*src[5] + tmp[10]*src[7];
    inv[2] -= tmp[3]*src[4] + tmp[6]*src[5] + tmp[11]*src[7];
    inv[3]  = tmp[5]*src[4] + tmp[8]*src[5] + tmp[11]*src[6];
    inv[3] -= tmp[4]*src[4] + tmp[9]*src[5] + tmp[10]*src[6];
    inv[4]  = tmp[1]*src[1] + tmp[2]*src[2] + tmp[ 5]*src[3];
    inv[4] -= tmp[0]*src[1] + tmp[3]*src[2] + tmp[ 4]*src[3];
    inv[5]  = tmp[0]*src[0] + tmp[7]*src[2] + tmp[ 8]*src[3];
    inv[5] -= tmp[1]*src[0] + tmp[6]*src[2] + tmp[ 9]*src[3];
    inv[6]  = tmp[3]*src[0] + tmp[6]*src[1] + tmp[11]*src[3];
    inv[6] -= tmp[2]*src[0] + tmp[7]*src[1] + tmp[10]*src[3];
    inv[7]  = tmp[4]*src[0] + tmp[9]*src[1] + tmp[10]*src[2];
    inv[7] -= tmp[5]*src[0] + tmp[8]*src[1] + tmp[11]*src[2];

    // Compute the pairs for the second 8 elements (cofactors).
    tmp[ 0] = src[2]*src[7];
    tmp[ 1] = src[3]*src[6];
    tmp[ 2] = src[1]*src[7];
    tmp[ 3] = src[3]*src[5];
    tmp[ 4] = src[1]*src[6];
    tmp[ 5] = src[2]*src[5];
    tmp[ 6] = src[0]*src[7];
    tmp[ 7] = src[3]*src[4];
    tmp[ 8] = src[0]*src[6];
    tmp[ 9] = src[2]*src[4];
    tmp[10] = src[0]*src[5];
    tmp[11] = src[1]*src[4];

    // Compute the second 8 elements (cofactors).
    inv[ 8]  = tmp[ 0]*src[13] + tmp[ 3]*src[14] + tmp[ 4]*src[15];
    inv[ 8] -= tmp[ 1]*src[13] + tmp[ 2]*src[14] + tmp[ 5]*src[15];
    inv[ 9]  = tmp[ 1]*src[12] + tmp[ 6]*src[14] + tmp[ 9]*src[15];
    inv[ 9] -= tmp[ 0]*src[12] + tmp[ 7]*src[14] + tmp[ 8]*src[15];
    inv[10]  = tmp[ 2]*src[12] + tmp[ 7]*src[13] + tmp[10]*src[15];
    inv[10] -= tmp[ 3]*src[12] + tmp[ 6]*src[13] + tmp[11]*src[15];
    inv[11]  = tmp[ 5]*src[12] + tmp[ 8]*src[13] + tmp[11]*src[14];
    inv[11] -= tmp[ 4]*src[12] + tmp[ 9]*src[13] + tmp[10]*src[14];
    inv[12]  = tmp[ 2]*src[10] + tmp[ 5]*src[11] + tmp[ 1]*src[ 9];
    inv[12] -= tmp[ 4]*src[11] + tmp[ 0]*src[ 9] + tmp[ 3]*src[10];
    inv[13]  = tmp[ 8]*src[11] + tmp[ 0]*src[ 8] + tmp[ 7]*src[10];
    inv[13] -= tmp[ 6]*src[10] + tmp[ 9]*src[11] + tmp[ 1]*src[ 8];
    inv[14]  = tmp[ 6]*src[ 9] + tmp[11]*src[11] + tmp[ 3]*src[ 8];
    inv[14] -= tmp[10]*src[11] + tmp[ 2]*src[ 8] + tmp[ 7]*src[ 9];
    inv[15]  = tmp[10]*src[10] + tmp[ 4]*src[ 8] + tmp[ 9]*src[ 9];
    inv[15] -= tmp[ 8]*src[ 9] + tmp[11]*src[10] + tmp[ 5]*src[ 8];

    // Compute the determinant.
    double det = src[0]*inv[0] + src[1]*inv[1] + src[2]*inv[2] + src[3]*inv[3];

    // Compute the inverse of the matrix.
    det = 1.0 / det;
    for( int k = 0 ; k < 16 ; ++k )
      inv[k] *= det;

    // Store the results.
    for( int i = 0 ; i < 4 ; ++i )
      for( int j = 0 ; j < 4 ; ++j )
        inv_diag[4*i*num_rows + 4*row + j] = inv[4*i+j];
  }
}

// --------------------------------------------------------------------------------------------------------------------

template< int BLOCK_SIZE >
__global__ __launch_bounds__(BLOCK_SIZE)
void jacobi_invert_diag_kernel_v1( const int num_rows, const double *__restrict diag, double *__restrict inv_diag )
{
  // One thread per diagonal element.
  int row = blockIdx.x*blockDim.x + threadIdx.x; 

  // Iterate over the diagonal elements.
  for( ; row < num_rows ; row += blockDim.x*gridDim.x )
  {
    double tmp[12], mat[16], src[16], inv[16];

    // Load the matrix and transpose the matrix.
    for( int i = 0 ; i < 4 ; ++i )
      for( int j = 0 ; j < 4 ; ++j )
        mat[4*i+j] = diag[4*i*num_rows + 4*row + j];

    // Transpose the matrix.
    for( int k = 0 ; k < 4 ; ++k )
    {
      src[k+ 0] = mat[4*k+0];
      src[k+ 4] = mat[4*k+1];
      src[k+ 8] = mat[4*k+2];
      src[k+12] = mat[4*k+3];
    }

    // Compute the pairs for the first 8 elements (cofactors).
    tmp[ 0] = src[10] * src[15];
    tmp[ 1] = src[11] * src[14];
    tmp[ 2] = src[ 9] * src[15];
    tmp[ 3] = src[11] * src[13];
    tmp[ 4] = src[ 9] * src[14];
    tmp[ 5] = src[10] * src[13];
    tmp[ 6] = src[ 8] * src[15];
    tmp[ 7] = src[11] * src[12];
    tmp[ 8] = src[ 8] * src[14];
    tmp[ 9] = src[10] * src[12];
    tmp[10] = src[ 8] * src[13];
    tmp[11] = src[ 9] * src[12];

    // Compute the 8 first elements (cofactors).
    inv[0]  = tmp[0]*src[5] + tmp[3]*src[6] + tmp[ 4]*src[7];
    inv[0] -= tmp[1]*src[5] + tmp[2]*src[6] + tmp[ 5]*src[7];
    inv[1]  = tmp[1]*src[4] + tmp[6]*src[6] + tmp[ 9]*src[7];
    inv[1] -= tmp[0]*src[4] + tmp[7]*src[6] + tmp[ 8]*src[7];
    inv[2]  = tmp[2]*src[4] + tmp[7]*src[5] + tmp[10]*src[7];
    inv[2] -= tmp[3]*src[4] + tmp[6]*src[5] + tmp[11]*src[7];
    inv[3]  = tmp[5]*src[4] + tmp[8]*src[5] + tmp[11]*src[6];
    inv[3] -= tmp[4]*src[4] + tmp[9]*src[5] + tmp[10]*src[6];
    inv[4]  = tmp[1]*src[1] + tmp[2]*src[2] + tmp[ 5]*src[3];
    inv[4] -= tmp[0]*src[1] + tmp[3]*src[2] + tmp[ 4]*src[3];
    inv[5]  = tmp[0]*src[0] + tmp[7]*src[2] + tmp[ 8]*src[3];
    inv[5] -= tmp[1]*src[0] + tmp[6]*src[2] + tmp[ 9]*src[3];
    inv[6]  = tmp[3]*src[0] + tmp[6]*src[1] + tmp[11]*src[3];
    inv[6] -= tmp[2]*src[0] + tmp[7]*src[1] + tmp[10]*src[3];
    inv[7]  = tmp[4]*src[0] + tmp[9]*src[1] + tmp[10]*src[2];
    inv[7] -= tmp[5]*src[0] + tmp[8]*src[1] + tmp[11]*src[2];

    // Compute the pairs for the second 8 elements (cofactors).
    tmp[ 0] = src[2]*src[7];
    tmp[ 1] = src[3]*src[6];
    tmp[ 2] = src[1]*src[7];
    tmp[ 3] = src[3]*src[5];
    tmp[ 4] = src[1]*src[6];
    tmp[ 5] = src[2]*src[5];
    tmp[ 6] = src[0]*src[7];
    tmp[ 7] = src[3]*src[4];
    tmp[ 8] = src[0]*src[6];
    tmp[ 9] = src[2]*src[4];
    tmp[10] = src[0]*src[5];
    tmp[11] = src[1]*src[4];

    // Compute the second 8 elements (cofactors).
    inv[ 8]  = tmp[ 0]*src[13] + tmp[ 3]*src[14] + tmp[ 4]*src[15];
    inv[ 8] -= tmp[ 1]*src[13] + tmp[ 2]*src[14] + tmp[ 5]*src[15];
    inv[ 9]  = tmp[ 1]*src[12] + tmp[ 6]*src[14] + tmp[ 9]*src[15];
    inv[ 9] -= tmp[ 0]*src[12] + tmp[ 7]*src[14] + tmp[ 8]*src[15];
    inv[10]  = tmp[ 2]*src[12] + tmp[ 7]*src[13] + tmp[10]*src[15];
    inv[10] -= tmp[ 3]*src[12] + tmp[ 6]*src[13] + tmp[11]*src[15];
    inv[11]  = tmp[ 5]*src[12] + tmp[ 8]*src[13] + tmp[11]*src[14];
    inv[11] -= tmp[ 4]*src[12] + tmp[ 9]*src[13] + tmp[10]*src[14];
    inv[12]  = tmp[ 2]*src[10] + tmp[ 5]*src[11] + tmp[ 1]*src[ 9];
    inv[12] -= tmp[ 4]*src[11] + tmp[ 0]*src[ 9] + tmp[ 3]*src[10];
    inv[13]  = tmp[ 8]*src[11] + tmp[ 0]*src[ 8] + tmp[ 7]*src[10];
    inv[13] -= tmp[ 6]*src[10] + tmp[ 9]*src[11] + tmp[ 1]*src[ 8];
    inv[14]  = tmp[ 6]*src[ 9] + tmp[11]*src[11] + tmp[ 3]*src[ 8];
    inv[14] -= tmp[10]*src[11] + tmp[ 2]*src[ 8] + tmp[ 7]*src[ 9];
    inv[15]  = tmp[10]*src[10] + tmp[ 4]*src[ 8] + tmp[ 9]*src[ 9];
    inv[15] -= tmp[ 8]*src[ 9] + tmp[11]*src[10] + tmp[ 5]*src[ 8];

    // Compute the determinant.
    double det = src[0]*inv[0] + src[1]*inv[1] + src[2]*inv[2] + src[3]*inv[3];

    // Compute the inverse of the matrix.
    det = 1.0 / det;
    for( int k = 0 ; k < 16 ; ++k )
      inv[k] *= det;

    // Store the results.
    for( int i = 0 ; i < 4 ; ++i )
      for( int j = 0 ; j < 4 ; ++j )
        inv_diag[4*i*num_rows + 4*row + j] = inv[4*i+j];
  }
}

// --------------------------------------------------------------------------------------------------------------------

template< int BLOCK_SIZE >
__global__ void jacobi_smooth_kernel_v0( const int num_rows, 
                                         const double omega, 
                                         const double *inv_diag, 
                                         const double *y, 
                                         const double *b, 
                                               double *x )
{
  int row = blockIdx.x*BLOCK_SIZE + threadIdx.x;

  for( ; row < num_rows ; row += BLOCK_SIZE*gridDim.x )
  {
    double my_inv_diag[16], my_b[4], my_x[4];

    // Load the inverse.
    for( int k = 0 ; k < 4 ; ++k )
    {
      my_inv_diag[4*k + 0] = inv_diag[4*k*num_rows + 4*row + 0];
      my_inv_diag[4*k + 1] = inv_diag[4*k*num_rows + 4*row + 1];
      my_inv_diag[4*k + 2] = inv_diag[4*k*num_rows + 4*row + 2];
      my_inv_diag[4*k + 3] = inv_diag[4*k*num_rows + 4*row + 3];
    }

    // Load y and b.
    for( int k = 0 ; k < 4 ; ++k )
      my_b[k] = b[4*row + k] - y[4*row + k];
    
    // Make sure x is 0.
    for( int k = 0 ; k < 4 ; ++k )
      my_x[k] = 0.0;

    // Compute matrix product.
    for( int k = 0 ; k < 4 ; ++k )
    {
      my_x[k]  = my_inv_diag[4*k + 0]*my_b[0];
      my_x[k] += my_inv_diag[4*k + 1]*my_b[1];
      my_x[k] += my_inv_diag[4*k + 2]*my_b[2];
      my_x[k] += my_inv_diag[4*k + 3]*my_b[3];
    }

    // Update x.
    for( int k = 0 ; k < 4 ; ++k )
      x[4*row + k] += omega*my_x[k];
  }
}

// --------------------------------------------------------------------------------------------------------------------

template< int BLOCK_SIZE >
__global__ void jacobi_smooth_kernel_v1( const int num_rows, 
                                         const double omega, 
                                         const double *inv_diag, 
                                         const double *y, 
                                         const double *b, 
                                               double *x )
{
  const int NUM_ROWS_PER_BLOCK = BLOCK_SIZE/4;

  __shared__ volatile double smem[4][BLOCK_SIZE];

  const int lane_id_mod_4 = laneid()%4;

  int row = blockIdx.x*NUM_ROWS_PER_BLOCK + threadIdx.x/4;

  for( ; row < num_rows ; row += NUM_ROWS_PER_BLOCK*gridDim.x )
  {
    double my_inv_diag[4], my_b, my_x[4];

    // Load the inverse.
    for( int k = 0 ; k < 4 ; ++k )
      my_inv_diag[k] = inv_diag[4*k*num_rows + 4*row + lane_id_mod_4];
    
    // Load y and b.
    my_b = b[4*row + lane_id_mod_4] - y[4*row + lane_id_mod_4];
    
    // Make sure x is 0.
    my_x[0] = my_inv_diag[0]*my_b;
    my_x[1] = my_inv_diag[1]*my_b;
    my_x[2] = my_inv_diag[2]*my_b;
    my_x[3] = my_inv_diag[3]*my_b;

    // Run a reduction to compute x, y, z and w.
    smem[0][threadIdx.x] = my_x[0];
    smem[1][threadIdx.x] = my_x[1];
    smem[2][threadIdx.x] = my_x[2];
    smem[3][threadIdx.x] = my_x[3];

    #pragma unroll
    for( int offset = 2 ; offset > 0 ; offset >>= 1 )
    {
      if( lane_id_mod_4 < offset )
      {
        smem[0][threadIdx.x] = my_x[0] += smem[0][threadIdx.x + offset];
        smem[1][threadIdx.x] = my_x[1] += smem[1][threadIdx.x + offset];
        smem[2][threadIdx.x] = my_x[2] += smem[2][threadIdx.x + offset];
        smem[3][threadIdx.x] = my_x[3] += smem[3][threadIdx.x + offset];
      }
    }

    // Rearrange in SMEM.
    if( lane_id_mod_4 == 0 )
    {
      smem[0][threadIdx.x+0] = my_x[0];
      smem[0][threadIdx.x+1] = my_x[1];
      smem[0][threadIdx.x+2] = my_x[2];
      smem[0][threadIdx.x+3] = my_x[3];
    }
    
    // Update x.
    x[4*row + lane_id_mod_4] += omega*smem[0][threadIdx.x];
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

enum { BLOCK_SIZE = 256 };

// --------------------------------------------------------------------------------------------------------------------

Jacobi::Jacobi(double omega) : m_omega(omega), m_inv_diag(NULL), m_y(NULL), m_r(NULL)
{}

// --------------------------------------------------------------------------------------------------------------------

Jacobi::~Jacobi()
{
  CUDA_SAFE_CALL( cudaFree(m_r) );
  CUDA_SAFE_CALL( cudaFree(m_y) );
  CUDA_SAFE_CALL( cudaFree(m_inv_diag) );
}

// --------------------------------------------------------------------------------------------------------------------

void Jacobi::invert_diag(Context *ctx, const Matrix *A)
{
  int grid_size = 0;
  switch(ctx->jacobi_invert_diag)
  {
  case 0:
    grid_size = std::min( (int) MAX_GRID_SIZE, (A->get_num_rows()+BLOCK_SIZE-1) / BLOCK_SIZE );
    jacobi_invert_diag_kernel_v0<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, ctx->get_stream(0)>>>(
      A->get_num_rows(), 
      A->get_diag(), 
      m_inv_diag);
    CUDA_SAFE_CALL( cudaGetLastError() );
    break;
  case 1:
    grid_size = std::min( (int) MAX_GRID_SIZE, (A->get_num_rows()+BLOCK_SIZE-1) / BLOCK_SIZE );
    jacobi_invert_diag_kernel_v1<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, ctx->get_stream(0)>>>(
      A->get_num_rows(), 
      A->get_diag(), 
      m_inv_diag);
    CUDA_SAFE_CALL( cudaGetLastError() );
    break;
  default:
    std::fprintf(stderr, "Invalid version for jacobi_invert_diag kernel=%d, valid values=[0,1]\n", ctx->jacobi_invert_diag);
    std::exit(1);
  }
}

// --------------------------------------------------------------------------------------------------------------------

void Jacobi::setup(Context *ctx, const Matrix *A)
{
  Solver::setup(ctx, A);

  if( A->get_num_rows() == 0 )
    return;
  if( m_y != NULL )
    CUDA_SAFE_CALL( cudaFree(m_y) );
  CUDA_SAFE_CALL( cudaMalloc((void**) &m_y, 4*A->get_num_rows()*sizeof(double)) );
  CUDA_SAFE_CALL( cudaMalloc((void**) &m_r, 4*A->get_num_rows()*sizeof(double)) );
  if( m_inv_diag != NULL )
    CUDA_SAFE_CALL( cudaFree(m_inv_diag) );
  CUDA_SAFE_CALL( cudaMalloc((void**) &m_inv_diag, 16*A->get_num_rows()*sizeof(double)) );
  invert_diag(ctx, A);
}

// --------------------------------------------------------------------------------------------------------------------

void Jacobi::solve(Context *ctx, const Matrix *A, double *x, const double *b)
{
  const int n = A->get_num_rows();

  // Compute the residual.
  spmv(ctx, A, x, m_y);
  axpby(ctx, n, 1.0, b, -1.0, m_y, m_r);

  // DEBUG.
  converged(ctx, n, m_r);
  print_norm("DEBUG");

  printf("#############################################################################################\n");
  printf("**                             J A C O B I   S O L V E R                                   **\n");
  printf("#############################################################################################\n\n");
  
  for( int iter = 0 ; iter < m_num_max_iters && !converged(ctx, n, m_r) ; ++iter )
  {
    char buffer[64];
    sprintf(buffer, "Iteration %3d: ", iter);
    print_norm(buffer);

    smooth(ctx, A, x, b);
    spmv(ctx, A, x, m_y);
    axpby(ctx, n, 1.0, b, -1.0, m_y, m_r);
  }

  printf("\n#############################################################################################\n\n");
  print_norm("Final error..: ");
  printf("\n#############################################################################################\n\n");
}

// --------------------------------------------------------------------------------------------------------------------

void Jacobi::smooth(Context *ctx, const Matrix *A, double *x, const double *b)
{
  int grid_size = 0;
  spmv(ctx, A, x, m_y);
  switch( ctx->jacobi_smooth )
  {
  case 0:
    grid_size = std::min( (int) MAX_GRID_SIZE, (A->get_num_rows()+BLOCK_SIZE-1) / BLOCK_SIZE );
    jacobi_smooth_kernel_v0<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, ctx->get_stream(0)>>>(
      A->get_num_rows(), 
      m_omega, 
      m_inv_diag, 
      m_y, 
      b, 
      x);
    CUDA_SAFE_CALL( cudaGetLastError() );
    break;
  case 1:
    grid_size = std::min( (int) MAX_GRID_SIZE, (4*A->get_num_rows()+BLOCK_SIZE-1) / BLOCK_SIZE );
    jacobi_smooth_kernel_v1<BLOCK_SIZE><<<grid_size, BLOCK_SIZE, 0, ctx->get_stream(0)>>>(
      A->get_num_rows(), 
      m_omega, 
      m_inv_diag, 
      m_y, 
      b, 
      x);
    CUDA_SAFE_CALL( cudaGetLastError() );
    break;
  default:
    std::fprintf(stderr, "Invalid version for jacobi_smooth_kernel kernel=%d, valid values=[0,1]\n", ctx->jacobi_smooth);
    std::exit(1);
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
