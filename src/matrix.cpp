#include "matrix.h"
#include "utils.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Matrix::Matrix() : m_num_rows(0), m_num_cols(0), m_num_vals(0), m_rows(NULL), m_cols(NULL), m_vals(NULL)
{}

// --------------------------------------------------------------------------------------------------------------------

Matrix::~Matrix()
{
  deallocate_rows();
  deallocate_cols();
  deallocate_vals();
}

// --------------------------------------------------------------------------------------------------------------------

void Matrix::allocate_rows(int num_rows)
{
  deallocate_rows();
  set_num_rows(num_rows);
  if( num_rows == 0 )
    return;
  CUDA_SAFE_CALL( cudaMalloc((void**) &m_rows, (num_rows+1)*sizeof(int)) );
}

// --------------------------------------------------------------------------------------------------------------------

void Matrix::allocate_cols(int num_cols)
{
  deallocate_cols();
  set_num_cols(num_cols);
  if( num_cols == 0 )
    return;
  CUDA_SAFE_CALL( cudaMalloc((void**) &m_cols, num_cols*sizeof(int)) );
}

// --------------------------------------------------------------------------------------------------------------------

void Matrix::allocate_vals(int num_vals)
{
  deallocate_vals();
  set_num_vals(num_vals);
  if( num_vals == 0 )
    return;
  CUDA_SAFE_CALL( cudaMalloc((void**) &m_vals, 16*num_vals*sizeof(double)) );
}

// --------------------------------------------------------------------------------------------------------------------

void Matrix::deallocate_rows()
{
  CUDA_SAFE_CALL( cudaFree(m_rows) );
  m_rows = NULL;
  set_num_rows(0);
}

// --------------------------------------------------------------------------------------------------------------------

void Matrix::deallocate_cols()
{
  CUDA_SAFE_CALL( cudaFree(m_cols) );
  m_cols = NULL;
  set_num_cols(0);
}

// --------------------------------------------------------------------------------------------------------------------

void Matrix::deallocate_vals()
{
  CUDA_SAFE_CALL( cudaFree(m_vals) );
  m_vals = NULL;
  set_num_vals(0);
}

// --------------------------------------------------------------------------------------------------------------------

void Matrix::set_rows_from_host(const int *rows_h, cudaStream_t stream)
{
  if(m_num_rows == 0)
    return;
  CUDA_SAFE_CALL( cudaMemcpyAsync(m_rows, rows_h, (m_num_rows+1)*sizeof(int), cudaMemcpyHostToDevice, stream) );
}

// --------------------------------------------------------------------------------------------------------------------

void Matrix::set_cols_from_host(const int *cols_h, cudaStream_t stream)
{
  if(m_num_cols == 0)
    return;
  CUDA_SAFE_CALL( cudaMemcpyAsync(m_cols, cols_h, m_num_cols*sizeof(int), cudaMemcpyHostToDevice, stream) );
}

// --------------------------------------------------------------------------------------------------------------------

void Matrix::set_vals_from_host(const double *vals_h, cudaStream_t stream)
{
  if(m_num_vals == 0)
    return;
  CUDA_SAFE_CALL( cudaMemcpyAsync(m_vals, vals_h, 16*m_num_vals*sizeof(double), cudaMemcpyHostToDevice, stream) );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
