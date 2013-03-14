#pragma once

#include <cuda_runtime_api.h>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Matrix
{
  // The number of rows, columns and non-zero values.
  int m_num_rows, m_num_cols, m_num_vals;
  // The row offsets.
  int *m_rows;
  // The column indices.
  int *m_cols;
  // The values. The diagonal elements are stored at the end of that array.
  double *m_vals;

public:
  // Create an empty matrix.
  Matrix();

  // Destructor.
  ~Matrix();

  // Allocate memory to store rows.
  void allocate_rows(int num_rows);
  // Allocate memory to store columns.
  void allocate_cols(int num_cols);
  // Allocate memory to store rows.
  void allocate_vals(int num_vals);

  // Get the number of rows.
  inline int get_num_rows() const { return m_num_rows; }
  // Get the number of columns.
  inline int get_num_cols() const { return m_num_cols; }
  // Get the number of non-zero blocks.
  inline int get_num_vals() const { return m_num_vals; }
  // Get the number of elements that are not on the diagonal.
  inline int get_num_off_diag_vals() const { return m_num_vals - m_num_rows; }

  // Get the rows.
  inline const int* get_rows() const { return m_rows; }
  // Get the cols.
  inline const int* get_cols() const { return m_cols; }
  // Get the values.
  inline const double* get_vals() const { return m_vals; }
  // Get the diagonal.
  inline const double* get_diag() const { return m_vals + 16*get_num_off_diag_vals(); }

  // Set the rows from the host.
  void set_rows_from_host(const int *rows_h, cudaStream_t stream);
  // Set the columns from the host.
  void set_cols_from_host(const int *cols_h, cudaStream_t stream);
  // Set the values from the host.
  void set_vals_from_host(const double *vals_h, cudaStream_t stream);

private:
  // Deallocate rows.
  void deallocate_rows();
  // Deallocate columns.
  void deallocate_cols();
  // Deallocate values.
  void deallocate_vals();

  // Set the number of rows.
  inline void set_num_rows(int num_rows) { m_num_rows = num_rows; }
  // Set the number of columns.
  inline void set_num_cols(int num_cols) { m_num_cols = num_cols; }
  // Set the number of non-zero blocks.
  inline void set_num_vals(int num_vals) { m_num_vals = num_vals; }
};


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
