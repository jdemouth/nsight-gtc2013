#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <cstring>
#include <algorithm>
#include "utils.h"
#include "matrix.h"
#include "jacobi.h"
#include "bicgstab.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void read_system_from_file(Context *ctx, const char *filename, Matrix *A, double **x, double **b)
{
  // Open the files.
  FILE *file = fopen(filename, "rb");
  if( !file )
  {
    fprintf(stderr, "ERROR: Cannot open file %s\n", filename);
    exit(1);
  }

  // Print info.
  printf("** SYSTEM      : %13s                                                             **\n", filename);
  printf("\n#############################################################################################\n\n");

  // Numbers of rows, vals and temp value.
  int num_rows, num_cols, num_vals;

  // Read the sparse linear system size
  fread(&num_rows, sizeof(int), 1, file);
  fread(&num_cols, sizeof(int), 1, file);
  fread(&num_vals, sizeof(int), 1, file);
  
  // Define the number of rows/vals.
  A->allocate_rows(num_rows);
  A->allocate_cols(num_cols);
  A->allocate_vals(num_vals);

  // Read the rows.
  int *rows_h = new int[num_rows+1];
  size_t num_items = fread(rows_h, sizeof(int), num_rows+1, file);
  if( num_items != num_rows+1 ) 
  {
    fprintf(stderr, "ERROR: Invalid number of rows\n");
    exit(1);
  }
  A->set_rows_from_host(rows_h, ctx->get_stream(0));

  // Read the columns.
  int *cols_h = new int[num_cols];
  num_items = fread(cols_h, sizeof(int), num_cols, file);
  if( num_items != num_cols ) 
  {
    fprintf(stderr, "ERROR: Invalid number of columns\n");
    exit(1);
  }
  A->set_cols_from_host(cols_h, ctx->get_stream(0));

  // Buffer to read the values.
  float *vals_fp32_h = new float[16*num_vals];
  num_items = fread(vals_fp32_h, sizeof(float), 16*num_vals, file);
  if( num_items != 16*num_vals )
  {
    fprintf(stderr, "ERROR: Invalid number of values\n");
    exit(1);
  }

  double *vals_h = new double[16*num_vals];
  for( int i = 0 ; i < 16*num_vals ; ++i )
    vals_h[i] = static_cast<double>(vals_fp32_h[i]);
  delete[] vals_fp32_h;
  A->set_vals_from_host(vals_h, ctx->get_stream(0));

  // Read the RHS.
  float *b_fp32_h = new float[4*num_rows];
  num_items = fread(b_fp32_h, sizeof(float), 4*num_rows, file);
  if( num_items != 4*num_rows )
  {
    fprintf(stderr, "ERROR: Invalid RHS\n");
    exit(1);
  }

  double *b_h = new double[4*num_rows];
  for( int i = 0 ; i < 4*num_rows ; ++i )
    b_h[i] = static_cast<double>(b_fp32_h[i]);
  delete[] b_fp32_h;

  // Allocate memory to store B and X (implicit synchronization).
  CUDA_SAFE_CALL( cudaMalloc((void**) b, 4*num_rows*sizeof(double)) );
  CUDA_SAFE_CALL( cudaMalloc((void**) x, 4*num_rows*sizeof(double)) );

  // Copy B and set X to 0.0.
  CUDA_SAFE_CALL( cudaMemcpy(*b, b_h, 4*num_rows*sizeof(double), cudaMemcpyHostToDevice) );
  CUDA_SAFE_CALL( cudaMemset(*x, 0, 4*num_rows*sizeof(double)) );

  // Release host memory.
  delete[] b_h;
  delete[] vals_h;
  delete[] cols_h;
  delete[] rows_h;

  // Close the files.
  fclose(file);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static void do_main()
{
  Matrix A;
  double *x, *b;

  printf("#############################################################################################\n");
  printf("**                           B I C G S T A B   S O L V E R                                 **\n");
  printf("#############################################################################################\n\n");

  cudaDeviceProp props;
  CUDA_SAFE_CALL( cudaGetDeviceProperties(&props, 0) );
  printf("** DEVICE      : %10s (ECC: %3s)                                                     **\n", props.name, props.ECCEnabled ? "ON" : "OFF");
  printf("\n#############################################################################################\n\n");

  Context ctx;
  ctx.read_from_file("config.txt");
  read_system_from_file(&ctx, "res/matrix.inp", &A, &x, &b);

  cudaEvent_t start, stop;
  CUDA_SAFE_CALL( cudaEventCreate(&start) );
  CUDA_SAFE_CALL( cudaEventCreate(&stop) );
  
  CUDA_SAFE_CALL( cudaEventRecord(start) );

  //Jacobi solver(0.6);
  Bicgstab solver;
  solver.setup(&ctx, &A);
  solver.solve(&ctx, &A, x, b);

  float elapsed_time = 0.0f;
  CUDA_SAFE_CALL( cudaEventRecord(stop) );
  CUDA_SAFE_CALL( cudaEventSynchronize(stop) );
  CUDA_SAFE_CALL( cudaEventElapsedTime(&elapsed_time, start, stop) );
  
  printf("** ELAPSED TIME: %9.3fms                                                               **\n", elapsed_time);
  printf("\n#############################################################################################\n\n");

  CUDA_SAFE_CALL( cudaEventDestroy(stop) );
  CUDA_SAFE_CALL( cudaEventDestroy(start) );

  CUDA_SAFE_CALL( cudaFree(x) );
  CUDA_SAFE_CALL( cudaFree(b) );
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

int main( int, char ** )
{
  do_main();

  CUDA_SAFE_CALL( cudaDeviceReset() );
  return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
