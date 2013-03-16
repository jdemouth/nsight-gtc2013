#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime_api.h>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#define CUDA_SAFE_CALL(call) { \
  cudaError_t status = call; \
  if( status != cudaSuccess ) { \
    std::fprintf(stderr, "CUDA ERROR %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
    std::exit((int) status); \
  } \
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

class Context
{
public:
  int dot;
  int spmv;
  int jacobi_invert_diag;
  int jacobi_smooth;

private:
  // Cuda stream.
  cudaStream_t m_stream;
  
public:
  // Set all version to 0.
  Context();
  // Destructor.
  ~Context();

  // Get a CUDA stream.
  inline cudaStream_t get_stream(int) { return m_stream; }
  
  // Read the ctx from a file.
  void read_from_file(const char *filename);

private:
  // No copy constructor.
  Context(const Context &);
  // No assignment operator.
  Context& operator=(const Context &);
};

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __CUDACC__

enum { WARP_SIZE = 32, LOG_WARP_SIZE = 5, MAX_GRID_SIZE = 4096 };

#if (defined(_MSC_VER) && defined(_WIN64)) || defined(__LP64__)
#define __PTR   "l"
#else
#define __PTR   "r"
#endif

__device__ __inline__ int warpid() 
{ 
  return threadIdx.x >> LOG_WARP_SIZE; 
}

__device__ __inline__ int laneid() 
{ 
  int lane_id;
  asm( "mov.u32 %0, %%laneid;" : "=r"(lane_id) );
  return lane_id; 
}

__device__ __inline__ int bfe( int mask, int first_bit, int num_bits ) 
{ 
  int res;
  asm( "bfe.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(mask), "r"(first_bit), "r"(num_bits) );
  return res; 
}

__device__ __inline__ double shfl_xor( double r, int mask )
{
#if __CUDA_ARCH__ >= 300
  int lo = __double2loint(r);
  lo = __shfl_xor( lo, mask );
  int hi = __double2hiint(r);
  hi = __shfl_xor( hi, mask );
  return __hiloint2double( hi, lo );
#else
  return 0.0;
#endif
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 350
__device__ __inline__ double __ldg(const double *addr)
{
  return *addr;
}
#endif

__device__ __inline__ void load_cg( double &r, const double *__restrict addr )
{
  asm( "ld.global.cg.f64 %0, [%1];" : "=d"(r) : __PTR(addr) );
}

__device__ __inline__ void load_cg_v4( double &r0, double &r1, double &r2, double &r3, const double *__restrict addr )
{
  asm( "ld.global.cg.f64.v2 {%0, %1}, [%2];" : "=d"(r0), "=d"(r1) : __PTR(addr+0) );
  asm( "ld.global.cg.f64.v2 {%0, %1}, [%2];" : "=d"(r2), "=d"(r3) : __PTR(addr+2) );
}

__device__ __inline__ void load_nc_v4( double (&r)[4], const double *__restrict addr )
{
  asm( "ld.global.nc.f64.v2 {%0, %1}, [%2];" : "=d"(r[0]), "=d"(r[1]) : __PTR(addr+0) );
  asm( "ld.global.nc.f64.v2 {%0, %1}, [%2];" : "=d"(r[2]), "=d"(r[3]) : __PTR(addr+2) );
}

#endif // defined __CUDACC__

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
