#include "utils.h"
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Context::Context() : jacobi_invert_diag(0), jacobi_smooth(0)
{
  CUDA_SAFE_CALL( cudaStreamCreate(&m_stream) );
}

// --------------------------------------------------------------------------------------------------------------------

Context::~Context()
{
  CUDA_SAFE_CALL( cudaStreamDestroy(m_stream) );
}

// --------------------------------------------------------------------------------------------------------------------

void Context::read_from_file(const char *filename)
{
  std::ifstream file(filename, std::ios::in);
  if( !file )
  {
    std::fprintf(stderr, "ERROR: Cannot open the ctx file %s\n", filename);
    std::exit(1);
  }

  for( ; !file.eof() ; )
  {
    std::string line;
    std::getline(file, line);

    if(line[0] == '%') // skip comments.
      continue;
    
    std::string::const_iterator it = std::find(line.begin(), line.end(), '=');
    if( it == line.end() )
    {
      std::fprintf(stderr, "WARNING: ctx line %s does not match the key=value pattern\n", line.c_str());
      continue;
    }
    const size_t eq_pos = it - line.begin();
    std::string key = line.substr(0, eq_pos);
    std::string val = line.substr(eq_pos+1);

    if( key == "dot" )
      this->dot = std::atoi(val.c_str());
    else if( key == "spmv" )
      this->spmv = std::atoi(val.c_str());
    else if( key == "jacobi_invert_diag" )
      jacobi_invert_diag = std::atoi(val.c_str());
    else if( key == "jacobi_smooth" )
      jacobi_smooth = std::atoi(val.c_str());
  }
  
  file.close();
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
