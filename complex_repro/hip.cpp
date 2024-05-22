#include <chrono>
#include "hip/hip_runtime.h"
#include <roctracer/roctx.h>

#define HIP_CALL(call)                                   \
	do {                                                  \
	hipError_t err = call;                                \
	if (hipSuccess != err) {                              \
	printf("HIP ERROR (code = %d, %s) at %s:%d\n", err,   \
			 hipGetErrorString(err), __FILE__, __LINE__);   \
	assert(0);                                            \
	exit(1);                                              \
	}                                                     \
} while (0)

__global__ void lite_loop_hip_kernel(const int N,
				     double * __restrict__ in1,
				     const double * __restrict__ in2,
				     const double * __restrict__ in3,
				     const double * __restrict__ in4,
				     const double * __restrict__ in5,
				     const double * __restrict__ in6,
				     const double * __restrict__ in7,
				     const double * __restrict__ in8,
				     const double * __restrict__ in9,
				     const double * __restrict__ in10,
				     double * __restrict__ out1)
{
  int tid = threadIdx.x + blockIdx.x*blockDim.x;
  if (tid<N)
    {
      double t = (in1[tid] + in2[tid] + in3[tid] + in4[tid] + in5[tid] +
		  in6[tid] + in7[tid] + in8[tid] + in9[tid] + in10[tid])*0.1;
      in1[tid] = t;
      out1[tid] = t;
    }
}

template<bool THREAD_BLOCK_2D, int UNROLL>
__global__ void lite_loop_reversed_hip_kernel(const int nproma,
					      const int nlev,
					      double * __restrict__ in1,
					      const double * __restrict__ in2,
					      const double * __restrict__ in3,
					      const double * __restrict__ in4,
					      const double * __restrict__ in5,
					      const double * __restrict__ in6,
					      const double * __restrict__ in7,
					      const double * __restrict__ in8,
					      const double * __restrict__ in9,
					      const double * __restrict__ in10,
					      double * __restrict__ out1)
{
  int tid;
  if (THREAD_BLOCK_2D)
    tid = threadIdx.x + (blockIdx.x*blockDim.y+threadIdx.y)*nlev*nproma;
  else
    tid = threadIdx.x + blockIdx.x*nlev*nproma;
  
#pragma unroll UNROLL
  for (int k=0; k<nlev; ++k)
    {
      double t = (in1[tid] + in2[tid] + in3[tid] + in4[tid] + in5[tid] +
		  in6[tid] + in7[tid] + in8[tid] + in9[tid] + in10[tid])*0.1;
      in1[tid] = t;
      out1[tid] = t;
      tid+=nproma;
    }
}

extern "C"
{
  void phys_kernel_lite_loop_hip ( int* DIM1, int* DIM2, int* DIM3,
				   int* i1, int* i2, int * nt,
				   double * in1,
				   double * in2,
				   double * in3,
				   double * in4,
				   double * in5,
				   double * in6,
				   double * in7,
				   double * in8,
				   double * in9,
				   double * in10,
				   double * out1,
				   double * dt)
  {
    double *din1, *din2, *din3, *din4, *din5, *din6, *din7, *din8, *din9, *din10, *dout1;
    int N = (*DIM1)*(*DIM2)*(*DIM3);
    //printf("DIM1=%d, DIM2=%d, DIM3=%d, in1(1,1,1)=%1.15g, in1(1,1,2)=%1.15g\n",*DIM1,*DIM2,*DIM3,*in1,*(in1+1));
    HIP_CALL(hipMalloc((void**)&din1, N*sizeof(double)));
    HIP_CALL(hipMalloc((void**)&din2, N*sizeof(double)));
    HIP_CALL(hipMalloc((void**)&din3, N*sizeof(double)));
    HIP_CALL(hipMalloc((void**)&din4, N*sizeof(double)));
    HIP_CALL(hipMalloc((void**)&din5, N*sizeof(double)));
    HIP_CALL(hipMalloc((void**)&din6, N*sizeof(double)));
    HIP_CALL(hipMalloc((void**)&din7, N*sizeof(double)));
    HIP_CALL(hipMalloc((void**)&din8, N*sizeof(double)));
    HIP_CALL(hipMalloc((void**)&din9, N*sizeof(double)));
    HIP_CALL(hipMalloc((void**)&din10, N*sizeof(double)));
    HIP_CALL(hipMalloc((void**)&dout1, N*sizeof(double)));
    HIP_CALL(hipMemcpy(din1,in1,N*sizeof(double),hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(din2,in2,N*sizeof(double),hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(din3,in3,N*sizeof(double),hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(din4,in4,N*sizeof(double),hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(din5,in5,N*sizeof(double),hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(din6,in6,N*sizeof(double),hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(din7,in7,N*sizeof(double),hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(din8,in8,N*sizeof(double),hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(din9,in9,N*sizeof(double),hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(din10,in10,N*sizeof(double),hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dout1,out1,N*sizeof(double),hipMemcpyHostToDevice));

    const int nthreads = *nt;
    const int nblocks = (N+nthreads-1)/nthreads;
    //printf("N=%d, nblocks=%d\n",N,nblocks);
    auto start = std::chrono::high_resolution_clock::now();
    roctxRangePush("lite_loop_hip_kernel");
    lite_loop_hip_kernel<<<nblocks, nthreads>>>(N,din1,din2,din3,din4,din5,din6,din7,din8,din9,din10,dout1);
    roctxRangePop();
    roctxMarkA("lite_loop_hip_kernel");
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());

    auto stop = std::chrono::high_resolution_clock::now();
    *dt = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() * 1.e-9;

    HIP_CALL(hipMemcpy(out1,dout1,N*sizeof(double),hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(in1,din1,N*sizeof(double),hipMemcpyHostToDevice));
    
    HIP_CALL(hipFree(din1));
    HIP_CALL(hipFree(din2));
    HIP_CALL(hipFree(din3));
    HIP_CALL(hipFree(din4));
    HIP_CALL(hipFree(din5));
    HIP_CALL(hipFree(din6));
    HIP_CALL(hipFree(din7));
    HIP_CALL(hipFree(din8));
    HIP_CALL(hipFree(din9));
    HIP_CALL(hipFree(din10));
    HIP_CALL(hipFree(dout1));
  }
  void phys_kernel_lite_loop_reversed_hip ( int* DIM1, int* DIM2, int* DIM3,
					    int* i1, int* i2, int * ntx, int * nty,
					    double * in1,
					    double * in2,
					    double * in3,
					    double * in4,
					    double * in5,
					    double * in6,
					    double * in7,
					    double * in8,
					    double * in9,
					    double * in10,
					    double * out1,
					    double * dt) {
    double *din1, *din2, *din3, *din4, *din5, *din6, *din7, *din8, *din9, *din10, *dout1;
    int N = (*DIM1)*(*DIM2)*(*DIM3);
    //printf("DIM1=%d, DIM2=%d, DIM3=%d, in1(1,1,1)=%1.15g, in1(1,1,2)=%1.15g\n",*DIM1,*DIM2,*DIM3,*in1,*(in1+1));
    HIP_CALL(hipMalloc((void**)&din1, N*sizeof(double)));
    HIP_CALL(hipMalloc((void**)&din2, N*sizeof(double)));
    HIP_CALL(hipMalloc((void**)&din3, N*sizeof(double)));
    HIP_CALL(hipMalloc((void**)&din4, N*sizeof(double)));
    HIP_CALL(hipMalloc((void**)&din5, N*sizeof(double)));
    HIP_CALL(hipMalloc((void**)&din6, N*sizeof(double)));
    HIP_CALL(hipMalloc((void**)&din7, N*sizeof(double)));
    HIP_CALL(hipMalloc((void**)&din8, N*sizeof(double)));
    HIP_CALL(hipMalloc((void**)&din9, N*sizeof(double)));
    HIP_CALL(hipMalloc((void**)&din10, N*sizeof(double)));
    HIP_CALL(hipMalloc((void**)&dout1, N*sizeof(double)));
    HIP_CALL(hipMemcpy(din1,in1,N*sizeof(double),hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(din2,in2,N*sizeof(double),hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(din3,in3,N*sizeof(double),hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(din4,in4,N*sizeof(double),hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(din5,in5,N*sizeof(double),hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(din6,in6,N*sizeof(double),hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(din7,in7,N*sizeof(double),hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(din8,in8,N*sizeof(double),hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(din9,in9,N*sizeof(double),hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(din10,in10,N*sizeof(double),hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(dout1,out1,N*sizeof(double),hipMemcpyHostToDevice));

    //printf("N=%d, nblocks=%d\n",N,nblocks);
    auto start = std::chrono::high_resolution_clock::now();
    roctxRangePush("lite_loop_reversed_hip_kernel");

    struct dim3 grid(*DIM3/(*nty), 1, 1);
    struct dim3 block((*ntx), (*nty), 1);
    if ((*nty)==1)
      lite_loop_reversed_hip_kernel<false,2><<<grid, block>>>
	(*DIM1,*DIM2,din1,din2,din3,din4,din5,din6,din7,din8,din9,din10,dout1);
    else
      lite_loop_reversed_hip_kernel<true,2><<<grid, block>>>
	(*DIM1,*DIM2,din1,din2,din3,din4,din5,din6,din7,din8,din9,din10,dout1);

    roctxRangePop();
    roctxMarkA("lite_loop_reversed_hip_kernel");
    HIP_CALL(hipGetLastError());
    HIP_CALL(hipDeviceSynchronize());

    auto stop = std::chrono::high_resolution_clock::now();
    *dt = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start).count() * 1.e-9;

    HIP_CALL(hipMemcpy(out1,dout1,N*sizeof(double),hipMemcpyHostToDevice));
    HIP_CALL(hipMemcpy(in1,din1,N*sizeof(double),hipMemcpyHostToDevice));

    HIP_CALL(hipFree(din1));
    HIP_CALL(hipFree(din2));
    HIP_CALL(hipFree(din3));
    HIP_CALL(hipFree(din4));
    HIP_CALL(hipFree(din5));
    HIP_CALL(hipFree(din6));
    HIP_CALL(hipFree(din7));
    HIP_CALL(hipFree(din8));
    HIP_CALL(hipFree(din9));
    HIP_CALL(hipFree(din10));
    HIP_CALL(hipFree(dout1));
  }
}
