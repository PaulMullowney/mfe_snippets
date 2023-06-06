# Mapping LITE_LOOP to GPU execution units

In our physics, the control flow after GPU adaptation is, very simplified, as follows:

```f90
!$acc parallel loop gang vector_length(block_size)
! -or- !$omp target teams distribute num_threads(block_size)
do iblock=1,nblock
  call vector_routine(1, block_size, ...)
end do
```

with the vector routines exposing another level of parallelism as

```f90
subroutine vector_routine(istart, iend, ...)
!$acc routine vector
! -or- !$omp target declare
!$acc loop vector
! -or- !$omp parallel do simd
do i=istart, iend
  do k=1,nlev
  ...
  end do
end do
end subroutine
```

with a fairly substantial amount of compute inside the vector loop but, notably, no data dependencies between loop iterations. Performance is limited by register file pressure due to temporaries in the kernel. More complex variants of this control flow exist in the real code and consist of multiple calls to vector routines from the same block loop, and/or nested calls inside the vector routine, which occasionally split the vector loop.

In our proxy mini-app CLOUDSC (https://github.com/ecmwf-ifs/dwarf-p-cloudsc), we have this control flow using OpenACC offloading, and a variation of the above where we hoist the loop from the kernel to the caller (converting the routine to a sequential routine). This last variant shows the best performance on NVIDIA GPUs. For OpenMP offloading we adapted this last variant and achieved fairly comparable performance on NVIDIA.

On LUMI, we observed extremely poor performance with the OpenMP offloading variant and started investigating the mapping to control flow units with a very small example, the LITE_LOOP in this repository. It seems the "hoisted" variant does not map very well to AMD GPUs in the first place (both for OpenACC and OpenMP), and the OpenMP performance is reversed between NVIDIA and AMD GPUs.

With the code in this repository, it is possible to investigate the mapping of a simplified physics control flow to GPU execution units. Notably, this does not have the hefty use of temporaries which causes performance bottlenecks in the real code. There are three different variants:

* LITE_LOOP: The original CPU control flow (with the vector dimension innermost)
* LITE_LOOP_REVERSED: The GPU-tailored control flow with the two inner loops reversed (pulling out parallelism)
* LITE_LOOP_REVERSED_HOIST: The GPU-tailored control flow where the vector loop is moved to the caller side

## Test protocol

```
$.../phys_mapping> mkdir build && cd build
$.../phys_mapping/build> cmake .. && make
$.../phys_mapping/build> ctest -R "open(acc|mp-offload)" --verbose
```

## AMD MI250X

```
$.../phys_mapping/build> module list

Currently Loaded Modules:
  1) LUMI/23.03              (S)   4) libfabric/1.15.2.0                          7) partition/G      (S)  10) cray-mpich/8.1.25      13) cce/15.0.1              16) cpeCray/23.03
  2) craype-x86-trento       (H)   5) craype-network-ofi                    (H)   8) craype/2.7.20         11) cray-libsci/23.02.1.1  14) rocm/5.2.3              17) buildtools/23.03
  3) craype-accel-amd-gfx90a (H)   6) xpmem/2.5.2-2.4_3.20__gd0f7936.shasta       9) cray-dsmml/0.2.2      12) PrgEnv-cray/8.3.3      15) perftools-base/23.03.0

  Where:
   S:  Module is Sticky, requires --force to unload or purge
   H:             Hidden Module
$.../phys_mapping/build> export CRAY_ACC_DEBUG=2
```

<table>
    <thead>
        <tr>
            <td rowspan="2"></td>
            <td colspan="4"><b>OpenACC</b></td>
            <td colspan="4"><b>OpenMP</b></td>
        </tr>
        <tr>
            <td colspan="4">Launch config ([blocks] x [threads])<br />Kernel runtime (8 invocations)</td>
            <td colspan="4">Launch config ([blocks] x [threads])<br />Kernel runtime (8 invocations)</td>
        </tr>
        <tr>
            <td><i>NPROMA</i></td>
            <td><i>32</i></td>
            <td><i>64</i></td>
            <td><i>128</i></td>
            <td><i>256</i></td>
            <td><i>32</i></td>
            <td><i>64</i></td>
            <td><i>128</i></td>
            <td><i>256</i></td>
        </tr>
    </thead>
    <tbody>
        <tr></tr>
        <tr>
            <td rowspan="2">LITE_LOOP</td>
            <td>32768x32</td>
            <td>16384x64</td>
            <td>8192x128</td>
            <td>4096x256</td>
            <td>32768x32</td>
            <td>16384x64</td>
            <td>8192x128</td>
            <td>4096x256</td>
        </tr>
        <tr>
            <td>0.259</td>
            <td>0.160</td>
            <td>0.118</td>
            <td>0.112</td>
            <td>0.264</td>
            <td>0.145</td>
            <td>0.144</td>
            <td>0.113</td>
        </tr>
        <tr>
            <td rowspan="2">LITE_LOOP_REVERSED</td>
            <td>32768x32</td>
            <td>16384x64</td>
            <td>8192x128</td>
            <td>4096x256</td>
            <td>32768x32</td>
            <td>16384x64</td>
            <td>8192x128</td>
            <td>4096x256</td>
        </tr>
        <tr>
            <td>0.259</td>
            <td>0.147</td>
            <td>0.118</td>
            <td>0.112</td>
            <td>0.262</td>
            <td>0.147</td>
            <td>0.116</td>
            <td>0.112</td>
        </tr>
        <tr>
            <td rowspan="2">LITE_LOOP_REVERSED_HOIST</td>
            <td>32768x32</td>
            <td>16384x64</td>
            <td>8192x128</td>
            <td>4096x256</td>
            <td>32768x32</td>
            <td><b>220x64</b></td>
            <td>8192x128</td>
            <td>4096x256</td>
        </tr>
        <tr>
            <td>0.260</td>
            <td>0.145</td>
            <td>0.118</td>
            <td>0.111</td>
            <td>1.963</td>
            <td>3.5275</td>
            <td>1.853</td>
            <td>1.870</td>
        </tr>
    </tbody>
</table>

## NVIDIA A100

```
$.../phys_mapping/build> nvfortran --version
nvfortran 22.11-0 64-bit target on x86-64 Linux -tp zen2
NVIDIA Compilers and Tools
Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
$.../phys_mapping/build> export NVCOMPILER_ACC_NOTIFY=1
```


<table>
    <thead>
        <tr>
            <td rowspan="2"></td>
            <td colspan="4"><b>OpenACC</b></td>
            <td colspan="4"><b>OpenMP</b></td>
        </tr>
        <tr>
            <td colspan="4">Launch config ([grid] x [block)<br />Kernel runtime (8 invocations)</td>
            <td colspan="4">Launch config ([grid] x [block])<br />Kernel runtime (8 invocations)</td>
        </tr>
        <tr>
            <td><i>NPROMA</i></td>
            <td><i>32</i></td>
            <td><i>64</i></td>
            <td><i>128</i></td>
            <td><i>256</i></td>
            <td><i>32</i></td>
            <td><i>64</i></td>
            <td><i>128</i></td>
            <td><i>256</i></td>
        </tr>
    </thead>
    <tbody>
        <tr></tr>
        <tr>
            <td rowspan="2">LITE_LOOP</td>
            <td>32768x32</td>
            <td>16384x64</td>
            <td>8192x128</td>
            <td>4096x256</td>
            <td>108x32</td>
            <td>108x64</td>
            <td>108x128</td>
            <td>108x128</td>
        </tr>
        <tr>
            <td>0.089</td>
            <td>0.090</td>
            <td>0.091</td>
            <td>0.090</td>
            <td>3.223</td>
            <td>3.221</td>
            <td>3.218</td>
            <td>3.215</td>
        </tr>
        <tr>
            <td rowspan="2">LITE_LOOP_REVERSED</td>
            <td>32768x32</td>
            <td>16384x64</td>
            <td>8192x128</td>
            <td>4096x256</td>
            <td>108x32</td>
            <td>108x64</td>
            <td>108x128</td>
            <td>108x128</td>
        </tr>
        <tr>
            <td>0.089</td>
            <td>0.090</td>
            <td>0.090</td>
            <td>0.090</td>
            <td>7.663</td>
            <td>7.725</td>
            <td>8.360</td>
            <td>8.149</td>
        </tr>
        <tr>
            <td rowspan="2">LITE_LOOP_REVERSED_HOIST</td>
            <td>32768x32</td>
            <td>16384x64</td>
            <td>8192x128</td>
            <td>4096x256</td>
            <td>108x64</td>
            <td>108x96</td>
            <td>108x160</td>
            <td>108x288</td>
        </tr>
        <tr>
            <td>0.083</td>
            <td>0.083</td>
            <td>0.084</td>
            <td>0.084</td>
            <td>0.387</td>
            <td>0.210</td>
            <td>0.126</td>
            <td>0.093</td>
        </tr>
    </tbody>
</table>

## Observations

* Hoisting works best on NVIDIA but worst on AMD
* Launch configuration for hoisting with OpenMP on AMD is strange with NPROMA=64
* Launch configuration with OpenMP on NVIDIA is strange
