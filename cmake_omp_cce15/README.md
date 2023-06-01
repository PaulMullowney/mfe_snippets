# FindOpenMP fails to discover OpenMP_Fortran_LIB_NAMES for CCE 15.0.1

## C++

Works as expected. Reproduce with the following steps:

```
bareuter@uan02:/.../mfe_snippets/cmake_omp_cce15> source env.sh
bareuter@uan02:/.../mfe_snippets/cmake_omp_cce15> mkdir build.cxx
bareuter@uan02:/.../mfe_snippets/cmake_omp_cce15> cd build.cxx
bareuter@uan02:/.../mfe_snippets/cmake_omp_cce15/build.cxx> CXX=CC cmake ../cxx/
-- The CXX compiler identification is Clang 15.0.6
-- Cray Programming Environment 2.7.20 CXX
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /opt/cray/pe/craype/2.7.20/bin/CC - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found OpenMP_CXX: -fopenmp (found version "5.0")
-- Found OpenMP: TRUE (found version "5.0") found components: CXX
-- Configuring done
-- Generating done
-- Build files have been written to: /.../mfe_snippets/cmake_omp_cce15/build.cxx
bareuter@uan02:/.../mfe_snippets/cmake_omp_cce15/build.cxx> make
[ 50%] Building CXX object CMakeFiles/main.dir/main.cc.o
[100%] Linking CXX executable main
[100%] Built target main
bareuter@uan02:/.../mfe_snippets/cmake_omp_cce15/build.cxx> ./main
SUCCESS
```


## Fortran

Fails to link the binary. Reproduce with the following steps:

```
bareuter@uan02:/.../mfe_snippets/cmake_omp_cce15> source env.sh
bareuter@uan02:/.../mfe_snippets/cmake_omp_cce15> mkdir build.fortran
bareuter@uan02:/.../mfe_snippets/cmake_omp_cce15> cd build.fortran/
bareuter@uan02:/.../mfe_snippets/cmake_omp_cce15/build.fortran> cmake ../fortran/
-- The Fortran compiler identification is Cray 15.0.1
-- Cray Programming Environment 2.7.20 Fortran
-- Detecting Fortran compiler ABI info
-- Detecting Fortran compiler ABI info - done
-- Check for working Fortran compiler: /opt/cray/pe/craype/2.7.20/bin/ftn - skipped
-- Found OpenMP_Fortran: -h omp (found version "4.5")
-- Found OpenMP: TRUE (found version "4.5") found components: Fortran
-- Configuring done
-- Generating done
-- Build files have been written to: /.../mfe_snippets/cmake_omp_cce15/build.fortran
bareuter@uan02:/.../mfe_snippets/cmake_omp_cce15/build.fortran> make
[ 50%] Building Fortran object CMakeFiles/myprog.dir/myprog.F90.o
[100%] Linking Fortran executable myprog
/opt/cray/pe/cce/15.0.1/binutils/x86_64/x86_64-pc-linux-gnu/bin/ld: CMakeFiles/myprog.dir/myprog.F90.o: in function `main':
The Cpu Module:(.text+0x2c): undefined reference to `_cray$mt_execute_parallel'
/opt/cray/pe/cce/15.0.1/binutils/x86_64/x86_64-pc-linux-gnu/bin/ld: CMakeFiles/myprog.dir/myprog.F90.o: in function `my_prog__cray$mt$p0001':
The Cpu Module:(.text+0x74): undefined reference to `omp_get_thread_num_'
/opt/cray/pe/cce/15.0.1/binutils/x86_64/x86_64-pc-linux-gnu/bin/ld: The Cpu Module:(.text+0x88): undefined reference to `_cray$mt_taskwaitall'
make[2]: *** [CMakeFiles/myprog.dir/build.make:96: myprog] Error 1
make[1]: *** [CMakeFiles/Makefile2:83: CMakeFiles/myprog.dir/all] Error 2
make: *** [Makefile:91: all] Error 2
```

## Fixing the Fortran manually by adding C++ and specifying LIB_NAMES

1. Edit `fortran/CMakeLists.txt` and change the `project` line to the following:
```cmake
project( find_libomp_fortran LANGUAGES Fortran CXX )
```

2. Manually specify `OpenMP_Fortran_LIB_NAMES`:
```
bareuter@uan02:/.../mfe_snippets/cmake_omp_cce15> rm -rf build.fortran
bareuter@uan02:/.../mfe_snippets/cmake_omp_cce15> mkdir build.fortran
bareuter@uan02:/.../mfe_snippets/cmake_omp_cce15> cd build.fortran
bareuter@uan02:/.../mfe_snippets/cmake_omp_cce15/build.fortran> grep OpenMP_CXX_LIB_NAMES ../build.cxx/CMakeCache.txt
OpenMP_CXX_LIB_NAMES:STRING=sci_cray_mpi_mp;sci_cray_mp;craymp
//ADVANCED property for variable: OpenMP_CXX_LIB_NAMES
OpenMP_CXX_LIB_NAMES-ADVANCED:INTERNAL=1
bareuter@uan02:/.../mfe_snippets/cmake_omp_cce15/build.fortran> CXX=CC cmake ../fortran/ -DOpenMP_Fortran_LIB_NAMES:STRING="sci_cray_mpi_mp;sci_cray_mp;craymp"
-- The CXX compiler identification is Clang 15.0.6
-- Cray Programming Environment 2.7.20 Fortran
-- Cray Programming Environment 2.7.20 CXX
-- Detecting CXX compiler ABI info
-- Detecting CXX compiler ABI info - done
-- Check for working CXX compiler: /opt/cray/pe/craype/2.7.20/bin/CC - skipped
-- Detecting CXX compile features
-- Detecting CXX compile features - done
-- Found OpenMP_Fortran: -h omp (found version "4.5")
-- Found OpenMP: TRUE (found version "4.5") found components: Fortran
-- Configuring done
-- Generating done
-- Build files have been written to: /.../mfe_snippets/cmake_omp_cce15/build.fortran
bareuter@uan02:/.../mfe_snippets/cmake_omp_cce15/build.fortran> make
[ 50%] Building Fortran object CMakeFiles/myprog.dir/myprog.F90.o
[100%] Linking Fortran executable myprog
[100%] Built target myprog
bareuter@uan02:/.../mfe_snippets/cmake_omp_cce15/build.fortran> ./myprog
SUCCESS
```
