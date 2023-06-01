# CMake generates wrong compile options if both OpenACC and OpenMP are used

To reproduce:

```
$:/...mfe_snippets/cmake_omp_acc_cce15> source env.sh
$:/...mfe_snippets/cmake_omp_acc_cce15> cmake --version
cmake version 3.25.2

CMake suite maintained and supported by Kitware (kitware.com/cmake).
$:/...mfe_snippets/cmake_omp_acc_cce15> mkdir build && cd build
$:/...mfe_snippets/cmake_omp_acc_cce15/build> cmake ..
$:/...mfe_snippets/cmake_omp_acc_cce15/build> make VERBOSE=1
...
/opt/cray/pe/craype/2.7.20/bin/ftn   -em -J. -h omp acc -c /.../mfe_snippets/cmake_omp_acc_cce15/myprog.F90 -o CMakeFiles/myprog.dir/myprog.F90.o


ftn-1981 ftn: WARNING MY_PROG, File = .../mfe_snippets/cmake_omp_acc_cce15/myprog.F90, Line = 12, Column = 6
  This compilation contains OpenACC directives.  -h acc is not active so the directives are ignored.

Cray Fortran : Version 15.0.1 (20230120205242_66f7391d6a03cf932f321b9f6b1d8612ef5f362c)
Cray Fortran : Thu Jun 01, 2023  16:06:41
Cray Fortran : Compile time:  0.0220 seconds
Cray Fortran : 27 source lines
Cray Fortran : 0 errors, 1 warnings, 0 other messages, 0 ansi
Cray Fortran : "explain ftn-message number" gives more information about each message.
...
```

The issue is, that CMake builds the compiler options as `-h omp acc` instead of repeating the `-h` prefix: `-h omp -h acc`.

A workaround is to manually supply `-h acc` as a compile option again, see for example:

```
$:/...mfe_snippets/cmake_omp_acc_cce15/build> cmake .. -DENABLE_WORKAROUND=1
$:/...mfe_snippets/cmake_omp_acc_cce15/build> make VERBOSE=1
...
/opt/cray/pe/craype/2.7.20/bin/ftn   -em -J. "-h acc" -h omp acc -c .../mfe_snippets/cmake_omp_acc_cce15/myprog.F90 -o CMakeFiles/myprog.dir/myprog.F90.o
...
```
