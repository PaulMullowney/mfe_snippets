cmake_minimum_required( VERSION 3.25 FATAL_ERROR )

project( find_libomp_fortran LANGUAGES Fortran )

find_package( OpenMP REQUIRED COMPONENTS Fortran )

add_executable( myprog myprog.F90 )
target_link_libraries( myprog OpenMP::OpenMP_Fortran )
