cmake_minimum_required( VERSION 3.25 FATAL_ERROR )

project( omp_acc_fortran LANGUAGES Fortran )

find_package( OpenMP REQUIRED COMPONENTS Fortran )
find_package( OpenACC REQUIRED COMPONENTS Fortran )

add_executable( myprog myprog.F90 )
target_link_libraries( myprog PRIVATE OpenMP::OpenMP_Fortran OpenACC::OpenACC_Fortran )
if( ${CMAKE_Fortran_COMPILER_ID} MATCHES ".*[cC]ray.*" )
    target_link_libraries( myprog PRIVATE craymp )
endif()

if( ENABLE_WORKAROUND )
    target_compile_options( myprog PRIVATE "-h acc" )
endif()
