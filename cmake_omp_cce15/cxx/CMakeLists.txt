cmake_minimum_required( VERSION 3.25 FATAL_ERROR )

project( find_libomp_cxx LANGUAGES CXX )

find_package( OpenMP REQUIRED COMPONENTS CXX )

add_executable( main main.cc )
target_link_libraries( main OpenMP::OpenMP_CXX )
