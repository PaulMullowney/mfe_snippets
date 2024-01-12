cmake_minimum_required( VERSION 3.23 FATAL_ERROR )

project( phys_mapping LANGUAGES CXX Fortran)

find_package( OpenMP COMPONENTS Fortran )
find_package( OpenACC COMPONENTS Fortran )
find_package( hip REQUIRED )
enable_testing()
set(CMAKE_C_COMPILER "${ROCM_PATH}/bin/hipcc")
set(CMAKE_CXX_COMPILER "${ROCM_PATH}/bin/hipcc")

add_library(my_hip_lib)
target_sources(my_hip_lib PRIVATE hip.cpp)
target_compile_options(my_hip_lib PRIVATE --offload-arch=gfx90a)
target_link_libraries(my_hip_lib PUBLIC hip::device hip::host)

function( phys_executable )

    set( options "" )
    set( oneValueArgs EXEFILE PHYS_KERNEL PROGRAMMING_MODEL )
    set( multiValueArgs DEFINITIONS LIBS )
    cmake_parse_arguments( _PAR "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    set( _TARGET ${_PAR_EXEFILE}_${_PAR_PHYS_KERNEL}_${_PAR_PROGRAMMING_MODEL} )

    if( "${_PAR_PROGRAMMING_MODEL}" STREQUAL "hip" )
	 	 add_executable( ${_TARGET} ${_PAR_EXEFILE}_${_PAR_PHYS_KERNEL}.F90 )
	 	 set_target_properties( ${_TARGET} PROPERTIES LINKER_LANGUAGE Fortran )
		 target_link_libraries(${_TARGET}  PRIVATE my_hip_lib)
	    target_link_libraries( ${_TARGET}  PUBLIC "-L${ROCM_PATH}/roctracer/lib -lroctracer64" "-L${ROCM_PATH}/roctracer/lib -lroctx64" )
    elseif( "${_PAR_EXEFILE}" STREQUAL "driver" )
	 	 add_executable( ${_TARGET} hip_profiling.F90 ${_PAR_EXEFILE}_${_PAR_PHYS_KERNEL}.F90)
	    target_link_libraries( ${_TARGET}  PUBLIC "-L${ROCM_PATH}/roctracer/lib -lroctracer64" "-L${ROCM_PATH}/roctracer/lib -lroctx64" )
    else()
	 	 add_executable( ${_TARGET} phys_mod.F90 ${_PAR_EXEFILE}.F90)
	 endif()


    target_compile_definitions( ${_TARGET} PRIVATE ${_PAR_PHYS_KERNEL} )

    if( TARGET OpenMP::OpenMP_Fortran )
        target_link_libraries( ${_TARGET} PRIVATE OpenMP::OpenMP_Fortran )
        if( ${CMAKE_Fortran_COMPILER_ID} MATCHES ".*[cC]ray.*" )
            target_link_libraries( ${_TARGET} PRIVATE craymp )
        endif()
    endif()

    if( "${_PAR_PROGRAMMING_MODEL}" STREQUAL "openmp" )
        target_compile_definitions( ${_TARGET} PRIVATE OMP_HOST )
    elseif( "${_PAR_PROGRAMMING_MODEL}" STREQUAL "openmp-offload" )
        target_compile_definitions( ${_TARGET} PRIVATE OMP_DEVICE )
        if( ${CMAKE_Fortran_COMPILER_ID} MATCHES ".*NVHPC.*" )
            target_compile_options( ${_TARGET} PRIVATE -mp=gpu -gpu=cc80,lineinfo )
            target_link_options( ${_TARGET} PRIVATE -mp=gpu -gpu=cc80,lineinfo )
        endif()
    elseif( "${_PAR_PROGRAMMING_MODEL}" STREQUAL "openacc" )
        target_compile_definitions( ${_TARGET} PRIVATE OPENACC )
        target_link_libraries( ${_TARGET} PRIVATE OpenMP::OpenMP_Fortran OpenACC::OpenACC_Fortran )
        if( ${CMAKE_Fortran_COMPILER_ID} MATCHES ".*[cC]ray.*" )
            target_compile_options( ${_TARGET} PRIVATE "-h acc" )
        elseif( ${CMAKE_Fortran_COMPILER_ID} MATCHES ".*NVHPC.*" )
            target_compile_options( ${_TARGET} PRIVATE -gpu=cc80,lineinfo,gvmode )
        endif()
    endif()

    if( _PAR_DEFINITIONS )
        target_compile_definitions( ${_TARGET} PRIVATE ${_PAR_DEFINITIONS} )
    endif()

    if( _PAR_LIBS )
        target_link_libraries( ${_TARGET} PRIVATE ${_PAR_LIBS} )
    endif()

    add_test(
        NAME test_${_TARGET}
        COMMAND ${_TARGET}
    )

endfunction()

if( TARGET OpenMP::OpenMP_Fortran )
    foreach( _kernel LITE_LOOP LITE_LOOP_REVERSED LITE_LOOP_REVERSED_HOIST VERT_SEARCH NASTY_EXPS LU_SOLVER LU_SOLVER_COMPACT )
        phys_executable(
            PHYS_KERNEL ${_kernel}
            PROGRAMMING_MODEL openmp
	         EXEFILE phys_driver
        )
    endforeach()
endif()

if( TARGET OpenMP::OpenMP_Fortran )
    foreach( _kernel LITE_LOOP LITE_LOOP_REVERSED LITE_LOOP_REVERSED_HOIST )
        phys_executable(
            PHYS_KERNEL ${_kernel}
            PROGRAMMING_MODEL openmp-offload
	         EXEFILE phys_driver
        )
    endforeach()
endif()

if( TARGET OpenACC::OpenACC_Fortran )
    foreach( _kernel LITE_LOOP LITE_LOOP_REVERSED LITE_LOOP_REVERSED_HOIST )
        phys_executable(
            PHYS_KERNEL ${_kernel}
            PROGRAMMING_MODEL openacc
	         EXEFILE phys_driver
        )
    endforeach()
endif()

if( TARGET OpenMP::OpenMP_Fortran )
    foreach( _kernel LITE_LOOP LITE_LOOP_REVERSED LITE_LOOP_REVERSED_HOIST )
        phys_executable(
            PHYS_KERNEL ${_kernel}
            PROGRAMMING_MODEL openmp-offload
	         EXEFILE driver
        )
    endforeach()
endif()

foreach( _kernel LITE_LOOP_HIP LITE_LOOP_REVERSED_HIP )
	phys_executable(
      PHYS_KERNEL ${_kernel}
      PROGRAMMING_MODEL hip
      EXEFILE driver
   )
endforeach()
