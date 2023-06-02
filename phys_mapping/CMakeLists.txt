cmake_minimum_required( VERSION 3.25 FATAL_ERROR )

project( phys_mapping LANGUAGES Fortran )

find_package( OpenMP COMPONENTS Fortran )
find_package( OpenACC COMPONENTS Fortran )
enable_testing()

function( phys_executable )

    set( options "" )
    set( oneValueArgs PHYS_KERNEL PROGRAMMING_MODEL )
    set( multiValueArgs DEFINITIONS LIBS )
    cmake_parse_arguments( _PAR "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN} )

    set( _TARGET phys_${_PAR_PHYS_KERNEL}_${_PAR_PROGRAMMING_MODEL} )

    add_executable( ${_TARGET} phys_mod.F90 phys_driver.F90 )

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
        )
    endforeach()
endif()

if( TARGET OpenMP::OpenMP_Fortran )
    foreach( _kernel LITE_LOOP LITE_LOOP_REVERSED LITE_LOOP_REVERSED_HOIST )
        phys_executable(
            PHYS_KERNEL ${_kernel}
            PROGRAMMING_MODEL openmp-offload
        )
    endforeach()
endif()

if( TARGET OpenACC::OpenACC_Fortran )
    foreach( _kernel LITE_LOOP LITE_LOOP_REVERSED LITE_LOOP_REVERSED_HOIST )
        phys_executable(
            PHYS_KERNEL ${_kernel}
            PROGRAMMING_MODEL openacc
        )
    endforeach()
endif()
