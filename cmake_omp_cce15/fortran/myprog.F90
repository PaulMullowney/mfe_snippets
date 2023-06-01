program my_prog
    use, intrinsic :: iso_fortran_env, only: output_unit
    use omp_lib, only: omp_get_thread_num
    implicit none
    integer :: tid, arr(2)

!$omp parallel num_threads(2) private(tid)
    tid = omp_get_thread_num() + 1
    arr(tid) = tid
!$omp end parallel

    if (arr(1) == 1 .and. arr(2) == 2) then
        write(output_unit, "(a)") "SUCCESS"
    else
        write(output_unit, "(a)") "FAILED"
    end if
end program my_prog
