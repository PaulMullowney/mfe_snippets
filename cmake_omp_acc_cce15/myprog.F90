program my_prog
    use, intrinsic :: iso_fortran_env, only: output_unit
    use omp_lib, only: omp_get_thread_num
    implicit none
    integer :: tid, arr(4)

!$omp parallel num_threads(4) private(tid)
    tid = omp_get_thread_num() + 1
    arr(tid) = tid
!$omp end parallel

!$acc data copy(arr)

!$acc parallel loop
    do tid=1,4
        arr(tid) = arr(tid) + 1
    end do
!$acc end parallel loop

!$acc end data

    if (arr(1) == 2 .and. arr(2) == 3 .and. arr(3) == 4 .and. arr(4) == 5) then
        write(output_unit, "(a)") "SUCCESS"
    else
        write(output_unit, "(a)") "FAILED"
    end if
end program my_prog
