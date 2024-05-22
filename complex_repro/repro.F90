#define __NPROMA__ 64

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! RocTx profiling
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

module hip_profiling

  interface
     subroutine roctxMarkA(message) bind(c, name="roctxMarkA")
       use iso_c_binding,   only: c_char
       implicit none
       character(c_char) :: message(*)
     end subroutine roctxMarkA

     function roctxRangePushA(message) bind(c, name="roctxRangePushA")
       use iso_c_binding,   only: c_int,&
            c_char
       implicit none
       integer(c_int) :: roctxRangePushA
       character(c_char) :: message(*)
     end function roctxRangePushA

     subroutine roctxRangePop() bind(c, name="roctxRangePop")
       implicit none
     end subroutine roctxRangePop

  end interface

end module hip_profiling

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Compute Module
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

module mod

use :: iso_fortran_env

implicit none

integer, parameter :: sp = REAL32
integer, parameter :: dp = REAL64
#ifdef FLOAT_SINGLE
integer, parameter :: lp = sp     !! lp : "local" precision
#else
integer, parameter :: lp = dp     !! lp : "local" precision
#endif

integer, parameter :: ip = INT32

real(kind=lp) :: cst1 = 2.5, cst2 = 3.14
integer, parameter :: nspecies = 5


contains

subroutine kernel(dim1,dim2,dim3,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,in11,out1,out2,i)
  integer(kind=ip),intent(in) :: dim1, dim2, dim3, i
  real(kind=lp),dimension(1:dim1,1:dim2),intent(in) :: in2,in3,in4,in5,in6,in7,in8,in9,in10
  real(kind=lp),dimension(1:dim1,1:dim2),intent(inout) :: in1, out1
  real(kind=lp),dimension(1:dim1,1:dim2,1:dim3),intent(in) :: in11
  real(kind=lp),dimension(1:dim1,1:dim2,1:dim3),intent(inout) :: out2
  integer(kind=ip) :: j, k
  !$omp declare target
  do j=1,dim2
    out1(i,j) = (in1(i,j) + in2(i,j) + in3(i,j) + in4(i,j) + in5(i,j) + &
 &               in6(i,j) + in7(i,j) + in8(i,j) + in9(i,j) + in10(i,j)) * 0.1
    in1(i,j) = out1(i,j)
  end do

  do k=1,dim3
     do j=1,dim2
        in12(i,j,k) = in11(i,j,k) + 1.0*(j+k)*out1(i,j)
     end do
  end do
  
end subroutine kernel

end module mod


!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Main subroutine
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

program phys_layout

use omp_lib

use mod, only: ip, lp, kernel

!omp declare target(kernel)

USE iso_c_binding   ,ONLY : c_null_char
use hip_profiling, only : roctxRangePushA, roctxRangePop, roctxMarkA

implicit none

!! arrays for passing to subroutine
real(kind=lp),dimension(:,:,:),allocatable :: arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8, arr9, arr10
real(kind=lp),dimension(:,:,:),allocatable :: out1
real(kind=lp),dimension(:,:,:,:),allocatable :: arr11, out2

integer(kind=ip) :: nproma, npoints, nlev, nclv, nblocks, ntotal, num_main_loops
integer(kind=ip) :: nb, nml, inp, i, k, real_bytes
real(kind=lp) :: est, dt, dttotal

integer :: i_seed, ret
integer, dimension(:), allocatable :: a_seed

real(kind=lp) :: time1, time2

! Constants
nlev    = 137
nclv    = 5
num_main_loops = 8

! Defaults
nproma = __NPROMA__
ntotal = nproma*16384

npoints = nproma  ! Can be unlocked later...
nblocks = ntotal / nproma
write(*, *)
write(*, '(A10,I8,A10,I8,A10,I8,A14,I8,A)') 'NPROMA ', nproma, ' TOTAL ', ntotal, ' NBLOCKS ', nblocks

#ifdef FLOAT32
real_bytes = 4
#else
real_bytes = 8
#endif
dttotal = 0
allocate(arr1 (npoints,nlev,nblocks))
allocate(arr2 (npoints,nlev,nblocks))
allocate(arr3 (npoints,nlev,nblocks))
allocate(arr4 (npoints,nlev,nblocks))
allocate(arr5 (npoints,nlev,nblocks))
allocate(arr6 (npoints,nlev,nblocks))
allocate(arr7 (npoints,nlev,nblocks))
allocate(arr8 (npoints,nlev,nblocks))
allocate(arr9 (npoints,nlev,nblocks))
allocate(arr10(npoints,nlev,nblocks))
allocate(out1 (npoints,nlev,nblocks))
allocate(arr11(npoints,nlev,nclv,nblocks))
allocate(out2 (npoints,nlev,nclv,nblocks))

! ----- Set up random seed portably -----
call random_seed(size=i_seed)
allocate(a_seed(1:i_seed))
call random_seed(get=a_seed)
a_seed = [ (i,i=1,i_seed) ]
call random_seed(put=a_seed)
deallocate(a_seed)

do inp = 1, ntotal, nproma
  nb = (inp-1) / nproma + 1
  call random_number(arr1(:,:,nb))
  call random_number(arr2(:,:,nb))
  call random_number(arr3(:,:,nb))
  call random_number(arr4(:,:,nb))
  call random_number(arr5(:,:,nb))
  call random_number(arr6(:,:,nb))
  call random_number(arr7(:,:,nb))
  call random_number(arr8(:,:,nb))
  call random_number(arr9(:,:,nb))
  call random_number(arr10(:,:,nb))
  call random_number(arr11(:,:,:,nb))
end do

! Data offload
write(*, '(A,I)') "Number of available devices ", omp_get_num_devices()
!$omp target data map(tofrom:arr1) map(to:arr2, arr3, arr4, arr5, arr6, arr7, arr8, arr9, arr10) map(from:out1)

time1 = omp_get_wtime()

! The classic BLOCKED-NPROMA structure from the IFS


do nml=1,num_main_loops
   !ret = roctxRangePushA("Kernel"//c_null_char)
   !$omp target teams distribute parallel do
   do nb = 1, nblocks
#ifdef BREAK_COMPILATION
      do i=1_ip, nproma
#else
      i=1
#endif
         call kernel( nproma, nlev, 1_ip, nproma, arr1(:,:,nb), arr2(:,:,nb), &
              &           arr3(:,:,nb), arr4(:,:,nb), arr5(:,:,nb), &
              &           arr6(:,:,nb), arr7(:,:,nb), arr8(:,:,nb), &
              &           arr9(:,:,nb), arr10(:,:,nb), out1(:,:,nb), i )
#ifdef BREAK_COMPILATION
      end do
#endif
   end do
   !$omp end target teams distribute parallel do
   !call roctxRangePop()
   !call roctxMarkA("Kernel"//c_null_char)
   
end do !nml

time2 = omp_get_wtime()
dttotal = time2-time1

!$omp end target data

write(*, '(A,3F12.6)') 'Result check : ', arr1(1,1,1), sum(arr1) / real(npoints*nlev*nblocks), sum(out1) / real(npoints*nlev*nblocks)
write(*, '(A,F8.4)') 'Time for kernel call : ', dttotal
est=1.e-6*num_main_loops*11*npoints*nlev*nblocks*real_bytes
write(*, '(A,F12.2,A,F12.2,A)') 'Bandwidth estimate : ', est, &
     ' MB transferred: ', est / dttotal, ' MB/s '

! TODO: Should really dealloc all this stuff... :(

end program phys_layout
