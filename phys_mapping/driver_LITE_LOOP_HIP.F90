#define __NPROMA__ 64

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

INTERFACE

   SUBROUTINE phys_kernel_lite_loop_hip(dim1,dim2,dim3,i1,i2,nt,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,out1,dt) BIND(C)
     USE, INTRINSIC :: ISO_C_BINDING
     !USE phys_mod, ONLY: ip, lp
     !IMPLICIT NONE
     integer(4),intent(in) :: dim1, dim2, dim3, i1, i2, nt
     real(8),dimension(1:dim1,1:dim2,1:dim3),intent(in) :: in2,in3,in4,in5,in6,in7,in8,in9,in10
     real(8),dimension(1:dim1,1:dim2,1:dim3),intent(inout) :: in1, out1
     real(8),intent(inout) :: dt
   END SUBROUTINE phys_kernel_lite_loop_hip
END INTERFACE

end module mod


!
! Main subroutine
!

program phys_layout

use omp_lib

use mod, only: ip, lp, phys_kernel_lite_loop_hip

implicit none

!! arrays for passing to subroutine
real(kind=lp),dimension(:,:,:),allocatable :: arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8, arr9, arr10
real(kind=lp),dimension(:,:,:),allocatable :: out1

integer(kind=ip) :: nproma, npoints, nlev, nblocks, ntotal, num_main_loops, nThreads
integer(kind=ip) :: nb, nml, inp, i, k, real_bytes
real(kind=lp) :: est, dt, dttotal

integer :: i_seed
integer, dimension(:), allocatable :: a_seed

integer :: iargs, lenarg
character(len=20) :: clarg

real(kind=lp) :: time1, time2

! Constants
nlev    = 137
num_main_loops = 8

! Defaults
nproma = __NPROMA__
ntotal = 64*16384
nThreads = 1

iargs = command_argument_count()
if (iargs >= 1) then
   call get_command_argument(1, clarg, lenarg)
   read(clarg(1:lenarg),*) nThreads

   if (iargs >= 2) then
      call get_command_argument(2, clarg, lenarg)
      read(clarg(1:lenarg),*) nproma

      if (iargs >= 3) then
         call get_command_argument(3, clarg, lenarg)
         read(clarg(1:lenarg),*) ntotal
      endif
   endif
endif
npoints = nproma  ! Can be unlocked later...
nblocks = ntotal / nproma
write(*, *)
write(*, '(A10,I8,A10,I8,A10,I8,A14,I8,A)') 'NPROMA ', nproma, ' TOTAL ', ntotal, ' NBLOCKS ', nblocks, 'NTHREADS ', nThreads

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

! ----- Set up random seed portably -----
call random_seed(size=i_seed)
allocate(a_seed(1:i_seed))
call random_seed(get=a_seed)
a_seed = [ (i,i=1,i_seed) ]
call random_seed(put=a_seed)
deallocate(a_seed)

! #ifdef OMP_HOST
! Initialize values in parallel region to avoid accidental
! NUMA issues in single-socket mode on dual-socket machines.
!$omp parallel

!$omp master
! #endif
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
end do
! #ifdef OMP_HOST
!$omp end master

!$omp end parallel
! #endif

do nml=1,num_main_loops

    call phys_kernel_lite_loop_hip( nproma, nlev, nblocks, 1_ip, nproma, nthreads, arr1(:,:,:), arr2(:,:,:), &
    &                  arr3(:,:,:), arr4(:,:,:), arr5(:,:,:), &
    &                  arr6(:,:,:), arr7(:,:,:), arr8(:,:,:), &
    &                  arr9(:,:,:), arr10(:,:,:), out1(:,:,:), dt)
    dttotal = dttotal+dt

end do !nml

write(*, '(A,3F12.6)') 'Result check : ', arr1(1,1,1), sum(arr1) / real(npoints*nlev*nblocks), sum(out1) / real(npoints*nlev*nblocks)
write(*, '(A,F8.4)') 'Time for kernel call : ', dttotal
est=1.e-6*num_main_loops*11*npoints*nlev*nblocks*real_bytes
write(*, '(A,F12.2,A,F12.2,A)') 'Bandwidth estimate : ', est, &
     ' MB transferred: ', est / dttotal, ' MB/s '

! TODO: Should really dealloc all this stuff... :(

end program phys_layout
