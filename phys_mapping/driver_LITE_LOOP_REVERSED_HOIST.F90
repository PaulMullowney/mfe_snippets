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

real(kind=lp) :: cst1 = 2.5, cst2 = 3.14
integer, parameter :: nspecies = 5

#ifdef OMP_DEVICE
!$omp declare target(kernel)
#endif

contains

subroutine kernel(dim1,dim2,i1,i2,in1,in2,in3,in4,in5,in6,in7,in8,in9,in10,out1,i)
  integer(kind=ip),intent(in) :: dim1, dim2, i1,i2, i
  real(kind=lp),dimension(1:dim1,1:dim2),intent(in) :: in2,in3,in4,in5,in6,in7,in8,in9,in10
  real(kind=lp),dimension(1:dim1,1:dim2),intent(inout) :: in1, out1

  integer(kind=ip) :: k
  do k=1,dim2
    out1(i,k) = (in1(i,k) + in2(i,k) + in3(i,k) + in4(i,k) + in5(i,k) + &
 &               in6(i,k) + in7(i,k) + in8(i,k) + in9(i,k) + in10(i,k)) * 0.1
    in1(i,k) = out1(i,k)
  end do
end subroutine kernel

end module mod


!
! Main subroutine
!

program phys_layout

use omp_lib

use mod, only: ip, lp, kernel

USE iso_c_binding   ,ONLY : c_null_char
use hip_profiling, only : roctxRangePushA, roctxRangePop, roctxMarkA

implicit none

!! arrays for passing to subroutine
real(kind=lp),dimension(:,:,:),allocatable :: arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8, arr9, arr10
real(kind=lp),dimension(:,:,:),allocatable :: out1

integer(kind=ip) :: nproma, npoints, nlev, nblocks, ntotal, num_main_loops, case
integer(kind=ip) :: nb, nml, inp, i, k, real_bytes
real(kind=lp) :: est, dt, dttotal

integer :: i_seed, ret
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
case = 1

iargs = command_argument_count()
if (iargs >= 1) then
   call get_command_argument(1, clarg, lenarg)
   read(clarg(1:lenarg),*) case

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
write(*, '(A10,I8,A10,I8,A10,I8,A14,I8,A)') 'NPROMA ', nproma, ' TOTAL ', ntotal, ' NBLOCKS ', nblocks, ' CASE ', case

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

! Data offload
#ifdef OMP_DEVICE
write(*, '(A,I)') "Number of available devices ", omp_get_num_devices()
!$omp target data map(tofrom:arr1) map(to:arr2, arr3, arr4, arr5, arr6, arr7, arr8, arr9, arr10) map(from:out1)
#endif

time1 = omp_get_wtime()

! The classic BLOCKED-NPROMA structure from the IFS
if (case == 1) then

   do nml=1,num_main_loops

      ret = roctxRangePushA("Kernel Case 1"//c_null_char)
#if defined(OMP_DEVICE)
!$omp target teams distribute parallel do collapse(2) thread_limit(__NPROMA__)
#endif
      do nb = 1, nblocks
         do i=1_ip, nproma
            call kernel( nproma, nlev, 1_ip, nproma, arr1(:,:,nb), arr2(:,:,nb), &
                 &           arr3(:,:,nb), arr4(:,:,nb), arr5(:,:,nb), &
                 &           arr6(:,:,nb), arr7(:,:,nb), arr8(:,:,nb), &
                 &           arr9(:,:,nb), arr10(:,:,nb), out1(:,:,nb), i )
         end do
      end do

     call roctxRangePop()
     call roctxMarkA("Kernel Case 1"//c_null_char)

   end do !nml

else if (case == 2) then

   do nml=1,num_main_loops

      ret = roctxRangePushA("Kernel Case 2"//c_null_char)
#if defined(OMP_DEVICE)
!$omp target teams distribute thread_limit(__NPROMA__)
#endif
      do nb = 1, nblocks
#if defined(OMP_DEVICE)
!$omp parallel do simd
#endif
         do i=1_ip, nproma
            call kernel( nproma, nlev, 1_ip, nproma, arr1(:,:,nb), arr2(:,:,nb), &
                 &           arr3(:,:,nb), arr4(:,:,nb), arr5(:,:,nb), &
                 &           arr6(:,:,nb), arr7(:,:,nb), arr8(:,:,nb), &
                 &           arr9(:,:,nb), arr10(:,:,nb), out1(:,:,nb), i )
         end do
      end do

      call roctxRangePop()
      call roctxMarkA("Kernel Case 2"//c_null_char)

   end do !nml

else if (case == 3) then

   do nml=1,num_main_loops

      ret = roctxRangePushA("Kernel Case 3"//c_null_char)
#if defined(OMP_DEVICE)
!$omp target teams distribute thread_limit(__NPROMA__)
#endif
      do nb = 1, nblocks
         do i=1_ip, nproma
            call kernel( nproma, nlev, 1_ip, nproma, arr1(:,:,nb), arr2(:,:,nb), &
                 &           arr3(:,:,nb), arr4(:,:,nb), arr5(:,:,nb), &
                 &           arr6(:,:,nb), arr7(:,:,nb), arr8(:,:,nb), &
                 &           arr9(:,:,nb), arr10(:,:,nb), out1(:,:,nb), i )
         end do
      end do

      call roctxRangePop()
      call roctxMarkA("Kernel Case 3"//c_null_char)

   end do !nml

endif

time2 = omp_get_wtime()
dttotal = time2-time1

#ifdef OMP_DEVICE
!$omp end target data
#endif

write(*, '(A,3F12.6)') 'Result check : ', arr1(1,1,1), sum(arr1) / real(npoints*nlev*nblocks), sum(out1) / real(npoints*nlev*nblocks)
write(*, '(A,F8.4)') 'Time for kernel call : ', dttotal
est=1.e-6*num_main_loops*11*npoints*nlev*nblocks*real_bytes
write(*, '(A,F12.2,A,F12.2,A)') 'Bandwidth estimate : ', est, &
     ' MB transferred: ', est / dttotal, ' MB/s '

! TODO: Should really dealloc all this stuff... :(

end program phys_layout
