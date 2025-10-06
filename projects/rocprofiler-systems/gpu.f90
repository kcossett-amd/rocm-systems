PROGRAM omp_hello_world

  ! Fortran environment module
  USE, INTRINSIC :: ISO_FORTRAN_ENV, ONLY: dp => real64

  ! OpenMP module
  USE omp_lib

  IMPLICIT NONE

  ! Intrinsic functions
  INTRINSIC :: SIN, COS, REAL

  INTEGER, PARAMETER :: N = 100000
  INTEGER i
  REAL(dp) :: a(N), b(N), c(N)
  REAL(dp) :: sum, tstart, tend

  !$OMP TARGET DATA MAP(ALLOC: a, b, c) MAP(FROM: sum, tstart, tend)

  !$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO
  DO i = 1, N
    a(i) = SIN(REAL(i, KIND=dp))**2
    b(i) = COS(REAL(i, KIND=dp))**2
    c(i) = 0.0_dp
  END DO

  tstart = omp_get_wtime()

  !$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO
  DO i = 1, N
    c(i) = a(i) + b(i)
  END DO

  sum = 0.0_dp
  !$OMP TARGET TEAMS DISTRIBUTE PARALLEL DO REDUCTION(+:sum)
  DO i = 1, N
    sum = sum + c(i)
  END DO

  tend = omp_get_wtime()

  !$OMP END TARGET DATA

  sum = sum / REAL(N, KIND=dp)

  PRINT '("Runtime is: ",F9.6," secs")', tend-tstart

  PRINT '("Final result: ",F9.6)', sum

END PROGRAM omp_hello_world