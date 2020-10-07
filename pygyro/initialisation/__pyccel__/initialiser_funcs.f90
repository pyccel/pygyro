module initialiser_funcs

implicit none

contains

!........................................
pure  function n0(r, CN0, kN0, deltaRN0, rp) result(Out_0001)

implicit none

real(kind=8) :: Out_0001
real(kind=8), value :: r
real(kind=8), value :: CN0
real(kind=8), value :: kN0
real(kind=8), value :: deltaRN0
real(kind=8), value :: rp

Out_0001 = CN0 * exp(-kN0 * deltaRN0 * tanh((r - rp) / deltaRN0))
return

end function n0
!........................................

!........................................
pure  function ti(r, Cti, kti, deltaRti, rp) result(Out_0002)

implicit none

real(kind=8) :: Out_0002
real(kind=8), value :: r
real(kind=8), value :: Cti
real(kind=8), value :: kti
real(kind=8), value :: deltaRti
real(kind=8), value :: rp

Out_0002 = Cti * exp(-kti * deltaRti * tanh((r - rp) / deltaRti))
return

end function ti
!........................................

!........................................
pure  function perturbation(r, theta, z, m, n, rp, deltaR, R0) result( &
      Out_0003)

implicit none

real(kind=8) :: Out_0003
real(kind=8), value :: r
real(kind=8), value :: theta
real(kind=8), value :: z
integer(kind=8), value :: m
integer(kind=8), value :: n
real(kind=8), value :: rp
real(kind=8), value :: deltaR
real(kind=8), value :: R0

Out_0003 = exp(-(r - rp) ** 2_8 / deltaR) * cos(m * theta + n * z / R0)
return

end function perturbation
!........................................

!........................................
pure  function f_eq(r, vPar, CN0, kN0, deltaRN0, rp, Cti, kti, deltaRti &
      ) result(Out_0004)

implicit none

real(kind=8) :: Out_0004
real(kind=8), value :: r
real(kind=8), value :: vPar
real(kind=8), value :: CN0
real(kind=8), value :: kN0
real(kind=8), value :: deltaRN0
real(kind=8), value :: rp
real(kind=8), value :: Cti
real(kind=8), value :: kti
real(kind=8), value :: deltaRti

Out_0004 = n0(r, CN0, kN0, deltaRN0, rp) * exp(-0.5d0 * vPar * vPar / ti &
      (r, Cti, kti, deltaRti, rp)) / Real(sqrt(2.0d0 * &
      3.14159265358979d0 * ti(r, Cti, kti, deltaRti, rp)), 8)
return

end function f_eq
!........................................

!........................................
pure  function n0deriv_normalised(r, kN0, rp, deltaRN0) result(Out_0005)

implicit none

real(kind=8) :: Out_0005
real(kind=8), value :: r
real(kind=8), value :: kN0
real(kind=8), value :: rp
real(kind=8), value :: deltaRN0

Out_0005 = -kN0 * (1_8 - tanh((r - rp) / deltaRN0) ** 2_8)
return

end function n0deriv_normalised
!........................................

!........................................
pure  function te(r, Cte, kte, deltaRte, rp) result(Out_0006)

implicit none

real(kind=8) :: Out_0006
real(kind=8), value :: r
real(kind=8), value :: Cte
real(kind=8), value :: kte
real(kind=8), value :: deltaRte
real(kind=8), value :: rp

Out_0006 = Cte * exp(-kte * deltaRte * tanh((r - rp) / deltaRte))
return

end function te
!........................................

!........................................
pure  function init_f(r, theta, z, vPar, m, n, eps, CN0, kN0, deltaRN0, &
      rp, Cti, kti, deltaRti, deltaR, R0) result(Out_0007)

implicit none

real(kind=8) :: Out_0007
real(kind=8), value :: r
real(kind=8), value :: theta
real(kind=8), value :: z
real(kind=8), value :: vPar
integer(kind=8), value :: m
integer(kind=8), value :: n
real(kind=8), value :: eps
real(kind=8), value :: CN0
real(kind=8), value :: kN0
real(kind=8), value :: deltaRN0
real(kind=8), value :: rp
real(kind=8), value :: Cti
real(kind=8), value :: kti
real(kind=8), value :: deltaRti
real(kind=8), value :: deltaR
real(kind=8), value :: R0

Out_0007 = f_eq(r, vPar, CN0, kN0, deltaRN0, rp, Cti, kti, deltaRti) * ( &
      1_8 + eps * perturbation(r, theta, z, m, n, rp, deltaR, R0))
return

end function init_f
!........................................

!........................................
pure  subroutine init_f_flux(surface, r, theta, zVec, vPar, m, n, eps, &
      CN0, kN0, deltaRN0, rp, Cti, kti, deltaRti, deltaR, R0)

implicit none

real(kind=8), intent(inout) :: surface(0:,0:)
real(kind=8), value :: r
real(kind=8), intent(in) :: theta(0:)
real(kind=8), intent(in) :: zVec(0:)
real(kind=8), value :: vPar
integer(kind=8), value :: m
integer(kind=8), value :: n
real(kind=8), value :: eps
real(kind=8), value :: CN0
real(kind=8), value :: kN0
real(kind=8), value :: deltaRN0
real(kind=8), value :: rp
real(kind=8), value :: Cti
real(kind=8), value :: kti
real(kind=8), value :: deltaRti
real(kind=8), value :: deltaR
real(kind=8), value :: R0
integer(kind=8) :: i
real(kind=8) :: q
integer(kind=8) :: j
real(kind=8) :: z

do i = 0_8, size(theta,1)-1_8, 1_8
q = theta(i)
do j = 0_8, size(zVec,1)-1_8, 1_8
z = zVec(j)
surface(j, i) = f_eq(r, vPar, CN0, kN0, deltaRN0, rp, Cti, kti, deltaRti &
      ) * (1_8 + eps * perturbation(r, q, z, m, n, rp, deltaR, R0))
end do
end do

end subroutine init_f_flux
!........................................

!........................................
pure  subroutine init_f_pol(surface, rVec, theta, z, vPar, m, n, eps, &
      CN0, kN0, deltaRN0, rp, Cti, kti, deltaRti, deltaR, R0)

implicit none

real(kind=8), intent(inout) :: surface(0:,0:)
real(kind=8), intent(in) :: rVec(0:)
real(kind=8), intent(in) :: theta(0:)
real(kind=8), value :: z
real(kind=8), value :: vPar
integer(kind=8), value :: m
integer(kind=8), value :: n
real(kind=8), value :: eps
real(kind=8), value :: CN0
real(kind=8), value :: kN0
real(kind=8), value :: deltaRN0
real(kind=8), value :: rp
real(kind=8), value :: Cti
real(kind=8), value :: kti
real(kind=8), value :: deltaRti
real(kind=8), value :: deltaR
real(kind=8), value :: R0
integer(kind=8) :: i
real(kind=8) :: q
integer(kind=8) :: j
real(kind=8) :: r

do i = 0_8, size(theta,1)-1_8, 1_8
q = theta(i)
do j = 0_8, size(rVec,1)-1_8, 1_8
r = rVec(j)
surface(j, i) = f_eq(r, vPar, CN0, kN0, deltaRN0, rp, Cti, kti, deltaRti &
      ) * (1_8 + eps * perturbation(r, q, z, m, n, rp, deltaR, R0))
end do
end do

end subroutine init_f_pol
!........................................

!........................................
pure  subroutine init_f_vpar(surface, r, theta, z, vPar, m, n, eps, CN0, &
      kN0, deltaRN0, rp, Cti, kti, deltaRti, deltaR, R0)

implicit none

real(kind=8), intent(inout) :: surface(0:,0:)
real(kind=8), value :: r
real(kind=8), intent(in) :: theta(0:)
real(kind=8), value :: z
real(kind=8), intent(in) :: vPar(0:)
integer(kind=8), value :: m
integer(kind=8), value :: n
real(kind=8), value :: eps
real(kind=8), value :: CN0
real(kind=8), value :: kN0
real(kind=8), value :: deltaRN0
real(kind=8), value :: rp
real(kind=8), value :: Cti
real(kind=8), value :: kti
real(kind=8), value :: deltaRti
real(kind=8), value :: deltaR
real(kind=8), value :: R0
integer(kind=8) :: i
real(kind=8) :: q
integer(kind=8) :: j
real(kind=8) :: v

do i = 0_8, size(theta,1)-1_8, 1_8
q = theta(i)
do j = 0_8, size(vPar,1)-1_8, 1_8
v = vPar(j)
surface(j, i) = f_eq(r, v, CN0, kN0, deltaRN0, rp, Cti, kti, deltaRti) * &
      (1_8 + eps * perturbation(r, q, z, m, n, rp, deltaR, R0))
end do
end do

end subroutine init_f_vpar
!........................................

!........................................
pure  subroutine feq_vector(surface, r_vec, vPar, CN0, kN0, deltaRN0, rp &
      , Cti, kti, deltaRti)

implicit none

real(kind=8), intent(inout) :: surface(0:,0:)
real(kind=8), intent(in) :: r_vec(0:)
real(kind=8), intent(in) :: vPar(0:)
real(kind=8), value :: CN0
real(kind=8), value :: kN0
real(kind=8), value :: deltaRN0
real(kind=8), value :: rp
real(kind=8), value :: Cti
real(kind=8), value :: kti
real(kind=8), value :: deltaRti
integer(kind=8) :: i
real(kind=8) :: r
integer(kind=8) :: j
real(kind=8) :: v

do i = 0_8, size(r_vec,1)-1_8, 1_8
r = r_vec(i)
do j = 0_8, size(vPar,1)-1_8, 1_8
v = vPar(j)
surface(j, i) = f_eq(r, v, CN0, kN0, deltaRN0, rp, Cti, kti, deltaRti)
end do
end do

end subroutine feq_vector
!........................................

end module initialiser_funcs
