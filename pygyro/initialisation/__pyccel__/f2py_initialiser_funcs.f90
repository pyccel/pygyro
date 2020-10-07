!........................................
function n0(r, CN0, kN0, deltaRN0, rp) result(Out_0001)

  use initialiser_funcs, only: mod_n0 => n0

  implicit none

  real(kind=8), intent(in) :: r
  real(kind=8), intent(in) :: CN0
  real(kind=8), intent(in) :: kN0
  real(kind=8), intent(in) :: deltaRN0
  real(kind=8), intent(in) :: rp
  real(kind=8) :: Out_0001

  Out_0001 = mod_n0(r, CN0, kN0, deltaRN0, rp)

end function n0
!........................................

!........................................
function ti(r, Cti, kti, deltaRti, rp) result(Out_0002)

  use initialiser_funcs, only: mod_ti => ti

  implicit none

  real(kind=8), intent(in) :: r
  real(kind=8), intent(in) :: Cti
  real(kind=8), intent(in) :: kti
  real(kind=8), intent(in) :: deltaRti
  real(kind=8), intent(in) :: rp
  real(kind=8) :: Out_0002

  Out_0002 = mod_ti(r, Cti, kti, deltaRti, rp)

end function ti
!........................................

!........................................
function perturbation(r, theta, z, m, n, rp, deltaR, R0) result(Out_0003 &
      )

  use initialiser_funcs, only: mod_perturbation => perturbation

  implicit none

  real(kind=8), intent(in) :: r
  real(kind=8), intent(in) :: theta
  real(kind=8), intent(in) :: z
  integer(kind=8), intent(in) :: m
  integer(kind=8), intent(in) :: n
  real(kind=8), intent(in) :: rp
  real(kind=8), intent(in) :: deltaR
  real(kind=8), intent(in) :: R0
  real(kind=8) :: Out_0003

  Out_0003 = mod_perturbation(r, theta, z, m, n, rp, deltaR, R0)

end function perturbation
!........................................

!........................................
function f_eq(r, vPar, CN0, kN0, deltaRN0, rp, Cti, kti, deltaRti) &
      result(Out_0004)

  use initialiser_funcs, only: mod_f_eq => f_eq

  implicit none

  real(kind=8), intent(in) :: r
  real(kind=8), intent(in) :: vPar
  real(kind=8), intent(in) :: CN0
  real(kind=8), intent(in) :: kN0
  real(kind=8), intent(in) :: deltaRN0
  real(kind=8), intent(in) :: rp
  real(kind=8), intent(in) :: Cti
  real(kind=8), intent(in) :: kti
  real(kind=8), intent(in) :: deltaRti
  real(kind=8) :: Out_0004

  Out_0004 = mod_f_eq(r, vPar, CN0, kN0, deltaRN0, rp, Cti, kti, &
      deltaRti)

end function f_eq
!........................................

!........................................
function n0deriv_normalised(r, kN0, rp, deltaRN0) result(Out_0005)

  use initialiser_funcs, only: mod_n0deriv_normalised => &
      n0deriv_normalised

  implicit none

  real(kind=8), intent(in) :: r
  real(kind=8), intent(in) :: kN0
  real(kind=8), intent(in) :: rp
  real(kind=8), intent(in) :: deltaRN0
  real(kind=8) :: Out_0005

  Out_0005 = mod_n0deriv_normalised(r, kN0, rp, deltaRN0)

end function n0deriv_normalised
!........................................

!........................................
function te(r, Cte, kte, deltaRte, rp) result(Out_0006)

  use initialiser_funcs, only: mod_te => te

  implicit none

  real(kind=8), intent(in) :: r
  real(kind=8), intent(in) :: Cte
  real(kind=8), intent(in) :: kte
  real(kind=8), intent(in) :: deltaRte
  real(kind=8), intent(in) :: rp
  real(kind=8) :: Out_0006

  Out_0006 = mod_te(r, Cte, kte, deltaRte, rp)

end function te
!........................................

!........................................
function init_f(r, theta, z, vPar, m, n, eps, CN0, kN0, deltaRN0, rp, &
      Cti, kti, deltaRti, deltaR, R0) result(Out_0007)

  use initialiser_funcs, only: mod_init_f => init_f

  implicit none

  real(kind=8), intent(in) :: r
  real(kind=8), intent(in) :: theta
  real(kind=8), intent(in) :: z
  real(kind=8), intent(in) :: vPar
  integer(kind=8), intent(in) :: m
  integer(kind=8), intent(in) :: n
  real(kind=8), intent(in) :: eps
  real(kind=8), intent(in) :: CN0
  real(kind=8), intent(in) :: kN0
  real(kind=8), intent(in) :: deltaRN0
  real(kind=8), intent(in) :: rp
  real(kind=8), intent(in) :: Cti
  real(kind=8), intent(in) :: kti
  real(kind=8), intent(in) :: deltaRti
  real(kind=8), intent(in) :: deltaR
  real(kind=8), intent(in) :: R0
  real(kind=8) :: Out_0007

  Out_0007 = mod_init_f(r, theta, z, vPar, m, n, eps, CN0, kN0, deltaRN0 &
      , rp, Cti, kti, deltaRti, deltaR, R0)

end function init_f
!........................................

!........................................
subroutine init_f_flux(n0_surface, n1_surface, surface, r, n0_theta, &
      theta, n0_zVec, zVec, vPar, m, n, eps, CN0, kN0, deltaRN0, rp, &
      Cti, kti, deltaRti, deltaR, R0)

  use initialiser_funcs, only: mod_init_f_flux => init_f_flux

  implicit none

  integer(kind=4), intent(in) :: n0_surface
  integer(kind=4), intent(in) :: n1_surface
  real(kind=8), intent(inout) :: surface(0:n1_surface-1,0:n0_surface-1)
  real(kind=8), intent(in) :: r
  integer(kind=4), intent(in) :: n0_theta
  real(kind=8), intent(in) :: theta(0:n0_theta-1)
  integer(kind=4), intent(in) :: n0_zVec
  real(kind=8), intent(in) :: zVec(0:n0_zVec-1)
  real(kind=8), intent(in) :: vPar
  integer(kind=8), intent(in) :: m
  integer(kind=8), intent(in) :: n
  real(kind=8), intent(in) :: eps
  real(kind=8), intent(in) :: CN0
  real(kind=8), intent(in) :: kN0
  real(kind=8), intent(in) :: deltaRN0
  real(kind=8), intent(in) :: rp
  real(kind=8), intent(in) :: Cti
  real(kind=8), intent(in) :: kti
  real(kind=8), intent(in) :: deltaRti
  real(kind=8), intent(in) :: deltaR
  real(kind=8), intent(in) :: R0

  !f2py integer(kind=8) :: n0_surface=shape(surface,0)
  !f2py integer(kind=8) :: n1_surface=shape(surface,1)
  !f2py intent(c) surface
  call mod_init_f_flux(surface, r, theta, zVec, vPar, m, n, eps, CN0, &
      kN0, deltaRN0, rp, Cti, kti, deltaRti, deltaR, R0)

end subroutine init_f_flux
!........................................

!........................................
subroutine init_f_pol(n0_surface, n1_surface, surface, n0_rVec, rVec, &
      n0_theta, theta, z, vPar, m, n, eps, CN0, kN0, deltaRN0, rp, Cti, &
      kti, deltaRti, deltaR, R0)

  use initialiser_funcs, only: mod_init_f_pol => init_f_pol

  implicit none

  integer(kind=4), intent(in) :: n0_surface
  integer(kind=4), intent(in) :: n1_surface
  real(kind=8), intent(inout) :: surface(0:n1_surface-1,0:n0_surface-1)
  integer(kind=4), intent(in) :: n0_rVec
  real(kind=8), intent(in) :: rVec(0:n0_rVec-1)
  integer(kind=4), intent(in) :: n0_theta
  real(kind=8), intent(in) :: theta(0:n0_theta-1)
  real(kind=8), intent(in) :: z
  real(kind=8), intent(in) :: vPar
  integer(kind=8), intent(in) :: m
  integer(kind=8), intent(in) :: n
  real(kind=8), intent(in) :: eps
  real(kind=8), intent(in) :: CN0
  real(kind=8), intent(in) :: kN0
  real(kind=8), intent(in) :: deltaRN0
  real(kind=8), intent(in) :: rp
  real(kind=8), intent(in) :: Cti
  real(kind=8), intent(in) :: kti
  real(kind=8), intent(in) :: deltaRti
  real(kind=8), intent(in) :: deltaR
  real(kind=8), intent(in) :: R0

  !f2py integer(kind=8) :: n0_surface=shape(surface,0)
  !f2py integer(kind=8) :: n1_surface=shape(surface,1)
  !f2py intent(c) surface
  call mod_init_f_pol(surface, rVec, theta, z, vPar, m, n, eps, CN0, kN0 &
      , deltaRN0, rp, Cti, kti, deltaRti, deltaR, R0)

end subroutine init_f_pol
!........................................

!........................................
subroutine init_f_vpar(n0_surface, n1_surface, surface, r, n0_theta, &
      theta, z, n0_vPar, vPar, m, n, eps, CN0, kN0, deltaRN0, rp, Cti, &
      kti, deltaRti, deltaR, R0)

  use initialiser_funcs, only: mod_init_f_vpar => init_f_vpar

  implicit none

  integer(kind=4), intent(in) :: n0_surface
  integer(kind=4), intent(in) :: n1_surface
  real(kind=8), intent(inout) :: surface(0:n1_surface-1,0:n0_surface-1)
  real(kind=8), intent(in) :: r
  integer(kind=4), intent(in) :: n0_theta
  real(kind=8), intent(in) :: theta(0:n0_theta-1)
  real(kind=8), intent(in) :: z
  integer(kind=4), intent(in) :: n0_vPar
  real(kind=8), intent(in) :: vPar(0:n0_vPar-1)
  integer(kind=8), intent(in) :: m
  integer(kind=8), intent(in) :: n
  real(kind=8), intent(in) :: eps
  real(kind=8), intent(in) :: CN0
  real(kind=8), intent(in) :: kN0
  real(kind=8), intent(in) :: deltaRN0
  real(kind=8), intent(in) :: rp
  real(kind=8), intent(in) :: Cti
  real(kind=8), intent(in) :: kti
  real(kind=8), intent(in) :: deltaRti
  real(kind=8), intent(in) :: deltaR
  real(kind=8), intent(in) :: R0

  !f2py integer(kind=8) :: n0_surface=shape(surface,0)
  !f2py integer(kind=8) :: n1_surface=shape(surface,1)
  !f2py intent(c) surface
  call mod_init_f_vpar(surface, r, theta, z, vPar, m, n, eps, CN0, kN0, &
      deltaRN0, rp, Cti, kti, deltaRti, deltaR, R0)

end subroutine init_f_vpar
!........................................

!........................................
subroutine feq_vector(n0_surface, n1_surface, surface, n0_r_vec, r_vec, &
      n0_vPar, vPar, CN0, kN0, deltaRN0, rp, Cti, kti, deltaRti)

  use initialiser_funcs, only: mod_feq_vector => feq_vector

  implicit none

  integer(kind=4), intent(in) :: n0_surface
  integer(kind=4), intent(in) :: n1_surface
  real(kind=8), intent(inout) :: surface(0:n1_surface-1,0:n0_surface-1)
  integer(kind=4), intent(in) :: n0_r_vec
  real(kind=8), intent(in) :: r_vec(0:n0_r_vec-1)
  integer(kind=4), intent(in) :: n0_vPar
  real(kind=8), intent(in) :: vPar(0:n0_vPar-1)
  real(kind=8), intent(in) :: CN0
  real(kind=8), intent(in) :: kN0
  real(kind=8), intent(in) :: deltaRN0
  real(kind=8), intent(in) :: rp
  real(kind=8), intent(in) :: Cti
  real(kind=8), intent(in) :: kti
  real(kind=8), intent(in) :: deltaRti

  !f2py integer(kind=8) :: n0_surface=shape(surface,0)
  !f2py integer(kind=8) :: n1_surface=shape(surface,1)
  !f2py intent(c) surface
  call mod_feq_vector(surface, r_vec, vPar, CN0, kN0, deltaRN0, rp, Cti, &
      kti, deltaRti)

end subroutine feq_vector
!........................................
