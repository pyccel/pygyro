function n0(r, CN0, kN0, deltaRN0, rp) result(Dummy_3122)

  use initialiser_funcs, only: mod_n0 => n0
  implicit none
  real(kind=8) :: Dummy_3122  
  real(kind=8), intent(in)  :: r 
  real(kind=8), intent(in)  :: CN0 
  real(kind=8), intent(in)  :: kN0 
  real(kind=8), intent(in)  :: deltaRN0 
  real(kind=8), intent(in)  :: rp 

  Dummy_3122 = mod_n0(r,CN0,kN0,deltaRN0,rp)
end function

function ti(r, Cti, kti, deltaRti, rp) result(Dummy_8566)

  use initialiser_funcs, only: mod_ti => ti
  implicit none
  real(kind=8) :: Dummy_8566  
  real(kind=8), intent(in)  :: r 
  real(kind=8), intent(in)  :: Cti 
  real(kind=8), intent(in)  :: kti 
  real(kind=8), intent(in)  :: deltaRti 
  real(kind=8), intent(in)  :: rp 

  Dummy_8566 = mod_ti(r,Cti,kti,deltaRti,rp)
end function

function perturbation(r, theta, z, m, n, rp, deltaR, R0) result( &
      Dummy_8372)

  use initialiser_funcs, only: mod_perturbation => perturbation
  implicit none
  real(kind=8) :: Dummy_8372  
  real(kind=8), intent(in)  :: r 
  real(kind=8), intent(in)  :: theta 
  real(kind=8), intent(in)  :: z 
  integer(kind=4), intent(in)  :: m 
  integer(kind=4), intent(in)  :: n 
  real(kind=8), intent(in)  :: rp 
  real(kind=8), intent(in)  :: deltaR 
  real(kind=8), intent(in)  :: R0 

  Dummy_8372 = mod_perturbation(r,theta,z,m,n,rp,deltaR,R0)
end function

function f_eq(r, vPar, CN0, kN0, deltaRN0, rp, Cti, kti, deltaRti) &
      result(Dummy_4753)

  use initialiser_funcs, only: mod_f_eq => f_eq
  implicit none
  real(kind=8) :: Dummy_4753  
  real(kind=8), intent(in)  :: r 
  real(kind=8), intent(in)  :: vPar 
  real(kind=8), intent(in)  :: CN0 
  real(kind=8), intent(in)  :: kN0 
  real(kind=8), intent(in)  :: deltaRN0 
  real(kind=8), intent(in)  :: rp 
  real(kind=8), intent(in)  :: Cti 
  real(kind=8), intent(in)  :: kti 
  real(kind=8), intent(in)  :: deltaRti 

  Dummy_4753 = mod_f_eq(r,vPar,CN0,kN0,deltaRN0,rp,Cti,kti,deltaRti)
end function

function n0deriv_normalised(r, kN0, rp, deltaRN0) result(Dummy_6369)

  use initialiser_funcs, only: mod_n0deriv_normalised => &
      n0deriv_normalised
  implicit none
  real(kind=8) :: Dummy_6369  
  real(kind=8), intent(in)  :: r 
  real(kind=8), intent(in)  :: kN0 
  real(kind=8), intent(in)  :: rp 
  real(kind=8), intent(in)  :: deltaRN0 

  Dummy_6369 = mod_n0deriv_normalised(r,kN0,rp,deltaRN0)
end function

function te(r, Cte, kte, deltaRte, rp) result(Dummy_7381)

  use initialiser_funcs, only: mod_te => te
  implicit none
  real(kind=8) :: Dummy_7381  
  real(kind=8), intent(in)  :: r 
  real(kind=8), intent(in)  :: Cte 
  real(kind=8), intent(in)  :: kte 
  real(kind=8), intent(in)  :: deltaRte 
  real(kind=8), intent(in)  :: rp 

  Dummy_7381 = mod_te(r,Cte,kte,deltaRte,rp)
end function

function init_f(r, theta, z, vPar, m, n, eps, CN0, kN0, deltaRN0, rp, &
      Cti, kti, deltaRti, deltaR, R0) result(Dummy_5048)

  use initialiser_funcs, only: mod_init_f => init_f
  implicit none
  real(kind=8) :: Dummy_5048  
  real(kind=8), intent(in)  :: r 
  real(kind=8), intent(in)  :: theta 
  real(kind=8), intent(in)  :: z 
  real(kind=8), intent(in)  :: vPar 
  integer(kind=4), intent(in)  :: m 
  integer(kind=4), intent(in)  :: n 
  real(kind=8), intent(in)  :: eps 
  real(kind=8), intent(in)  :: CN0 
  real(kind=8), intent(in)  :: kN0 
  real(kind=8), intent(in)  :: deltaRN0 
  real(kind=8), intent(in)  :: rp 
  real(kind=8), intent(in)  :: Cti 
  real(kind=8), intent(in)  :: kti 
  real(kind=8), intent(in)  :: deltaRti 
  real(kind=8), intent(in)  :: deltaR 
  real(kind=8), intent(in)  :: R0 

  Dummy_5048 = mod_init_f(r,theta,z,vPar,m,n,eps,CN0,kN0,deltaRN0,rp,Cti &
      ,kti,deltaRti,deltaR,R0)
end function

subroutine init_f_flux(n0_surface, n1_surface, surface, r, n0_theta, &
      theta, n0_zVec, zVec, vPar, m, n, eps, CN0, kN0, deltaRN0, rp, &
      Cti, kti, deltaRti, deltaR, R0)

  use initialiser_funcs, only: mod_init_f_flux => init_f_flux
  implicit none
  integer(kind=4), intent(in)  :: n0_surface 
  integer(kind=4), intent(in)  :: n1_surface 
  real(kind=8), intent(inout)  :: surface (0:n0_surface - 1,0:n1_surface &
      - 1)
  real(kind=8), intent(in)  :: r 
  integer(kind=4), intent(in)  :: n0_theta 
  real(kind=8), intent(in)  :: theta (0:n0_theta - 1)
  integer(kind=4), intent(in)  :: n0_zVec 
  real(kind=8), intent(in)  :: zVec (0:n0_zVec - 1)
  real(kind=8), intent(in)  :: vPar 
  integer(kind=4), intent(in)  :: m 
  integer(kind=4), intent(in)  :: n 
  real(kind=8), intent(in)  :: eps 
  real(kind=8), intent(in)  :: CN0 
  real(kind=8), intent(in)  :: kN0 
  real(kind=8), intent(in)  :: deltaRN0 
  real(kind=8), intent(in)  :: rp 
  real(kind=8), intent(in)  :: Cti 
  real(kind=8), intent(in)  :: kti 
  real(kind=8), intent(in)  :: deltaRti 
  real(kind=8), intent(in)  :: deltaR 
  real(kind=8), intent(in)  :: R0 

  !f2py integer(kind=4) :: n0_surface=shape(surface,1)
  !f2py integer(kind=4) :: n1_surface=shape(surface,0)
  !f2py intent(c) surface
  call mod_init_f_flux(surface,r,theta,zVec,vPar,m,n,eps,CN0,kN0, &
      deltaRN0,rp,Cti,kti,deltaRti,deltaR,R0)
end subroutine

subroutine init_f_pol(n0_surface, n1_surface, surface, n0_rVec, rVec, &
      n0_theta, theta, z, vPar, m, n, eps, CN0, kN0, deltaRN0, rp, Cti, &
      kti, deltaRti, deltaR, R0)

  use initialiser_funcs, only: mod_init_f_pol => init_f_pol
  implicit none
  integer(kind=4), intent(in)  :: n0_surface 
  integer(kind=4), intent(in)  :: n1_surface 
  real(kind=8), intent(inout)  :: surface (0:n0_surface - 1,0:n1_surface &
      - 1)
  integer(kind=4), intent(in)  :: n0_rVec 
  real(kind=8), intent(in)  :: rVec (0:n0_rVec - 1)
  integer(kind=4), intent(in)  :: n0_theta 
  real(kind=8), intent(in)  :: theta (0:n0_theta - 1)
  real(kind=8), intent(in)  :: z 
  real(kind=8), intent(in)  :: vPar 
  integer(kind=4), intent(in)  :: m 
  integer(kind=4), intent(in)  :: n 
  real(kind=8), intent(in)  :: eps 
  real(kind=8), intent(in)  :: CN0 
  real(kind=8), intent(in)  :: kN0 
  real(kind=8), intent(in)  :: deltaRN0 
  real(kind=8), intent(in)  :: rp 
  real(kind=8), intent(in)  :: Cti 
  real(kind=8), intent(in)  :: kti 
  real(kind=8), intent(in)  :: deltaRti 
  real(kind=8), intent(in)  :: deltaR 
  real(kind=8), intent(in)  :: R0 

  !f2py integer(kind=4) :: n0_surface=shape(surface,1)
  !f2py integer(kind=4) :: n1_surface=shape(surface,0)
  !f2py intent(c) surface
  call mod_init_f_pol(surface,rVec,theta,z,vPar,m,n,eps,CN0,kN0,deltaRN0 &
      ,rp,Cti,kti,deltaRti,deltaR,R0)
end subroutine

subroutine init_f_vpar(n0_surface, n1_surface, surface, r, n0_theta, &
      theta, z, n0_vPar, vPar, m, n, eps, CN0, kN0, deltaRN0, rp, Cti, &
      kti, deltaRti, deltaR, R0)

  use initialiser_funcs, only: mod_init_f_vpar => init_f_vpar
  implicit none
  integer(kind=4), intent(in)  :: n0_surface 
  integer(kind=4), intent(in)  :: n1_surface 
  real(kind=8), intent(inout)  :: surface (0:n0_surface - 1,0:n1_surface &
      - 1)
  real(kind=8), intent(in)  :: r 
  integer(kind=4), intent(in)  :: n0_theta 
  real(kind=8), intent(in)  :: theta (0:n0_theta - 1)
  real(kind=8), intent(in)  :: z 
  integer(kind=4), intent(in)  :: n0_vPar 
  real(kind=8), intent(in)  :: vPar (0:n0_vPar - 1)
  integer(kind=4), intent(in)  :: m 
  integer(kind=4), intent(in)  :: n 
  real(kind=8), intent(in)  :: eps 
  real(kind=8), intent(in)  :: CN0 
  real(kind=8), intent(in)  :: kN0 
  real(kind=8), intent(in)  :: deltaRN0 
  real(kind=8), intent(in)  :: rp 
  real(kind=8), intent(in)  :: Cti 
  real(kind=8), intent(in)  :: kti 
  real(kind=8), intent(in)  :: deltaRti 
  real(kind=8), intent(in)  :: deltaR 
  real(kind=8), intent(in)  :: R0 

  !f2py integer(kind=4) :: n0_surface=shape(surface,1)
  !f2py integer(kind=4) :: n1_surface=shape(surface,0)
  !f2py intent(c) surface
  call mod_init_f_vpar(surface,r,theta,z,vPar,m,n,eps,CN0,kN0,deltaRN0, &
      rp,Cti,kti,deltaRti,deltaR,R0)
end subroutine

subroutine feq_vector(n0_surface, n1_surface, surface, n0_r_vec, r_vec, &
      n0_vPar, vPar, CN0, kN0, deltaRN0, rp, Cti, kti, deltaRti)

  use initialiser_funcs, only: mod_feq_vector => feq_vector
  implicit none
  integer(kind=4), intent(in)  :: n0_surface 
  integer(kind=4), intent(in)  :: n1_surface 
  real(kind=8), intent(inout)  :: surface (0:n0_surface - 1,0:n1_surface &
      - 1)
  integer(kind=4), intent(in)  :: n0_r_vec 
  real(kind=8), intent(in)  :: r_vec (0:n0_r_vec - 1)
  integer(kind=4), intent(in)  :: n0_vPar 
  real(kind=8), intent(in)  :: vPar (0:n0_vPar - 1)
  real(kind=8), intent(in)  :: CN0 
  real(kind=8), intent(in)  :: kN0 
  real(kind=8), intent(in)  :: deltaRN0 
  real(kind=8), intent(in)  :: rp 
  real(kind=8), intent(in)  :: Cti 
  real(kind=8), intent(in)  :: kti 
  real(kind=8), intent(in)  :: deltaRti 

  !f2py integer(kind=4) :: n0_surface=shape(surface,1)
  !f2py integer(kind=4) :: n1_surface=shape(surface,0)
  !f2py intent(c) surface
  call mod_feq_vector(surface,r_vec,vPar,CN0,kN0,deltaRN0,rp,Cti,kti, &
      deltaRti)
end subroutine