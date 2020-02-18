module initialiser_funcs

implicit none




contains

!........................................
pure function n0(r, CN0, kN0, deltaRN0, rp) result(Dummy_3122)

implicit none
real(kind=8) :: Dummy_3122  
real(kind=8), intent(in)  :: r 
real(kind=8), intent(in)  :: CN0 
real(kind=8), intent(in)  :: kN0 
real(kind=8), intent(in)  :: deltaRN0 
real(kind=8), intent(in)  :: rp 


Dummy_3122 = CN0*exp((-kN0)*(deltaRN0*tanh((r - rp)/deltaRN0)))
return
end function
!........................................

!........................................
pure function ti(r, Cti, kti, deltaRti, rp) result(Dummy_8566)

implicit none
real(kind=8) :: Dummy_8566  
real(kind=8), intent(in)  :: r 
real(kind=8), intent(in)  :: Cti 
real(kind=8), intent(in)  :: kti 
real(kind=8), intent(in)  :: deltaRti 
real(kind=8), intent(in)  :: rp 


Dummy_8566 = Cti*exp((-kti)*(deltaRti*tanh((r - rp)/deltaRti)))
return
end function
!........................................

!........................................
pure function perturbation(r, theta, z, m, n, rp, deltaR, R0) result( &
      Dummy_8372)

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


Dummy_8372 = exp((-(r - rp)**2)/deltaR)*cos(m*theta + n*(z/R0))
return
end function
!........................................

!........................................
pure function f_eq(r, vPar, CN0, kN0, deltaRN0, rp, Cti, kti, deltaRti) &
      result(Dummy_4753)

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


Dummy_4753 = (exp(-0.5d0*vPar*vPar/ti(r, Cti, kti, deltaRti, rp))/Real( &
      sqrt(2.0d0*(3.14159265358979d0*ti(r, Cti, kti, deltaRti, rp))), 8 &
      ))*n0(r, CN0, kN0, deltaRN0, rp)
return
end function
!........................................

!........................................
pure function n0deriv_normalised(r, kN0, rp, deltaRN0) result(Dummy_6369 &
      )

implicit none
real(kind=8) :: Dummy_6369  
real(kind=8), intent(in)  :: r 
real(kind=8), intent(in)  :: kN0 
real(kind=8), intent(in)  :: rp 
real(kind=8), intent(in)  :: deltaRN0 


Dummy_6369 = (-kN0)*(1 - tanh((r - rp)/deltaRN0)**2)
return
end function
!........................................

!........................................
pure function te(r, Cte, kte, deltaRte, rp) result(Dummy_7381)

implicit none
real(kind=8) :: Dummy_7381  
real(kind=8), intent(in)  :: r 
real(kind=8), intent(in)  :: Cte 
real(kind=8), intent(in)  :: kte 
real(kind=8), intent(in)  :: deltaRte 
real(kind=8), intent(in)  :: rp 


Dummy_7381 = Cte*exp((-kte)*(deltaRte*tanh((r - rp)/deltaRte)))
return
end function
!........................................

!........................................
pure function init_f(r, theta, z, vPar, m, n, eps, CN0, kN0, deltaRN0, &
      rp, Cti, kti, deltaRti, deltaR, R0) result(Dummy_5048)

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

Dummy_5048 = (eps*perturbation(r, theta, z, m, n, rp, deltaR, R0) + 1)* &
      f_eq(r, vPar, CN0, kN0, deltaRN0, rp, Cti, kti, deltaRti)
return
end function
!........................................

!........................................
pure subroutine init_f_flux(surface, r, theta, zVec, vPar, m, n, eps, &
      CN0, kN0, deltaRN0, rp, Cti, kti, deltaRti, deltaR, R0)

implicit none
real(kind=8), intent(inout)  :: surface (0:,0:)
real(kind=8), intent(in)  :: r 
real(kind=8), intent(in)  :: theta (0:)
real(kind=8), intent(in)  :: zVec (0:)
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
integer(kind=4) :: i  
real(kind=8) :: q  
integer(kind=4) :: j  
real(kind=8) :: z  

do i = 0, size(theta,1) - 1, 1
q = theta(i)
do j = 0, size(zVec,1) - 1, 1
z = zVec(j)
surface(j, i) = (eps*perturbation(r, q, z, m, n, rp, deltaR, R0) + 1)* &
      f_eq(r, vPar, CN0, kN0, deltaRN0, rp, Cti, kti, deltaRti)


end do

end do

end subroutine
!........................................

!........................................
pure subroutine init_f_pol(surface, rVec, theta, z, vPar, m, n, eps, CN0 &
      , kN0, deltaRN0, rp, Cti, kti, deltaRti, deltaR, R0)

implicit none
real(kind=8), intent(inout)  :: surface (0:,0:)
real(kind=8), intent(in)  :: rVec (0:)
real(kind=8), intent(in)  :: theta (0:)
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
integer(kind=4) :: i  
real(kind=8) :: q  
integer(kind=4) :: j  
real(kind=8) :: r  

do i = 0, size(theta,1) - 1, 1
q = theta(i)
do j = 0, size(rVec,1) - 1, 1
r = rVec(j)
surface(j, i) = (eps*perturbation(r, q, z, m, n, rp, deltaR, R0) + 1)* &
      f_eq(r, vPar, CN0, kN0, deltaRN0, rp, Cti, kti, deltaRti)


end do

end do

end subroutine
!........................................

!........................................
pure subroutine init_f_vpar(surface, r, theta, z, vPar, m, n, eps, CN0, &
      kN0, deltaRN0, rp, Cti, kti, deltaRti, deltaR, R0)

implicit none
real(kind=8), intent(inout)  :: surface (0:,0:)
real(kind=8), intent(in)  :: r 
real(kind=8), intent(in)  :: theta (0:)
real(kind=8), intent(in)  :: z 
real(kind=8), intent(in)  :: vPar (0:)
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
integer(kind=4) :: i  
real(kind=8) :: q  
integer(kind=4) :: j  
real(kind=8) :: v  

do i = 0, size(theta,1) - 1, 1
q = theta(i)
do j = 0, size(vPar,1) - 1, 1
v = vPar(j)
surface(j, i) = (eps*perturbation(r, q, z, m, n, rp, deltaR, R0) + 1)* &
      f_eq(r, v, CN0, kN0, deltaRN0, rp, Cti, kti, deltaRti)


end do

end do

end subroutine
!........................................

!........................................
pure subroutine feq_vector(surface, r_vec, vPar, CN0, kN0, deltaRN0, rp, &
      Cti, kti, deltaRti)

implicit none
real(kind=8), intent(inout)  :: surface (0:,0:)
real(kind=8), intent(in)  :: r_vec (0:)
real(kind=8), intent(in)  :: vPar (0:)
real(kind=8), intent(in)  :: CN0 
real(kind=8), intent(in)  :: kN0 
real(kind=8), intent(in)  :: deltaRN0 
real(kind=8), intent(in)  :: rp 
real(kind=8), intent(in)  :: Cti 
real(kind=8), intent(in)  :: kti 
real(kind=8), intent(in)  :: deltaRti 
integer(kind=4) :: i  
real(kind=8) :: r  
integer(kind=4) :: j  
real(kind=8) :: v  

do i = 0, size(r_vec,1) - 1, 1
r = r_vec(i)
do j = 0, size(vPar,1) - 1, 1
v = vPar(j)
surface(j, i) = f_eq(r, v, CN0, kN0, deltaRN0, rp, Cti, kti, deltaRti)


end do

end do

end subroutine
!........................................

end module