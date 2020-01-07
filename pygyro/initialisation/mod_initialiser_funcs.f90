module mod_initialiser_funcs

implicit none




contains

!........................................
pure real(kind=8) function n0(r, CN0, kN0, deltaRN0, rp)  result( &
      Dummy_0043)

implicit none
real(kind=8), intent(in)  :: r 
real(kind=8), intent(in)  :: CN0 
real(kind=8), intent(in)  :: kN0 
real(kind=8), intent(in)  :: deltaRN0 
real(kind=8), intent(in)  :: rp 


Dummy_0043 = CN0*exp((-kN0)*(deltaRN0*tanh((r - rp)/deltaRN0)))
return
end function
!........................................

!........................................
pure real(kind=8) function Ti(r, CTi, kTi, deltaRTi, rp)  result( &
      Dummy_0218)

implicit none
real(kind=8), intent(in)  :: r 
real(kind=8), intent(in)  :: CTi 
real(kind=8), intent(in)  :: kTi 
real(kind=8), intent(in)  :: deltaRTi 
real(kind=8), intent(in)  :: rp 


Dummy_0218 = CTi*exp((-kTi)*(deltaRTi*tanh((r - rp)/deltaRTi)))
return
end function
!........................................

!........................................
pure real(kind=8) function perturbation(r, theta, z, m, n, rp, deltaR, &
      R0)  result(Dummy_7739)

implicit none
real(kind=8), intent(in)  :: r 
real(kind=8), intent(in)  :: theta 
real(kind=8), intent(in)  :: z 
integer(kind=4), intent(in)  :: m 
integer(kind=4), intent(in)  :: n 
real(kind=8), intent(in)  :: rp 
real(kind=8), intent(in)  :: deltaR 
real(kind=8), intent(in)  :: R0 


Dummy_7739 = exp((-(r - rp)**2)/deltaR)*cos(m*theta + n*(z/R0))
return
end function
!........................................

!........................................
pure complex(kind=8) function fEq(r, vPar, CN0, kN0, deltaRN0, rp, CTi, &
      kTi, deltaRTi)  result(Dummy_5289)

implicit none
real(kind=8), intent(in)  :: r 
real(kind=8), intent(in)  :: vPar 
real(kind=8), intent(in)  :: CN0 
real(kind=8), intent(in)  :: kN0 
real(kind=8), intent(in)  :: deltaRN0 
real(kind=8), intent(in)  :: rp 
real(kind=8), intent(in)  :: CTi 
real(kind=8), intent(in)  :: kTi 
real(kind=8), intent(in)  :: deltaRTi 


Dummy_5289 = (exp(-0.5d0*vPar*vPar/Ti(r, CTi, kTi, deltaRTi, rp))/sqrt( &
      2.0d0*(3.14159265358979d0*Ti(r, CTi, kTi, deltaRTi, rp))))*n0(r, &
      CN0, kN0, deltaRN0, rp)
return
end function
!........................................

!........................................
pure real(kind=8) function n0derivNormalised(r, kN0, rp, deltaRN0) &
      result(Dummy_3404)

implicit none
real(kind=8), intent(in)  :: r 
real(kind=8), intent(in)  :: kN0 
real(kind=8), intent(in)  :: rp 
real(kind=8), intent(in)  :: deltaRN0 


Dummy_3404 = (-kN0)*(1 - tanh((r - rp)/deltaRN0)**2)
return
end function
!........................................

!........................................
pure real(kind=8) function Te(r, CTe, kTe, deltaRTe, rp)  result( &
      Dummy_8219)

implicit none
real(kind=8), intent(in)  :: r 
real(kind=8), intent(in)  :: CTe 
real(kind=8), intent(in)  :: kTe 
real(kind=8), intent(in)  :: deltaRTe 
real(kind=8), intent(in)  :: rp 


Dummy_8219 = CTe*exp((-kTe)*(deltaRTe*tanh((r - rp)/deltaRTe)))
return
end function
!........................................

end module