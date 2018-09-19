module mod_initialiser_funcs

implicit none



contains

! ........................................
real(kind=8) function n0(r, CN0, kN0, deltaRN0, rp)  result(result_8960)

implicit none
real(kind=8), intent(in)  :: r
real(kind=8), intent(in)  :: CN0
real(kind=8), intent(in)  :: kN0
real(kind=8), intent(in)  :: deltaRN0
real(kind=8), intent(in)  :: rp


result_8960 = CN0*exp((-kN0)*(deltaRN0*tanh((r - rp)/deltaRN0)))
return


end function
! ........................................

! ........................................
real(kind=8) function Ti(r, CTi, kTi, deltaRTi, rp)  result(result_5939)

implicit none
real(kind=8), intent(in)  :: r
real(kind=8), intent(in)  :: CTi
real(kind=8), intent(in)  :: kTi
real(kind=8), intent(in)  :: deltaRTi
real(kind=8), intent(in)  :: rp


result_5939 = CTi*exp((-kTi)*(deltaRTi*tanh((r - rp)/deltaRTi)))
return


end function
! ........................................

! ........................................
real(kind=8) function perturbation(r, theta, z, m, n, rp, deltaR, R0) &
      result(result_9949)

implicit none
real(kind=8), intent(in)  :: r
real(kind=8), intent(in)  :: theta
real(kind=8), intent(in)  :: z
integer(kind=4), intent(in)  :: m
integer(kind=4), intent(in)  :: n
real(kind=8), intent(in)  :: rp
real(kind=8), intent(in)  :: deltaR
real(kind=8), intent(in)  :: R0


result_9949 = exp((-(r - rp)**2)/deltaR)*cos(m*theta + n*(z/R0))
return


end function
! ........................................

! ........................................
real(kind=8) function fEq(r, vPar, CN0, kN0, deltaRN0, rp, CTi, kTi, &
      deltaRTi)  result(result_6242)

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


result_6242 = ((0.707106781186547d0/sqrt(3.141592653589793*Ti(r, CTi, &
      kTi, deltaRTi, rp)))*exp(-0.5d0*vPar*vPar/Ti(r, CTi, kTi, &
      deltaRTi, rp)))*n0(r, CN0, kN0, deltaRN0, rp)
return


end function
! ........................................

! ........................................
real(kind=8) function n0derivNormalised(r, kN0, rp, deltaRN0)  result( &
      result_9373)

implicit none
real(kind=8), intent(in)  :: r
real(kind=8), intent(in)  :: kN0
real(kind=8), intent(in)  :: rp
real(kind=8), intent(in)  :: deltaRN0


result_9373 = (-kN0)*(-tanh((r - rp)/deltaRN0)**2 + 1)
return


end function
! ........................................

! ........................................
real(kind=8) function Te(r, CTe, kTe, deltaRTe, rp)  result(result_0509)

implicit none
real(kind=8), intent(in)  :: r
real(kind=8), intent(in)  :: CTe
real(kind=8), intent(in)  :: kTe
real(kind=8), intent(in)  :: deltaRTe
real(kind=8), intent(in)  :: rp


result_0509 = CTe*exp((-kTe)*(deltaRTe*tanh((r - rp)/deltaRTe)))
return


end function
! ........................................

end module