module accelerated_advection_steps

use spline_eval_funcs, only: eval_spline_2d_cross
use spline_eval_funcs, only: eval_spline_2d_scalar
use spline_eval_funcs, only: eval_spline_1d_scalar

use initialiser_funcs, only: f_eq
implicit none




contains

!........................................
pure subroutine poloidal_advection_step_expl(f, dt, v, rPts, qPts, nPts, &
      drPhi_0, dthetaPhi_0, drPhi_k, dthetaPhi_k, endPts_k1_q, &
      endPts_k1_r, endPts_k2_q, endPts_k2_r, kts1Phi, kts2Phi, &
      coeffsPhi, deg1Phi, deg2Phi, kts1Pol, kts2Pol, coeffsPol, deg1Pol &
      , deg2Pol, CN0, kN0, deltaRN0, rp, CTi, kTi, deltaRTi, B0, &
      nulBound)

implicit none
real(kind=8), intent(inout)  :: f (0:,0:)
real(kind=8), intent(in)  :: dt 
real(kind=8), intent(in)  :: v 
real(kind=8), intent(in)  :: rPts (0:)
real(kind=8), intent(in)  :: qPts (0:)
integer(kind=4), intent(in)  :: nPts (0:)
real(kind=8), intent(inout)  :: drPhi_0 (0:,0:)
real(kind=8), intent(inout)  :: dthetaPhi_0 (0:,0:)
real(kind=8), intent(inout)  :: drPhi_k (0:,0:)
real(kind=8), intent(inout)  :: dthetaPhi_k (0:,0:)
real(kind=8), intent(inout)  :: endPts_k1_q (0:,0:)
real(kind=8), intent(inout)  :: endPts_k1_r (0:,0:)
real(kind=8), intent(inout)  :: endPts_k2_q (0:,0:)
real(kind=8), intent(inout)  :: endPts_k2_r (0:,0:)
real(kind=8), intent(in)  :: kts1Phi (0:)
real(kind=8), intent(in)  :: kts2Phi (0:)
real(kind=8), intent(in)  :: coeffsPhi (0:,0:)
integer(kind=4), intent(in)  :: deg1Phi 
integer(kind=4), intent(in)  :: deg2Phi 
real(kind=8), intent(in)  :: kts1Pol (0:)
real(kind=8), intent(in)  :: kts2Pol (0:)
real(kind=8), intent(in)  :: coeffsPol (0:,0:)
integer(kind=4), intent(in)  :: deg1Pol 
integer(kind=4), intent(in)  :: deg2Pol 
real(kind=8), intent(in)  :: CN0 
real(kind=8), intent(in)  :: kN0 
real(kind=8), intent(in)  :: deltaRN0 
real(kind=8), intent(in)  :: rp 
real(kind=8), intent(in)  :: CTi 
real(kind=8), intent(in)  :: kTi 
real(kind=8), intent(in)  :: deltaRTi 
real(kind=8), intent(in)  :: B0 
logical(kind=1), intent(in)  :: nulBound 
real(kind=8) :: multFactor  
real(kind=8) :: multFactor_half  
integer(kind=4) :: idx  
real(kind=8) :: rMax  
integer(kind=4) :: i  
integer(kind=4) :: j  

!_______________________CommentBlock_______________________!
!                                                          !
!    Carry out an advection step for the poloidal advection!
!                                                          !
!    Parameters                                            !
!    ----------                                            !
!    f: array_like                                         !
!        The current value of the function at the nodes.   !
!        The result will be stored here                    !
!                                                          !
!    dt: float                                             !
!        Time-step                                         !
!                                                          !
!    phi: Spline2D                                         !
!        Advection parameter d_tf + {phi,f}=0              !
!                                                          !
!    r: float                                              !
!        The parallel velocity coordinate                  !
!                                                          !
!                                                          !
!__________________________________________________________!





multFactor = dt/B0
multFactor_half = 0.5d0*multFactor


call eval_spline_2d_cross(qPts, rPts, kts1Phi, deg1Phi, kts2Phi, deg2Phi &
      , coeffsPhi, drPhi_0, 0, 1)
call eval_spline_2d_cross(qPts, rPts, kts1Phi, deg1Phi, kts2Phi, deg2Phi &
      , coeffsPhi, dthetaPhi_0, 1, 0)


idx = nPts(1) - 1
rMax = rPts(idx)


do i = 0, nPts(0) - 1, 1
  do j = 0, nPts(1) - 1, 1
    drPhi_0(j, i) = drPhi_0(j, i)/rPts(j)
    dthetaPhi_0(j, i) = dthetaPhi_0(j, i)/rPts(j)
    endPts_k1_q(j, i) = multFactor*(-drPhi_0(j, i)) + qPts(i)
    endPts_k1_r(j, i) = multFactor*dthetaPhi_0(j, i) + rPts(j)


    do while (endPts_k1_q(j, i) < 0) 
      endPts_k1_q(j, i) = 2*3.14159265358979d0 + endPts_k1_q(j, i)
    end do
    do while (endPts_k1_q(j, i) > 2*3.14159265358979d0) 
      endPts_k1_q(j, i) = -2*3.14159265358979d0 + endPts_k1_q(j, i)


    end do
    if (.not. (endPts_k1_r(j, i) > rMax .or. endPts_k1_r(j, i) < rPts(0 &
      ))) then
      !Add the new value of phi to the derivatives
      !x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
      !^^^^^^^^^^^^^^^
      drPhi_k(j, i) = eval_spline_2d_scalar(endPts_k1_q(j, i), &
      endPts_k1_r(j, i), kts1Phi, deg1Phi, kts2Phi, deg2Phi, coeffsPhi, &
      0, 1)
      drPhi_k(j, i) = drPhi_k(j, i)/endPts_k1_r(j, i)


      dthetaPhi_k(j, i) = eval_spline_2d_scalar(endPts_k1_q(j, i), &
      endPts_k1_r(j, i), kts1Phi, deg1Phi, kts2Phi, deg2Phi, coeffsPhi, &
      1, 0)
      dthetaPhi_k(j, i) = dthetaPhi_k(j, i)/endPts_k1_r(j, i)
    else
      drPhi_k(j, i) = 0.0d0
      dthetaPhi_k(j, i) = 0.0d0
    end if
    endPts_k2_q(j, i) = modulo(multFactor_half*(-drPhi_0(j, i) - drPhi_k &
      (j, i)) + qPts(i),2*3.14159265358979d0)
    endPts_k2_r(j, i) = multFactor_half*(dthetaPhi_0(j, i) + dthetaPhi_k &
      (j, i)) + rPts(j)


  end do

end do

!Step one of Heun method
!x' = x^n + f(x^n)
!Handle theta boundary conditions
!Step two of Heun method
!x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
!Find value at the determined point
if (nulBound) then
  do i = 0, nPts(0) - 1, 1
    do j = 0, nPts(1) - 1, 1
      if (endPts_k2_r(j, i) < rPts(0)) then
        f(j, i) = 0.0d0
      else if (endPts_k2_r(j, i) > rMax) then
        f(j, i) = 0.0d0
      else
        do while (endPts_k2_q(j, i) > 2*3.14159265358979d0) 
          endPts_k2_q(j, i) = -2*3.14159265358979d0 + endPts_k2_q(j, i)
        end do
        do while (endPts_k2_q(j, i) < 0) 
          endPts_k2_q(j, i) = 2*3.14159265358979d0 + endPts_k2_q(j, i)
        end do
        f(j, i) = eval_spline_2d_scalar(endPts_k2_q(j, i), endPts_k2_r(j &
      , i), kts1Pol, deg1Pol, kts2Pol, deg2Pol, coeffsPol, 0, 0)
      end if
    end do

  end do

else
  do i = 0, nPts(0) - 1, 1
    do j = 0, nPts(1) - 1, 1
      if (endPts_k2_r(j, i) < rPts(0)) then
        f(j, i) = f_eq(rPts(0), v, CN0, kN0, deltaRN0, rp, CTi, kTi, &
      deltaRTi)
      else if (endPts_k2_r(j, i) > rMax) then
        f(j, i) = f_eq(endPts_k2_r(j, i), v, CN0, kN0, deltaRN0, rp, CTi &
      , kTi, deltaRTi)
      else
        do while (endPts_k2_q(j, i) > 2*3.14159265358979d0) 
          endPts_k2_q(j, i) = -2*3.14159265358979d0 + endPts_k2_q(j, i)
        end do
        do while (endPts_k2_q(j, i) < 0) 
          endPts_k2_q(j, i) = 2*3.14159265358979d0 + endPts_k2_q(j, i)
        end do
        f(j, i) = eval_spline_2d_scalar(endPts_k2_q(j, i), endPts_k2_r(j &
      , i), kts1Pol, deg1Pol, kts2Pol, deg2Pol, coeffsPol, 0, 0)
      end if
    end do

  end do

end if
end subroutine
!........................................

!........................................
pure subroutine v_parallel_advection_eval_step(f, vPts, rPos, vMin, vMax &
      , kts, deg, coeffs, CN0, kN0, deltaRN0, rp, CTi, kTi, deltaRTi, &
      bound)

implicit none
real(kind=8), intent(inout)  :: f (0:)
real(kind=8), intent(in)  :: vPts (0:)
real(kind=8), intent(in)  :: rPos 
real(kind=8), intent(in)  :: vMin 
real(kind=8), intent(in)  :: vMax 
real(kind=8), intent(in)  :: kts (0:)
integer(kind=4), intent(in)  :: deg 
real(kind=8), intent(in)  :: coeffs (0:)
real(kind=8), intent(in)  :: CN0 
real(kind=8), intent(in)  :: kN0 
real(kind=8), intent(in)  :: deltaRN0 
real(kind=8), intent(in)  :: rp 
real(kind=8), intent(in)  :: CTi 
real(kind=8), intent(in)  :: kTi 
real(kind=8), intent(in)  :: deltaRTi 
integer(kind=4), intent(in)  :: bound 
real(kind=8) :: vDiff  
integer(kind=4) :: i  
real(kind=8) :: v  

!Find value at the determined point
if (bound == 0 ) then
do i = 0, size(vPts,1) - 1, 1
  v = vPts(i)
  if (v > vMax .or. v < vMin) then
    f(i) = f_eq(rPos, v, CN0, kN0, deltaRN0, rp, CTi, kTi, deltaRTi)
  else
    f(i) = eval_spline_1d_scalar(v, kts, deg, coeffs, 0)
  end if
end do

else if (bound == 1 ) then
do i = 0, size(vPts,1) - 1, 1
  v = vPts(i)
  if (v > vMax .or. v < vMin) then
    f(i) = 0.0d0
  else
    f(i) = eval_spline_1d_scalar(v, kts, deg, coeffs, 0)
  end if
end do

else if (bound == 2 ) then
vDiff = vMax - vMin
do i = 0, size(vPts,1) - 1, 1
  v = vPts(i)
  do while (v < vMin) 
    v = v + vDiff
  end do
  do while (v > vMax) 
    v = v - vDiff
  end do
  f(i) = eval_spline_1d_scalar(v, kts, deg, coeffs, 0)




end do

end if
end subroutine
!........................................

!........................................
pure subroutine get_lagrange_vals(i, nz, shifts, vals, qVals, &
      thetaShifts, kts, deg, coeffs)

implicit none
integer(kind=4), intent(in)  :: i 
integer(kind=4), intent(in)  :: nz 
integer(kind=4), intent(in)  :: shifts (0:)
real(kind=8), intent(inout)  :: vals (0:,0:,0:)
real(kind=8), intent(in)  :: qVals (0:)
real(kind=8), intent(in)  :: thetaShifts (0:)
real(kind=8), intent(in)  :: kts (0:)
integer(kind=4), intent(in)  :: deg 
real(kind=8), intent(in)  :: coeffs (0:)
integer(kind=4) :: j  
integer(kind=4) :: s  
integer(kind=4) :: k  
real(kind=8) :: q  
real(kind=8) :: new_q  


do j = 0, size(shifts,1) - 1, 1
s = shifts(j)
do k = 0, size(qVals,1) - 1, 1
q = qVals(k)
new_q = q + thetaShifts(j)
do while (new_q < 0) 
  new_q = 2*3.14159265358979d0 + new_q
end do
do while (new_q > 2*3.14159265358979d0) 
  new_q = -2*3.14159265358979d0 + new_q
end do
vals(j, k, modulo(i - s,nz)) = eval_spline_1d_scalar(new_q, kts, deg, &
      coeffs, 0)


end do

end do

end subroutine
!........................................

!........................................
pure subroutine flux_advection(nq, nr, f, coeffs, vals) 

implicit none
integer(kind=4), intent(in)  :: nq 
integer(kind=4), intent(in)  :: nr 
real(kind=8), intent(inout)  :: f (0:,0:)
real(kind=8), intent(in)  :: coeffs (0:)
real(kind=8), intent(in)  :: vals (0:,0:,0:)
integer(kind=4) :: j  
integer(kind=4) :: i  
integer(kind=4) :: k  

do j = 0, nq - 1, 1
do i = 0, nr - 1, 1
f(i, j) = coeffs(0)*vals(0, j, i)
do k = 1, size(coeffs,1) - 1, 1
f(i, j) = coeffs(k)*vals(k, j, i) + f(i, j)


end do

end do

end do

end subroutine
!........................................

!........................................
pure subroutine poloidal_advection_step_impl(f, dt, v, rPts, qPts, nPts, &
      drPhi_0, dthetaPhi_0, drPhi_k, dthetaPhi_k, endPts_k1_q, &
      endPts_k1_r, endPts_k2_q, endPts_k2_r, kts1Phi, kts2Phi, &
      coeffsPhi, deg1Phi, deg2Phi, kts1Pol, kts2Pol, coeffsPol, deg1Pol &
      , deg2Pol, CN0, kN0, deltaRN0, rp, CTi, kTi, deltaRTi, B0, tol, &
      nulBound)

implicit none
real(kind=8), intent(inout)  :: f (0:,0:)
real(kind=8), intent(in)  :: dt 
real(kind=8), intent(in)  :: v 
real(kind=8), intent(in)  :: rPts (0:)
real(kind=8), intent(in)  :: qPts (0:)
integer(kind=4), intent(in)  :: nPts (0:)
real(kind=8), intent(inout)  :: drPhi_0 (0:,0:)
real(kind=8), intent(inout)  :: dthetaPhi_0 (0:,0:)
real(kind=8), intent(inout)  :: drPhi_k (0:,0:)
real(kind=8), intent(inout)  :: dthetaPhi_k (0:,0:)
real(kind=8), intent(inout)  :: endPts_k1_q (0:,0:)
real(kind=8), intent(inout)  :: endPts_k1_r (0:,0:)
real(kind=8), intent(inout)  :: endPts_k2_q (0:,0:)
real(kind=8), intent(inout)  :: endPts_k2_r (0:,0:)
real(kind=8), intent(in)  :: kts1Phi (0:)
real(kind=8), intent(in)  :: kts2Phi (0:)
real(kind=8), intent(in)  :: coeffsPhi (0:,0:)
integer(kind=4), intent(in)  :: deg1Phi 
integer(kind=4), intent(in)  :: deg2Phi 
real(kind=8), intent(in)  :: kts1Pol (0:)
real(kind=8), intent(in)  :: kts2Pol (0:)
real(kind=8), intent(in)  :: coeffsPol (0:,0:)
integer(kind=4), intent(in)  :: deg1Pol 
integer(kind=4), intent(in)  :: deg2Pol 
real(kind=8), intent(in)  :: CN0 
real(kind=8), intent(in)  :: kN0 
real(kind=8), intent(in)  :: deltaRN0 
real(kind=8), intent(in)  :: rp 
real(kind=8), intent(in)  :: CTi 
real(kind=8), intent(in)  :: kTi 
real(kind=8), intent(in)  :: deltaRTi 
real(kind=8), intent(in)  :: B0 
real(kind=8), intent(in)  :: tol 
logical(kind=1), intent(in)  :: nulBound 
real(kind=8) :: multFactor  
integer(kind=4) :: idx  
real(kind=8) :: rMax  
real(kind=8) :: norm  
integer(kind=4) :: i  
integer(kind=4) :: j  
real(kind=8) :: diff  

!_______________________CommentBlock_______________________!
!                                                          !
!    Carry out an advection step for the poloidal advection!
!                                                          !
!    Parameters                                            !
!    ----------                                            !
!    f: array_like                                         !
!        The current value of the function at the nodes.   !
!        The result will be stored here                    !
!                                                          !
!    dt: float                                             !
!        Time-step                                         !
!                                                          !
!    phi: Spline2D                                         !
!        Advection parameter d_tf + {phi,f}=0              !
!                                                          !
!    r: float                                              !
!        The parallel velocity coordinate                  !
!                                                          !
!                                                          !
!__________________________________________________________!



multFactor = dt/B0


call eval_spline_2d_cross(qPts, rPts, kts1Phi, deg1Phi, kts2Phi, deg2Phi &
      , coeffsPhi, drPhi_0, 0, 1)
call eval_spline_2d_cross(qPts, rPts, kts1Phi, deg1Phi, kts2Phi, deg2Phi &
      , coeffsPhi, dthetaPhi_0, 1, 0)


idx = nPts(1) - 1
rMax = rPts(idx)


do i = 0, nPts(0) - 1, 1
do j = 0, nPts(1) - 1, 1
drPhi_0(j, i) = drPhi_0(j, i)/rPts(j)
dthetaPhi_0(j, i) = dthetaPhi_0(j, i)/rPts(j)
endPts_k1_q(j, i) = multFactor*(-drPhi_0(j, i)) + qPts(i)
endPts_k1_r(j, i) = multFactor*dthetaPhi_0(j, i) + rPts(j)


end do

end do

!Step one of Heun method
!x' = x^n + f(x^n)
multFactor = 0.5d0*multFactor


norm = tol + 1
do while (norm > tol) 
norm = 0.0d0
do i = 0, nPts(0) - 1, 1
do j = 0, nPts(1) - 1, 1
!Handle theta boundary conditions
do while (endPts_k1_q(j, i) < 0) 
endPts_k1_q(j, i) = 2*3.14159265358979d0 + endPts_k1_q(j, i)
end do
do while (endPts_k1_q(j, i) > 2*3.14159265358979d0) 
endPts_k1_q(j, i) = -2*3.14159265358979d0 + endPts_k1_q(j, i)


end do
if (.not. (endPts_k1_r(j, i) > rMax .or. endPts_k1_r(j, i) < rPts(0))) &
      then
!Add the new value of phi to the derivatives
!x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
!^^^^^^^^^^^^^^^
drPhi_k(j, i) = eval_spline_2d_scalar(endPts_k1_q(j, i), endPts_k1_r(j, &
      i), kts1Phi, deg1Phi, kts2Phi, deg2Phi, coeffsPhi, 0, 1)
drPhi_k(j, i) = drPhi_k(j, i)/endPts_k1_r(j, i)
dthetaPhi_k(j, i) = eval_spline_2d_scalar(endPts_k1_q(j, i), endPts_k1_r &
      (j, i), kts1Phi, deg1Phi, kts2Phi, deg2Phi, coeffsPhi, 1, 0)
dthetaPhi_k(j, i) = dthetaPhi_k(j, i)/endPts_k1_r(j, i)
else
drPhi_k(j, i) = 0.0d0
dthetaPhi_k(j, i) = 0.0d0


!Step two of Heun method
!x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
!Clipping is one method of avoiding infinite loops due to
!boundary conditions
!Using the splines to extrapolate is not sufficient
end if
endPts_k2_q(j, i) = modulo(multFactor*(-drPhi_0(j, i) - drPhi_k(j, i)) + &
      qPts(i),2*3.14159265358979d0)
endPts_k2_r(j, i) = multFactor*(dthetaPhi_0(j, i) + dthetaPhi_k(j, i)) + &
      rPts(j)


if (endPts_k2_r(j, i) < rPts(0)) then
endPts_k2_r(j, i) = rPts(0)
else if (endPts_k2_r(j, i) > rMax) then
endPts_k2_r(j, i) = rMax


end if
diff = Abs(-endPts_k1_q(j, i) + endPts_k2_q(j, i))
if (diff > norm) then
norm = diff
end if
diff = Abs(-endPts_k1_r(j, i) + endPts_k2_r(j, i))
if (diff > norm) then
norm = diff
end if
endPts_k1_q(j, i) = endPts_k2_q(j, i)
endPts_k1_r(j, i) = endPts_k2_r(j, i)


!Find value at the determined point
end do

end do

end do
if (nulBound) then
do i = 0, nPts(0) - 1, 1
do j = 0, nPts(1) - 1, 1
if (endPts_k2_r(j, i) < rPts(0)) then
f(j, i) = 0.0d0
else if (endPts_k2_r(j, i) > rMax) then
f(j, i) = 0.0d0
else
do while (endPts_k2_q(j, i) > 2*3.14159265358979d0) 
  endPts_k2_q(j, i) = -2*3.14159265358979d0 + endPts_k2_q(j, i)
end do
do while (endPts_k2_q(j, i) < 0) 
  endPts_k2_q(j, i) = 2*3.14159265358979d0 + endPts_k2_q(j, i)
end do
f(j, i) = eval_spline_2d_scalar(endPts_k2_q(j, i), endPts_k2_r(j, i), &
      kts1Pol, deg1Pol, kts2Pol, deg2Pol, coeffsPol, 0, 0)
end if
end do

end do

else
do i = 0, nPts(0) - 1, 1
do j = 0, nPts(1) - 1, 1
if (endPts_k2_r(j, i) < rPts(0)) then
f(j, i) = f_eq(rPts(0), v, CN0, kN0, deltaRN0, rp, CTi, kTi, deltaRTi)
else if (endPts_k2_r(j, i) > rMax) then
f(j, i) = f_eq(endPts_k2_r(j, i), v, CN0, kN0, deltaRN0, rp, CTi, kTi, &
      deltaRTi)
else
do while (endPts_k2_q(j, i) > 2*3.14159265358979d0) 
  endPts_k2_q(j, i) = -2*3.14159265358979d0 + endPts_k2_q(j, i)
end do
do while (endPts_k2_q(j, i) < 0) 
  endPts_k2_q(j, i) = 2*3.14159265358979d0 + endPts_k2_q(j, i)
end do
f(j, i) = eval_spline_2d_scalar(endPts_k2_q(j, i), endPts_k2_r(j, i), &
      kts1Pol, deg1Pol, kts2Pol, deg2Pol, coeffsPol, 0, 0)
end if
end do

end do

end if
end subroutine
!........................................

end module