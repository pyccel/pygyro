module accelerated_advection_steps

use initialiser_funcs, only: f_eq
use spline_eval_funcs, only: eval_spline_2d_cross
use spline_eval_funcs, only: eval_spline_2d_scalar
use spline_eval_funcs, only: eval_spline_1d_scalar

implicit none

contains

!........................................
pure  subroutine poloidal_advection_step_expl(f, dt, v, rPts, qPts, nPts &
      , drPhi_0, dthetaPhi_0, drPhi_k, dthetaPhi_k, endPts_k1_q, &
      endPts_k1_r, endPts_k2_q, endPts_k2_r, kts1Phi, kts2Phi, &
      coeffsPhi, deg1Phi, deg2Phi, kts1Pol, kts2Pol, coeffsPol, deg1Pol &
      , deg2Pol, CN0, kN0, deltaRN0, rp, CTi, kTi, deltaRTi, B0, &
      nulBound)

implicit none

real(kind=8), intent(inout) :: f(0:,0:)
real(kind=8), value :: dt
real(kind=8), value :: v
real(kind=8), intent(in) :: rPts(0:)
real(kind=8), intent(in) :: qPts(0:)
integer(kind=8), intent(in) :: nPts(0:)
real(kind=8), intent(inout) :: drPhi_0(0:,0:)
real(kind=8), intent(inout) :: dthetaPhi_0(0:,0:)
real(kind=8), intent(inout) :: drPhi_k(0:,0:)
real(kind=8), intent(inout) :: dthetaPhi_k(0:,0:)
real(kind=8), intent(inout) :: endPts_k1_q(0:,0:)
real(kind=8), intent(inout) :: endPts_k1_r(0:,0:)
real(kind=8), intent(inout) :: endPts_k2_q(0:,0:)
real(kind=8), intent(inout) :: endPts_k2_r(0:,0:)
real(kind=8), intent(in) :: kts1Phi(0:)
real(kind=8), intent(in) :: kts2Phi(0:)
real(kind=8), intent(in) :: coeffsPhi(0:,0:)
integer(kind=8), value :: deg1Phi
integer(kind=8), value :: deg2Phi
real(kind=8), intent(in) :: kts1Pol(0:)
real(kind=8), intent(in) :: kts2Pol(0:)
real(kind=8), intent(in) :: coeffsPol(0:,0:)
integer(kind=8), value :: deg1Pol
integer(kind=8), value :: deg2Pol
real(kind=8), value :: CN0
real(kind=8), value :: kN0
real(kind=8), value :: deltaRN0
real(kind=8), value :: rp
real(kind=8), value :: CTi
real(kind=8), value :: kTi
real(kind=8), value :: deltaRTi
real(kind=8), value :: B0
logical(kind=4), value :: nulBound
real(kind=8) :: multFactor
real(kind=8) :: multFactor_half
integer(kind=8) :: idx
real(kind=8) :: rMax
integer(kind=8) :: i
integer(kind=8) :: j

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
multFactor = dt / B0
multFactor_half = 0.5d0 * multFactor
call eval_spline_2d_cross(qPts, rPts, kts1Phi, deg1Phi, kts2Phi, deg2Phi &
      , coeffsPhi, drPhi_0, 0_8, 1_8)
call eval_spline_2d_cross(qPts, rPts, kts1Phi, deg1Phi, kts2Phi, deg2Phi &
      , coeffsPhi, dthetaPhi_0, 1_8, 0_8)
idx = nPts(1_8) - 1_8
rMax = rPts(idx)
do i = 0_8, nPts(0_8)-1_8, 1_8
  do j = 0_8, nPts(1_8)-1_8, 1_8
    !Step one of Heun method
    !x' = x^n + f(x^n)
    drPhi_0(j, i) = drPhi_0(j, i) / rPts(j)
    dthetaPhi_0(j, i) = dthetaPhi_0(j, i) / rPts(j)
    endPts_k1_q(j, i) = qPts(i) - drPhi_0(j, i) * multFactor
    endPts_k1_r(j, i) = rPts(j) + dthetaPhi_0(j, i) * multFactor
    !Handle theta boundary conditions
    do while (endPts_k1_q(j, i) < 0_8)
      endPts_k1_q(j, i) = endPts_k1_q(j, i) + 2_8 * 3.14159265358979d0
    end do
    do while (endPts_k1_q(j, i) > 2_8 * 3.14159265358979d0)
      endPts_k1_q(j, i) = endPts_k1_q(j, i) - 2_8 * 3.14159265358979d0
    end do
    if (.not. (endPts_k1_r(j, i) < rPts(0_8) .or. endPts_k1_r(j, i) > &
      rMax)) then
      !Add the new value of phi to the derivatives
      !x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
      !^^^^^^^^^^^^^^^
      drPhi_k(j, i) = eval_spline_2d_scalar(endPts_k1_q(j, i), &
      endPts_k1_r(j, i), kts1Phi, deg1Phi, kts2Phi, deg2Phi, coeffsPhi, &
      0_8, 1_8)
      drPhi_k(j, i) = drPhi_k(j, i) / endPts_k1_r(j, i)
      dthetaPhi_k(j, i) = eval_spline_2d_scalar(endPts_k1_q(j, i), &
      endPts_k1_r(j, i), kts1Phi, deg1Phi, kts2Phi, deg2Phi, coeffsPhi, &
      1_8, 0_8)
      dthetaPhi_k(j, i) = dthetaPhi_k(j, i) / endPts_k1_r(j, i)
    else
      drPhi_k(j, i) = 0.0d0
      dthetaPhi_k(j, i) = 0.0d0
    end if
    !Step two of Heun method
    !x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
    endPts_k2_q(j, i) = MODULO((qPts(i) - (drPhi_0(j, i) + drPhi_k(j, i &
      )) * multFactor_half),(2_8 * 3.14159265358979d0))
    endPts_k2_r(j, i) = rPts(j) + (dthetaPhi_0(j, i) + dthetaPhi_k(j, i &
      )) * multFactor_half
  end do
  !Find value at the determined point
  !Find value at the determined point
  !Step one of Heun method
  !x' = x^n + f(x^n)
  !Handle theta boundary conditions
  !Add the new value of phi to the derivatives
  !x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
  !^^^^^^^^^^^^^^^
  !Step two of Heun method
  !x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
  !Clipping is one method of avoiding infinite loops due to
  !boundary conditions
  !Using the splines to extrapolate is not sufficient
  !Find value at the determined point
end do
if (nulBound) then
  do i = 0_8, nPts(0_8)-1_8, 1_8
    do j = 0_8, nPts(1_8)-1_8, 1_8
      if (endPts_k2_r(j, i) < rPts(0_8)) then
        f(j, i) = 0.0d0
      else if (endPts_k2_r(j, i) > rMax) then
        f(j, i) = 0.0d0
      else
        do while (endPts_k2_q(j, i) > 2_8 * 3.14159265358979d0)
          endPts_k2_q(j, i) = endPts_k2_q(j, i) - 2_8 * &
      3.14159265358979d0
        end do
        do while (endPts_k2_q(j, i) < 0_8)
          endPts_k2_q(j, i) = endPts_k2_q(j, i) + 2_8 * &
      3.14159265358979d0
        end do
        f(j, i) = eval_spline_2d_scalar(endPts_k2_q(j, i), endPts_k2_r(j &
      , i), kts1Pol, deg1Pol, kts2Pol, deg2Pol, coeffsPol, 0_8, 0_8)
      end if
    end do
  end do
else
  do i = 0_8, nPts(0_8)-1_8, 1_8
    do j = 0_8, nPts(1_8)-1_8, 1_8
      if (endPts_k2_r(j, i) < rPts(0_8)) then
        f(j, i) = f_eq(rPts(0_8), v, CN0, kN0, deltaRN0, rp, CTi, kTi, &
      deltaRTi)
      else if (endPts_k2_r(j, i) > rMax) then
        f(j, i) = f_eq(endPts_k2_r(j, i), v, CN0, kN0, deltaRN0, rp, CTi &
      , kTi, deltaRTi)
      else
        do while (endPts_k2_q(j, i) > 2_8 * 3.14159265358979d0)
          endPts_k2_q(j, i) = endPts_k2_q(j, i) - 2_8 * &
      3.14159265358979d0
        end do
        do while (endPts_k2_q(j, i) < 0_8)
          endPts_k2_q(j, i) = endPts_k2_q(j, i) + 2_8 * &
      3.14159265358979d0
        end do
        f(j, i) = eval_spline_2d_scalar(endPts_k2_q(j, i), endPts_k2_r(j &
      , i), kts1Pol, deg1Pol, kts2Pol, deg2Pol, coeffsPol, 0_8, 0_8)
      end if
    end do
  end do
end if

end subroutine poloidal_advection_step_expl
!........................................

!........................................
pure  subroutine v_parallel_advection_eval_step(f, vPts, rPos, vMin, &
      vMax, kts, deg, coeffs, CN0, kN0, deltaRN0, rp, CTi, kTi, &
      deltaRTi, bound)

implicit none

real(kind=8), intent(inout) :: f(0:)
real(kind=8), intent(in) :: vPts(0:)
real(kind=8), value :: rPos
real(kind=8), value :: vMin
real(kind=8), value :: vMax
real(kind=8), intent(in) :: kts(0:)
integer(kind=8), value :: deg
real(kind=8), intent(in) :: coeffs(0:)
real(kind=8), value :: CN0
real(kind=8), value :: kN0
real(kind=8), value :: deltaRN0
real(kind=8), value :: rp
real(kind=8), value :: CTi
real(kind=8), value :: kTi
real(kind=8), value :: deltaRTi
integer(kind=8), value :: bound
real(kind=8) :: vDiff
integer(kind=8) :: i
real(kind=8) :: v

if (bound == 0_8) then
do i = 0_8, size(vPts,1)-1_8, 1_8
  v = vPts(i)
  if (v < vMin .or. v > vMax) then
    f(i) = f_eq(rPos, v, CN0, kN0, deltaRN0, rp, CTi, kTi, deltaRTi)
  else
    f(i) = eval_spline_1d_scalar(v, kts, deg, coeffs, 0_8)
  end if
end do
else if (bound == 1_8) then
do i = 0_8, size(vPts,1)-1_8, 1_8
  v = vPts(i)
  if (v < vMin .or. v > vMax) then
    f(i) = 0.0d0
  else
    f(i) = eval_spline_1d_scalar(v, kts, deg, coeffs, 0_8)
  end if
end do
else if (bound == 2_8) then
vDiff = vMax - vMin
do i = 0_8, size(vPts,1)-1_8, 1_8
  v = vPts(i)
  do while (v < vMin)
    v = v + vDiff
  end do
  do while (v > vMax)
    v = v - vDiff
  end do
  f(i) = eval_spline_1d_scalar(v, kts, deg, coeffs, 0_8)
end do
end if

end subroutine v_parallel_advection_eval_step
!........................................

!........................................
pure  subroutine get_lagrange_vals(i, nz, shifts, vals, qVals, &
      thetaShifts, kts, deg, coeffs)

implicit none

integer(kind=8), value :: i
integer(kind=8), value :: nz
integer(kind=8), intent(in) :: shifts(0:)
real(kind=8), intent(inout) :: vals(0:,0:,0:)
real(kind=8), intent(in) :: qVals(0:)
real(kind=8), intent(in) :: thetaShifts(0:)
real(kind=8), intent(in) :: kts(0:)
integer(kind=8), value :: deg
real(kind=8), intent(in) :: coeffs(0:)
integer(kind=8) :: j
integer(kind=8) :: s
integer(kind=8) :: k
real(kind=8) :: q
real(kind=8) :: new_q

do j = 0_8, size(shifts,1)-1_8, 1_8
s = shifts(j)
do k = 0_8, size(qVals,1)-1_8, 1_8
q = qVals(k)
new_q = q + thetaShifts(j)
do while (new_q < 0_8)
  new_q = new_q + 2_8 * 3.14159265358979d0
end do
do while (new_q > 2_8 * 3.14159265358979d0)
  new_q = new_q - 2_8 * 3.14159265358979d0
end do
vals(j, k, MODULO((i - s),nz)) = eval_spline_1d_scalar(new_q, kts, deg, &
      coeffs, 0_8)
end do
end do

end subroutine get_lagrange_vals
!........................................

!........................................
pure  subroutine flux_advection(nq, nr, f, coeffs, vals) 

implicit none

integer(kind=8), value :: nq
integer(kind=8), value :: nr
real(kind=8), intent(inout) :: f(0:,0:)
real(kind=8), intent(in) :: coeffs(0:)
real(kind=8), intent(in) :: vals(0:,0:,0:)
integer(kind=8) :: j
integer(kind=8) :: i
integer(kind=8) :: k

do j = 0_8, nq-1_8, 1_8
do i = 0_8, nr-1_8, 1_8
f(i, j) = coeffs(0_8) * vals(0_8, j, i)
do k = 1_8, size(coeffs,1)-1_8, 1_8
f(i, j) = f(i, j) + coeffs(k) * vals(k, j, i)
end do
end do
end do

end subroutine flux_advection
!........................................

!........................................
pure  subroutine poloidal_advection_step_impl(f, dt, v, rPts, qPts, nPts &
      , drPhi_0, dthetaPhi_0, drPhi_k, dthetaPhi_k, endPts_k1_q, &
      endPts_k1_r, endPts_k2_q, endPts_k2_r, kts1Phi, kts2Phi, &
      coeffsPhi, deg1Phi, deg2Phi, kts1Pol, kts2Pol, coeffsPol, deg1Pol &
      , deg2Pol, CN0, kN0, deltaRN0, rp, CTi, kTi, deltaRTi, B0, tol, &
      nulBound)

implicit none

real(kind=8), intent(inout) :: f(0:,0:)
real(kind=8), value :: dt
real(kind=8), value :: v
real(kind=8), intent(in) :: rPts(0:)
real(kind=8), intent(in) :: qPts(0:)
integer(kind=8), intent(in) :: nPts(0:)
real(kind=8), intent(inout) :: drPhi_0(0:,0:)
real(kind=8), intent(inout) :: dthetaPhi_0(0:,0:)
real(kind=8), intent(inout) :: drPhi_k(0:,0:)
real(kind=8), intent(inout) :: dthetaPhi_k(0:,0:)
real(kind=8), intent(inout) :: endPts_k1_q(0:,0:)
real(kind=8), intent(inout) :: endPts_k1_r(0:,0:)
real(kind=8), intent(inout) :: endPts_k2_q(0:,0:)
real(kind=8), intent(inout) :: endPts_k2_r(0:,0:)
real(kind=8), intent(in) :: kts1Phi(0:)
real(kind=8), intent(in) :: kts2Phi(0:)
real(kind=8), intent(in) :: coeffsPhi(0:,0:)
integer(kind=8), value :: deg1Phi
integer(kind=8), value :: deg2Phi
real(kind=8), intent(in) :: kts1Pol(0:)
real(kind=8), intent(in) :: kts2Pol(0:)
real(kind=8), intent(in) :: coeffsPol(0:,0:)
integer(kind=8), value :: deg1Pol
integer(kind=8), value :: deg2Pol
real(kind=8), value :: CN0
real(kind=8), value :: kN0
real(kind=8), value :: deltaRN0
real(kind=8), value :: rp
real(kind=8), value :: CTi
real(kind=8), value :: kTi
real(kind=8), value :: deltaRTi
real(kind=8), value :: B0
real(kind=8), value :: tol
logical(kind=4), value :: nulBound
real(kind=8) :: multFactor
integer(kind=8) :: idx
real(kind=8) :: rMax
real(kind=8) :: norm
integer(kind=8) :: i
integer(kind=8) :: j
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
multFactor = dt / B0
call eval_spline_2d_cross(qPts, rPts, kts1Phi, deg1Phi, kts2Phi, deg2Phi &
      , coeffsPhi, drPhi_0, 0_8, 1_8)
call eval_spline_2d_cross(qPts, rPts, kts1Phi, deg1Phi, kts2Phi, deg2Phi &
      , coeffsPhi, dthetaPhi_0, 1_8, 0_8)
idx = nPts(1_8) - 1_8
rMax = rPts(idx)
do i = 0_8, nPts(0_8)-1_8, 1_8
do j = 0_8, nPts(1_8)-1_8, 1_8
drPhi_0(j, i) = drPhi_0(j, i) / rPts(j)
dthetaPhi_0(j, i) = dthetaPhi_0(j, i) / rPts(j)
endPts_k1_q(j, i) = qPts(i) - drPhi_0(j, i) * multFactor
endPts_k1_r(j, i) = rPts(j) + dthetaPhi_0(j, i) * multFactor
end do
end do
multFactor = multFactor * 0.5d0
norm = tol + 1_8
do while (norm > tol)
norm = 0.0d0
do i = 0_8, nPts(0_8)-1_8, 1_8
do j = 0_8, nPts(1_8)-1_8, 1_8
do while (endPts_k1_q(j, i) < 0_8)
endPts_k1_q(j, i) = endPts_k1_q(j, i) + 2_8 * 3.14159265358979d0
end do
do while (endPts_k1_q(j, i) > 2_8 * 3.14159265358979d0)
endPts_k1_q(j, i) = endPts_k1_q(j, i) - 2_8 * 3.14159265358979d0
end do
if (.not. (endPts_k1_r(j, i) < rPts(0_8) .or. endPts_k1_r(j, i) > rMax &
      )) then
drPhi_k(j, i) = eval_spline_2d_scalar(endPts_k1_q(j, i), endPts_k1_r(j, &
      i), kts1Phi, deg1Phi, kts2Phi, deg2Phi, coeffsPhi, 0_8, 1_8)
drPhi_k(j, i) = drPhi_k(j, i) / endPts_k1_r(j, i)
dthetaPhi_k(j, i) = eval_spline_2d_scalar(endPts_k1_q(j, i), endPts_k1_r &
      (j, i), kts1Phi, deg1Phi, kts2Phi, deg2Phi, coeffsPhi, 1_8, 0_8)
dthetaPhi_k(j, i) = dthetaPhi_k(j, i) / endPts_k1_r(j, i)
else
drPhi_k(j, i) = 0.0d0
dthetaPhi_k(j, i) = 0.0d0
end if
endPts_k2_q(j, i) = MODULO((qPts(i) - (drPhi_0(j, i) + drPhi_k(j, i)) * &
      multFactor),(2_8 * 3.14159265358979d0))
endPts_k2_r(j, i) = rPts(j) + (dthetaPhi_0(j, i) + dthetaPhi_k(j, i)) * &
      multFactor
if (endPts_k2_r(j, i) < rPts(0_8)) then
endPts_k2_r(j, i) = rPts(0_8)
else if (endPts_k2_r(j, i) > rMax) then
endPts_k2_r(j, i) = rMax
end if
diff = abs(endPts_k2_q(j, i) - endPts_k1_q(j, i))
if (diff > norm) then
norm = diff
end if
diff = abs(endPts_k2_r(j, i) - endPts_k1_r(j, i))
if (diff > norm) then
norm = diff
end if
endPts_k1_q(j, i) = endPts_k2_q(j, i)
endPts_k1_r(j, i) = endPts_k2_r(j, i)
end do
end do
end do
if (nulBound) then
do i = 0_8, nPts(0_8)-1_8, 1_8
do j = 0_8, nPts(1_8)-1_8, 1_8
if (endPts_k2_r(j, i) < rPts(0_8)) then
f(j, i) = 0.0d0
else if (endPts_k2_r(j, i) > rMax) then
f(j, i) = 0.0d0
else
do while (endPts_k2_q(j, i) > 2_8 * 3.14159265358979d0)
  endPts_k2_q(j, i) = endPts_k2_q(j, i) - 2_8 * 3.14159265358979d0
end do
do while (endPts_k2_q(j, i) < 0_8)
  endPts_k2_q(j, i) = endPts_k2_q(j, i) + 2_8 * 3.14159265358979d0
end do
f(j, i) = eval_spline_2d_scalar(endPts_k2_q(j, i), endPts_k2_r(j, i), &
      kts1Pol, deg1Pol, kts2Pol, deg2Pol, coeffsPol, 0_8, 0_8)
end if
end do
end do
else
do i = 0_8, nPts(0_8)-1_8, 1_8
do j = 0_8, nPts(1_8)-1_8, 1_8
if (endPts_k2_r(j, i) < rPts(0_8)) then
f(j, i) = f_eq(rPts(0_8), v, CN0, kN0, deltaRN0, rp, CTi, kTi, deltaRTi)
else if (endPts_k2_r(j, i) > rMax) then
f(j, i) = f_eq(endPts_k2_r(j, i), v, CN0, kN0, deltaRN0, rp, CTi, kTi, &
      deltaRTi)
else
do while (endPts_k2_q(j, i) > 2_8 * 3.14159265358979d0)
  endPts_k2_q(j, i) = endPts_k2_q(j, i) - 2_8 * 3.14159265358979d0
end do
do while (endPts_k2_q(j, i) < 0_8)
  endPts_k2_q(j, i) = endPts_k2_q(j, i) + 2_8 * 3.14159265358979d0
end do
f(j, i) = eval_spline_2d_scalar(endPts_k2_q(j, i), endPts_k2_r(j, i), &
      kts1Pol, deg1Pol, kts2Pol, deg2Pol, coeffsPol, 0_8, 0_8)
end if
end do
end do
end if

end subroutine poloidal_advection_step_impl
!........................................

end module accelerated_advection_steps
