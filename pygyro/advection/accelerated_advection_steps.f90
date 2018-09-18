module mod_pygyro_advection_accelerated_advection_steps

use mod_initialiser_funcs, only: fEq

use mod_spline_eval_funcs, only: eval_spline_2d_cross
use mod_spline_eval_funcs, only: eval_spline_2d_scalar
implicit none



contains

! ........................................
subroutine poloidal_advection_step_expl(n0_f, n1_f, f, dt, v, n0_rPts, &
      rPts, n0_qPts, qPts, n0_nPts, nPts, n0_drPhi_0, n1_drPhi_0, &
      drPhi_0, n0_dthetaPhi_0, n1_dthetaPhi_0, dthetaPhi_0, n0_drPhi_k, &
      n1_drPhi_k, drPhi_k, n0_dthetaPhi_k, n1_dthetaPhi_k, dthetaPhi_k, &
      n0_endPts_k1_q, n1_endPts_k1_q, endPts_k1_q, n0_endPts_k1_r, &
      n1_endPts_k1_r, endPts_k1_r, n0_endPts_k2_q, n1_endPts_k2_q, &
      endPts_k2_q, n0_endPts_k2_r, n1_endPts_k2_r, endPts_k2_r, &
      n0_kts1Phi, kts1Phi, n0_kts2Phi, kts2Phi, n0_coeffsPhi, &
      n1_coeffsPhi, coeffsPhi, deg1Phi, deg2Phi, n0_kts1Pol, kts1Pol, &
      n0_kts2Pol, kts2Pol, n0_coeffsPol, n1_coeffsPol, coeffsPol, &
      deg1Pol, deg2Pol, CN0, kN0, deltaRN0, rp, CTi, kTi, deltaRTi, B0, &
      nulBound)

  implicit none
  integer(kind=4), intent(in)  :: n0_f
  integer(kind=4), intent(in)  :: n1_f
  real(kind=8), intent(inout)  :: f (0:n0_f - 1,0:n1_f - 1)
  real(kind=8), intent(in)  :: dt
  real(kind=8), intent(in)  :: v
  integer(kind=4), intent(in)  :: n0_rPts
  real(kind=8), intent(in)  :: rPts (0:n0_rPts - 1)
  integer(kind=4), intent(in)  :: n0_qPts
  real(kind=8), intent(in)  :: qPts (0:n0_qPts - 1)
  integer(kind=4), intent(in)  :: n0_nPts
  integer(kind=4), intent(in)  :: nPts (0:n0_nPts - 1)
  integer(kind=4), intent(in)  :: n0_drPhi_0
  integer(kind=4), intent(in)  :: n1_drPhi_0
  real(kind=8), intent(inout)  :: drPhi_0 (0:n0_drPhi_0 - 1,0:n1_drPhi_0 &
      - 1)
  integer(kind=4), intent(in)  :: n0_dthetaPhi_0
  integer(kind=4), intent(in)  :: n1_dthetaPhi_0
  real(kind=8), intent(inout)  :: dthetaPhi_0 (0:n0_dthetaPhi_0 - 1,0: &
      n1_dthetaPhi_0 - 1)
  integer(kind=4), intent(in)  :: n0_drPhi_k
  integer(kind=4), intent(in)  :: n1_drPhi_k
  real(kind=8), intent(inout)  :: drPhi_k (0:n0_drPhi_k - 1,0:n1_drPhi_k &
      - 1)
  integer(kind=4), intent(in)  :: n0_dthetaPhi_k
  integer(kind=4), intent(in)  :: n1_dthetaPhi_k
  real(kind=8), intent(inout)  :: dthetaPhi_k (0:n0_dthetaPhi_k - 1,0: &
      n1_dthetaPhi_k - 1)
  integer(kind=4), intent(in)  :: n0_endPts_k1_q
  integer(kind=4), intent(in)  :: n1_endPts_k1_q
  real(kind=8), intent(inout)  :: endPts_k1_q (0:n0_endPts_k1_q - 1,0: &
      n1_endPts_k1_q - 1)
  integer(kind=4), intent(in)  :: n0_endPts_k1_r
  integer(kind=4), intent(in)  :: n1_endPts_k1_r
  real(kind=8), intent(inout)  :: endPts_k1_r (0:n0_endPts_k1_r - 1,0: &
      n1_endPts_k1_r - 1)
  integer(kind=4), intent(in)  :: n0_endPts_k2_q
  integer(kind=4), intent(in)  :: n1_endPts_k2_q
  real(kind=8), intent(inout)  :: endPts_k2_q (0:n0_endPts_k2_q - 1,0: &
      n1_endPts_k2_q - 1)
  integer(kind=4), intent(in)  :: n0_endPts_k2_r
  integer(kind=4), intent(in)  :: n1_endPts_k2_r
  real(kind=8), intent(inout)  :: endPts_k2_r (0:n0_endPts_k2_r - 1,0: &
      n1_endPts_k2_r - 1)
  integer(kind=4), intent(in)  :: n0_kts1Phi
  real(kind=8), intent(in)  :: kts1Phi (0:n0_kts1Phi - 1)
  integer(kind=4), intent(in)  :: n0_kts2Phi
  real(kind=8), intent(in)  :: kts2Phi (0:n0_kts2Phi - 1)
  integer(kind=4), intent(in)  :: n0_coeffsPhi
  integer(kind=4), intent(in)  :: n1_coeffsPhi
  real(kind=8), intent(in)  :: coeffsPhi (0:n0_coeffsPhi - 1,0: &
      n1_coeffsPhi - 1)
  integer(kind=4), intent(in)  :: deg1Phi
  integer(kind=4), intent(in)  :: deg2Phi
  integer(kind=4), intent(in)  :: n0_kts1Pol
  real(kind=8), intent(in)  :: kts1Pol (0:n0_kts1Pol - 1)
  integer(kind=4), intent(in)  :: n0_kts2Pol
  real(kind=8), intent(in)  :: kts2Pol (0:n0_kts2Pol - 1)
  integer(kind=4), intent(in)  :: n0_coeffsPol
  integer(kind=4), intent(in)  :: n1_coeffsPol
  real(kind=8), intent(in)  :: coeffsPol (0:n0_coeffsPol - 1,0: &
      n1_coeffsPol - 1)
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
  integer(kind=4) :: i
  real(kind=8) :: r
  real(kind=8) :: multFactor_half
  integer(kind=4) :: j
  real(kind=8) :: rMax
  real(kind=8) :: theta
  integer(kind=4) :: idx

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
  multFactor_half = 0.5d0*(dt/B0)


  call eval_spline_2d_cross(qPts, rPts, kts1Phi, deg1Phi, kts2Phi, &
      deg2Phi, coeffsPhi, drPhi_0, 0, 1)
  call eval_spline_2d_cross(qPts, rPts, kts1Phi, deg1Phi, kts2Phi, &
      deg2Phi, coeffsPhi, dthetaPhi_0, 1, 0)


  idx = nPts(1) - 1
  rMax = rPts(idx)


  do i = 0, nPts(0) - 1, 1
    do j = 0, nPts(1) - 1, 1
      drPhi_0(j, i) = drPhi_0(j, i)/rPts(j)
      dthetaPhi_0(j, i) = dthetaPhi_0(j, i)/rPts(j)
      endPts_k1_q(j, i) = multFactor*(-drPhi_0(j, i)) + qPts(i)
      endPts_k1_r(j, i) = multFactor*dthetaPhi_0(j, i) + rPts(j)


      do while (endPts_k1_q(j, i) < 0)
        endPts_k1_q(j, i) = 2*3.141592653589793 + endPts_k1_q(j, i)
      end do
      do while (endPts_k1_q(j, i) > 2*3.141592653589793)
        endPts_k1_q(j, i) = -2*3.141592653589793 + endPts_k1_q(j, i)


      end do
      if (.not. (endPts_k1_r(j, i) > rMax .or. endPts_k1_r(j, i) < rPts( &
      0))) then
        ! Add the new value of phi to the derivatives
        ! x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
        ! ^^^^^^^^^^^^^^^
        drPhi_k(j, i) = eval_spline_2d_scalar(endPts_k1_q(j, i), &
      endPts_k1_r(j, i), kts1Phi, deg1Phi, kts2Phi, deg2Phi, coeffsPhi, &
      0, 1)/endPts_k1_r(j, i)
        dthetaPhi_k(j, i) = eval_spline_2d_scalar(endPts_k1_q(j, i), &
      endPts_k1_r(j, i), kts1Phi, deg1Phi, kts2Phi, deg2Phi, coeffsPhi, &
      1, 0)/endPts_k1_r(j, i)
      else
        drPhi_k(j, i) = 0.0d0
        dthetaPhi_k(j, i) = 0.0d0


        ! Step two of Heun method
        ! x^{n+1} = x^n + 0.5( f(x^n) + f(x^n + f(x^n)) )
      end if
      endPts_k2_q(j, i) = Mod(multFactor_half*(-drPhi_0(j, i) - drPhi_k( &
      j, i)) + qPts(i), 2*3.141592653589793)
      endPts_k2_r(j, i) = multFactor_half*(dthetaPhi_0(j, i) + &
      dthetaPhi_k(j, i)) + rPts(j)


    end do

  end do

  ! Step one of Heun method
  ! x' = x^n + f(x^n)
  ! Handle theta boundary conditions
  ! Find value at the determined point
  if (nulBound) then
    do i = 0, n0_qPts - 1, 1
      theta = qPts(i)
      do j = 0, n0_rPts - 1, 1
        r = rPts(j)
        if (endPts_k2_r(j, i) < rPts(0)) then
          f(j, i) = 0.0d0
        else if (endPts_k2_r(j, i) > rMax) then
          f(j, i) = 0.0d0
        else
          do while (endPts_k2_q(j, i) > 2*3.141592653589793)
            endPts_k2_q(j, i) = -2*3.141592653589793 + endPts_k2_q(j, i)
          end do
          do while (endPts_k2_q(j, i) < 0)
            endPts_k2_q(j, i) = 2*3.141592653589793 + endPts_k2_q(j, i)
          end do
          f(j, i) = eval_spline_2d_scalar(endPts_k2_q(j, i), endPts_k2_r &
      (j, i), kts1Pol, deg1Pol, kts2Pol, deg2Pol, coeffsPol, 0, 0)
        end if
      end do

    end do

  else
    do i = 0, n0_qPts - 1, 1
      theta = qPts(i)
      do j = 0, n0_rPts - 1, 1
        r = rPts(j)
        if (endPts_k2_r(j, i) < rPts(0)) then
          f(j, i) = fEq(rPts(0), v, CN0, kN0, deltaRN0, rp, CTi, kTi, &
      deltaRTi)
        else if (endPts_k2_r(j, i) > rMax) then
          f(j, i) = fEq(endPts_k2_r(j, i), v, CN0, kN0, deltaRN0, rp, &
      CTi, kTi, deltaRTi)
        else
          do while (endPts_k2_q(j, i) > 2*3.141592653589793)
            endPts_k2_q(j, i) = -2*3.141592653589793 + endPts_k2_q(j, i)
          end do
          do while (endPts_k2_q(j, i) < 0)
            endPts_k2_q(j, i) = 2*3.141592653589793 + endPts_k2_q(j, i)
          end do
          f(j, i) = eval_spline_2d_scalar(endPts_k2_q(j, i), endPts_k2_r &
      (j, i), kts1Pol, deg1Pol, kts2Pol, deg2Pol, coeffsPol, 0, 0)


        end if
      end do

    end do

  end if
end subroutine
! ........................................

end module