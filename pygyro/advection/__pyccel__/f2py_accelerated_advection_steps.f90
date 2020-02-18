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

  use accelerated_advection_steps, only: &
      mod_poloidal_advection_step_expl => poloidal_advection_step_expl
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

  !f2py integer(kind=4) :: n0_f=shape(f,1)
  !f2py integer(kind=4) :: n1_f=shape(f,0)
  !f2py intent(c) f
  !f2py integer(kind=4) :: n0_drPhi_0=shape(drPhi_0,1)
  !f2py integer(kind=4) :: n1_drPhi_0=shape(drPhi_0,0)
  !f2py intent(c) drPhi_0
  !f2py integer(kind=4) :: n0_dthetaPhi_0=shape(dthetaPhi_0,1)
  !f2py integer(kind=4) :: n1_dthetaPhi_0=shape(dthetaPhi_0,0)
  !f2py intent(c) dthetaPhi_0
  !f2py integer(kind=4) :: n0_drPhi_k=shape(drPhi_k,1)
  !f2py integer(kind=4) :: n1_drPhi_k=shape(drPhi_k,0)
  !f2py intent(c) drPhi_k
  !f2py integer(kind=4) :: n0_dthetaPhi_k=shape(dthetaPhi_k,1)
  !f2py integer(kind=4) :: n1_dthetaPhi_k=shape(dthetaPhi_k,0)
  !f2py intent(c) dthetaPhi_k
  !f2py integer(kind=4) :: n0_endPts_k1_q=shape(endPts_k1_q,1)
  !f2py integer(kind=4) :: n1_endPts_k1_q=shape(endPts_k1_q,0)
  !f2py intent(c) endPts_k1_q
  !f2py integer(kind=4) :: n0_endPts_k1_r=shape(endPts_k1_r,1)
  !f2py integer(kind=4) :: n1_endPts_k1_r=shape(endPts_k1_r,0)
  !f2py intent(c) endPts_k1_r
  !f2py integer(kind=4) :: n0_endPts_k2_q=shape(endPts_k2_q,1)
  !f2py integer(kind=4) :: n1_endPts_k2_q=shape(endPts_k2_q,0)
  !f2py intent(c) endPts_k2_q
  !f2py integer(kind=4) :: n0_endPts_k2_r=shape(endPts_k2_r,1)
  !f2py integer(kind=4) :: n1_endPts_k2_r=shape(endPts_k2_r,0)
  !f2py intent(c) endPts_k2_r
  !f2py integer(kind=4) :: n0_coeffsPhi=shape(coeffsPhi,1)
  !f2py integer(kind=4) :: n1_coeffsPhi=shape(coeffsPhi,0)
  !f2py intent(c) coeffsPhi
  !f2py integer(kind=4) :: n0_coeffsPol=shape(coeffsPol,1)
  !f2py integer(kind=4) :: n1_coeffsPol=shape(coeffsPol,0)
  !f2py intent(c) coeffsPol
  call mod_poloidal_advection_step_expl(f,dt,v,rPts,qPts,nPts,drPhi_0, &
      dthetaPhi_0,drPhi_k,dthetaPhi_k,endPts_k1_q,endPts_k1_r, &
      endPts_k2_q,endPts_k2_r,kts1Phi,kts2Phi,coeffsPhi,deg1Phi,deg2Phi &
      ,kts1Pol,kts2Pol,coeffsPol,deg1Pol,deg2Pol,CN0,kN0,deltaRN0,rp, &
      CTi,kTi,deltaRTi,B0,nulBound)
end subroutine

subroutine v_parallel_advection_eval_step(n0_f, f, n0_vPts, vPts, rPos, &
      vMin, vMax, n0_kts, kts, deg, n0_coeffs, coeffs, CN0, kN0, &
      deltaRN0, rp, CTi, kTi, deltaRTi, bound)

  use accelerated_advection_steps, only: &
      mod_v_parallel_advection_eval_step => &
      v_parallel_advection_eval_step
  implicit none
  integer(kind=4), intent(in)  :: n0_f 
  real(kind=8), intent(inout)  :: f (0:n0_f - 1)
  integer(kind=4), intent(in)  :: n0_vPts 
  real(kind=8), intent(in)  :: vPts (0:n0_vPts - 1)
  real(kind=8), intent(in)  :: rPos 
  real(kind=8), intent(in)  :: vMin 
  real(kind=8), intent(in)  :: vMax 
  integer(kind=4), intent(in)  :: n0_kts 
  real(kind=8), intent(in)  :: kts (0:n0_kts - 1)
  integer(kind=4), intent(in)  :: deg 
  integer(kind=4), intent(in)  :: n0_coeffs 
  real(kind=8), intent(in)  :: coeffs (0:n0_coeffs - 1)
  real(kind=8), intent(in)  :: CN0 
  real(kind=8), intent(in)  :: kN0 
  real(kind=8), intent(in)  :: deltaRN0 
  real(kind=8), intent(in)  :: rp 
  real(kind=8), intent(in)  :: CTi 
  real(kind=8), intent(in)  :: kTi 
  real(kind=8), intent(in)  :: deltaRTi 
  integer(kind=4), intent(in)  :: bound 

  call mod_v_parallel_advection_eval_step(f,vPts,rPos,vMin,vMax,kts,deg, &
      coeffs,CN0,kN0,deltaRN0,rp,CTi,kTi,deltaRTi,bound)
end subroutine

subroutine get_lagrange_vals(i, nz, n0_shifts, shifts, n0_vals, n1_vals, &
      n2_vals, vals, n0_qVals, qVals, n0_thetaShifts, thetaShifts, &
      n0_kts, kts, deg, n0_coeffs, coeffs)

  use accelerated_advection_steps, only: mod_get_lagrange_vals => &
      get_lagrange_vals
  implicit none
  integer(kind=4), intent(in)  :: i 
  integer(kind=4), intent(in)  :: nz 
  integer(kind=4), intent(in)  :: n0_shifts 
  integer(kind=4), intent(in)  :: shifts (0:n0_shifts - 1)
  integer(kind=4), intent(in)  :: n0_vals 
  integer(kind=4), intent(in)  :: n1_vals 
  integer(kind=4), intent(in)  :: n2_vals 
  real(kind=8), intent(inout)  :: vals (0:n0_vals - 1,0:n1_vals - 1,0: &
      n2_vals - 1)
  integer(kind=4), intent(in)  :: n0_qVals 
  real(kind=8), intent(in)  :: qVals (0:n0_qVals - 1)
  integer(kind=4), intent(in)  :: n0_thetaShifts 
  real(kind=8), intent(in)  :: thetaShifts (0:n0_thetaShifts - 1)
  integer(kind=4), intent(in)  :: n0_kts 
  real(kind=8), intent(in)  :: kts (0:n0_kts - 1)
  integer(kind=4), intent(in)  :: deg 
  integer(kind=4), intent(in)  :: n0_coeffs 
  real(kind=8), intent(in)  :: coeffs (0:n0_coeffs - 1)

  !f2py integer(kind=4) :: n0_vals=shape(vals,2)
  !f2py integer(kind=4) :: n1_vals=shape(vals,1)
  !f2py integer(kind=4) :: n2_vals=shape(vals,0)
  !f2py intent(c) vals
  call mod_get_lagrange_vals(i,nz,shifts,vals,qVals,thetaShifts,kts,deg, &
      coeffs)
end subroutine

subroutine flux_advection(nq, nr, n0_f, n1_f, f, n0_coeffs, coeffs, &
      n0_vals, n1_vals, n2_vals, vals)

  use accelerated_advection_steps, only: mod_flux_advection => &
      flux_advection
  implicit none
  integer(kind=4), intent(in)  :: nq 
  integer(kind=4), intent(in)  :: nr 
  integer(kind=4), intent(in)  :: n0_f 
  integer(kind=4), intent(in)  :: n1_f 
  real(kind=8), intent(inout)  :: f (0:n0_f - 1,0:n1_f - 1)
  integer(kind=4), intent(in)  :: n0_coeffs 
  real(kind=8), intent(in)  :: coeffs (0:n0_coeffs - 1)
  integer(kind=4), intent(in)  :: n0_vals 
  integer(kind=4), intent(in)  :: n1_vals 
  integer(kind=4), intent(in)  :: n2_vals 
  real(kind=8), intent(in)  :: vals (0:n0_vals - 1,0:n1_vals - 1,0: &
      n2_vals - 1)

  !f2py integer(kind=4) :: n0_f=shape(f,1)
  !f2py integer(kind=4) :: n1_f=shape(f,0)
  !f2py intent(c) f
  !f2py integer(kind=4) :: n0_vals=shape(vals,2)
  !f2py integer(kind=4) :: n1_vals=shape(vals,1)
  !f2py integer(kind=4) :: n2_vals=shape(vals,0)
  !f2py intent(c) vals
  call mod_flux_advection(nq,nr,f,coeffs,vals)
end subroutine

subroutine poloidal_advection_step_impl(n0_f, n1_f, f, dt, v, n0_rPts, &
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
      tol, nulBound)

  use accelerated_advection_steps, only: &
      mod_poloidal_advection_step_impl => poloidal_advection_step_impl
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
  real(kind=8), intent(in)  :: tol 
  logical(kind=1), intent(in)  :: nulBound 

  !f2py integer(kind=4) :: n0_f=shape(f,1)
  !f2py integer(kind=4) :: n1_f=shape(f,0)
  !f2py intent(c) f
  !f2py integer(kind=4) :: n0_drPhi_0=shape(drPhi_0,1)
  !f2py integer(kind=4) :: n1_drPhi_0=shape(drPhi_0,0)
  !f2py intent(c) drPhi_0
  !f2py integer(kind=4) :: n0_dthetaPhi_0=shape(dthetaPhi_0,1)
  !f2py integer(kind=4) :: n1_dthetaPhi_0=shape(dthetaPhi_0,0)
  !f2py intent(c) dthetaPhi_0
  !f2py integer(kind=4) :: n0_drPhi_k=shape(drPhi_k,1)
  !f2py integer(kind=4) :: n1_drPhi_k=shape(drPhi_k,0)
  !f2py intent(c) drPhi_k
  !f2py integer(kind=4) :: n0_dthetaPhi_k=shape(dthetaPhi_k,1)
  !f2py integer(kind=4) :: n1_dthetaPhi_k=shape(dthetaPhi_k,0)
  !f2py intent(c) dthetaPhi_k
  !f2py integer(kind=4) :: n0_endPts_k1_q=shape(endPts_k1_q,1)
  !f2py integer(kind=4) :: n1_endPts_k1_q=shape(endPts_k1_q,0)
  !f2py intent(c) endPts_k1_q
  !f2py integer(kind=4) :: n0_endPts_k1_r=shape(endPts_k1_r,1)
  !f2py integer(kind=4) :: n1_endPts_k1_r=shape(endPts_k1_r,0)
  !f2py intent(c) endPts_k1_r
  !f2py integer(kind=4) :: n0_endPts_k2_q=shape(endPts_k2_q,1)
  !f2py integer(kind=4) :: n1_endPts_k2_q=shape(endPts_k2_q,0)
  !f2py intent(c) endPts_k2_q
  !f2py integer(kind=4) :: n0_endPts_k2_r=shape(endPts_k2_r,1)
  !f2py integer(kind=4) :: n1_endPts_k2_r=shape(endPts_k2_r,0)
  !f2py intent(c) endPts_k2_r
  !f2py integer(kind=4) :: n0_coeffsPhi=shape(coeffsPhi,1)
  !f2py integer(kind=4) :: n1_coeffsPhi=shape(coeffsPhi,0)
  !f2py intent(c) coeffsPhi
  !f2py integer(kind=4) :: n0_coeffsPol=shape(coeffsPol,1)
  !f2py integer(kind=4) :: n1_coeffsPol=shape(coeffsPol,0)
  !f2py intent(c) coeffsPol
  call mod_poloidal_advection_step_impl(f,dt,v,rPts,qPts,nPts,drPhi_0, &
      dthetaPhi_0,drPhi_k,dthetaPhi_k,endPts_k1_q,endPts_k1_r, &
      endPts_k2_q,endPts_k2_r,kts1Phi,kts2Phi,coeffsPhi,deg1Phi,deg2Phi &
      ,kts1Pol,kts2Pol,coeffsPol,deg1Pol,deg2Pol,CN0,kN0,deltaRN0,rp, &
      CTi,kTi,deltaRTi,B0,tol,nulBound)
end subroutine