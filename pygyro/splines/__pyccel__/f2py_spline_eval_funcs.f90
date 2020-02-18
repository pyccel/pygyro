function find_span(n0_knots, knots, degree, x) result(returnVal)

  use spline_eval_funcs, only: mod_find_span => find_span
  implicit none
  integer(kind=4) :: returnVal  
  integer(kind=4), intent(in)  :: n0_knots 
  real(kind=8), intent(in)  :: knots (0:n0_knots - 1)
  integer(kind=4), intent(in)  :: degree 
  real(kind=8), intent(in)  :: x 

  returnVal = mod_find_span(knots,degree,x)
end function

subroutine basis_funs(n0_knots, knots, degree, x, span, n0_values, &
      values)

  use spline_eval_funcs, only: mod_basis_funs => basis_funs
  implicit none
  integer(kind=4), intent(in)  :: n0_knots 
  real(kind=8), intent(in)  :: knots (0:n0_knots - 1)
  integer(kind=4), intent(in)  :: degree 
  real(kind=8), intent(in)  :: x 
  integer(kind=4), intent(in)  :: span 
  integer(kind=4), intent(in)  :: n0_values 
  real(kind=8), intent(inout)  :: values (0:n0_values - 1)

  call mod_basis_funs(knots,degree,x,span,values)
end subroutine

subroutine basis_funs_1st_der(n0_knots, knots, degree, x, span, n0_ders, &
      ders)

  use spline_eval_funcs, only: mod_basis_funs_1st_der => &
      basis_funs_1st_der
  implicit none
  integer(kind=4), intent(in)  :: n0_knots 
  real(kind=8), intent(in)  :: knots (0:n0_knots - 1)
  integer(kind=4), intent(in)  :: degree 
  real(kind=8), intent(in)  :: x 
  integer(kind=4), intent(in)  :: span 
  integer(kind=4), intent(in)  :: n0_ders 
  real(kind=8), intent(inout)  :: ders (0:n0_ders - 1)

  call mod_basis_funs_1st_der(knots,degree,x,span,ders)
end subroutine

function eval_spline_1d_scalar(x, n0_knots, knots, degree, n0_coeffs, &
      coeffs, der) result(y)

  use spline_eval_funcs, only: mod_eval_spline_1d_scalar => &
      eval_spline_1d_scalar
  implicit none
  real(kind=8) :: y  
  real(kind=8), intent(in)  :: x 
  integer(kind=4), intent(in)  :: n0_knots 
  real(kind=8), intent(in)  :: knots (0:n0_knots - 1)
  integer(kind=4), intent(in)  :: degree 
  integer(kind=4), intent(in)  :: n0_coeffs 
  real(kind=8), intent(in)  :: coeffs (0:n0_coeffs - 1)
  integer(kind=4), intent(in)  :: der 

  y = mod_eval_spline_1d_scalar(x,knots,degree,coeffs,der)
end function

subroutine eval_spline_1d_vector(n0_x, x, n0_knots, knots, degree, &
      n0_coeffs, coeffs, n0_y, y, der)

  use spline_eval_funcs, only: mod_eval_spline_1d_vector => &
      eval_spline_1d_vector
  implicit none
  integer(kind=4), intent(in)  :: n0_x 
  real(kind=8), intent(in)  :: x (0:n0_x - 1)
  integer(kind=4), intent(in)  :: n0_knots 
  real(kind=8), intent(in)  :: knots (0:n0_knots - 1)
  integer(kind=4), intent(in)  :: degree 
  integer(kind=4), intent(in)  :: n0_coeffs 
  real(kind=8), intent(in)  :: coeffs (0:n0_coeffs - 1)
  integer(kind=4), intent(in)  :: n0_y 
  real(kind=8), intent(inout)  :: y (0:n0_y - 1)
  integer(kind=4), intent(in)  :: der 

  call mod_eval_spline_1d_vector(x,knots,degree,coeffs,y,der)
end subroutine

function eval_spline_2d_scalar(x, y, n0_kts1, kts1, deg1, n0_kts2, kts2, &
      deg2, n0_coeffs, n1_coeffs, coeffs, der1, der2) result(z)

  use spline_eval_funcs, only: mod_eval_spline_2d_scalar => &
      eval_spline_2d_scalar
  implicit none
  real(kind=8) :: z  
  real(kind=8), intent(in)  :: x 
  real(kind=8), intent(in)  :: y 
  integer(kind=4), intent(in)  :: n0_kts1 
  real(kind=8), intent(in)  :: kts1 (0:n0_kts1 - 1)
  integer(kind=4), intent(in)  :: deg1 
  integer(kind=4), intent(in)  :: n0_kts2 
  real(kind=8), intent(in)  :: kts2 (0:n0_kts2 - 1)
  integer(kind=4), intent(in)  :: deg2 
  integer(kind=4), intent(in)  :: n0_coeffs 
  integer(kind=4), intent(in)  :: n1_coeffs 
  real(kind=8), intent(in)  :: coeffs (0:n0_coeffs - 1,0:n1_coeffs - 1)
  integer(kind=4), intent(in)  :: der1 
  integer(kind=4), intent(in)  :: der2 

  !f2py integer(kind=4) :: n0_coeffs=shape(coeffs,1)
  !f2py integer(kind=4) :: n1_coeffs=shape(coeffs,0)
  !f2py intent(c) coeffs
  z = mod_eval_spline_2d_scalar(x,y,kts1,deg1,kts2,deg2,coeffs,der1,der2 &
      )
end function

subroutine eval_spline_2d_cross(n0_xVec, xVec, n0_yVec, yVec, n0_kts1, &
      kts1, deg1, n0_kts2, kts2, deg2, n0_coeffs, n1_coeffs, coeffs, &
      n0_z, n1_z, z, der1, der2)

  use spline_eval_funcs, only: mod_eval_spline_2d_cross => &
      eval_spline_2d_cross
  implicit none
  integer(kind=4), intent(in)  :: n0_xVec 
  real(kind=8), intent(in)  :: xVec (0:n0_xVec - 1)
  integer(kind=4), intent(in)  :: n0_yVec 
  real(kind=8), intent(in)  :: yVec (0:n0_yVec - 1)
  integer(kind=4), intent(in)  :: n0_kts1 
  real(kind=8), intent(in)  :: kts1 (0:n0_kts1 - 1)
  integer(kind=4), intent(in)  :: deg1 
  integer(kind=4), intent(in)  :: n0_kts2 
  real(kind=8), intent(in)  :: kts2 (0:n0_kts2 - 1)
  integer(kind=4), intent(in)  :: deg2 
  integer(kind=4), intent(in)  :: n0_coeffs 
  integer(kind=4), intent(in)  :: n1_coeffs 
  real(kind=8), intent(in)  :: coeffs (0:n0_coeffs - 1,0:n1_coeffs - 1)
  integer(kind=4), intent(in)  :: n0_z 
  integer(kind=4), intent(in)  :: n1_z 
  real(kind=8), intent(inout)  :: z (0:n0_z - 1,0:n1_z - 1)
  integer(kind=4), intent(in)  :: der1 
  integer(kind=4), intent(in)  :: der2 

  !f2py integer(kind=4) :: n0_coeffs=shape(coeffs,1)
  !f2py integer(kind=4) :: n1_coeffs=shape(coeffs,0)
  !f2py intent(c) coeffs
  !f2py integer(kind=4) :: n0_z=shape(z,1)
  !f2py integer(kind=4) :: n1_z=shape(z,0)
  !f2py intent(c) z
  call mod_eval_spline_2d_cross(xVec,yVec,kts1,deg1,kts2,deg2,coeffs,z, &
      der1,der2)
end subroutine

subroutine eval_spline_2d_vector(n0_x, x, n0_y, y, n0_kts1, kts1, deg1, &
      n0_kts2, kts2, deg2, n0_coeffs, n1_coeffs, coeffs, n0_z, z, der1, &
      der2)

  use spline_eval_funcs, only: mod_eval_spline_2d_vector => &
      eval_spline_2d_vector
  implicit none
  integer(kind=4), intent(in)  :: n0_x 
  real(kind=8), intent(in)  :: x (0:n0_x - 1)
  integer(kind=4), intent(in)  :: n0_y 
  real(kind=8), intent(in)  :: y (0:n0_y - 1)
  integer(kind=4), intent(in)  :: n0_kts1 
  real(kind=8), intent(in)  :: kts1 (0:n0_kts1 - 1)
  integer(kind=4), intent(in)  :: deg1 
  integer(kind=4), intent(in)  :: n0_kts2 
  real(kind=8), intent(in)  :: kts2 (0:n0_kts2 - 1)
  integer(kind=4), intent(in)  :: deg2 
  integer(kind=4), intent(in)  :: n0_coeffs 
  integer(kind=4), intent(in)  :: n1_coeffs 
  real(kind=8), intent(in)  :: coeffs (0:n0_coeffs - 1,0:n1_coeffs - 1)
  integer(kind=4), intent(in)  :: n0_z 
  real(kind=8), intent(inout)  :: z (0:n0_z - 1)
  integer(kind=4), intent(in)  :: der1 
  integer(kind=4), intent(in)  :: der2 

  !f2py integer(kind=4) :: n0_coeffs=shape(coeffs,1)
  !f2py integer(kind=4) :: n1_coeffs=shape(coeffs,0)
  !f2py intent(c) coeffs
  call mod_eval_spline_2d_vector(x,y,kts1,deg1,kts2,deg2,coeffs,z,der1, &
      der2)
end subroutine