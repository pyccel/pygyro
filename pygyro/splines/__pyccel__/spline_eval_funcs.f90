module spline_eval_funcs

implicit none

contains

!........................................
pure  function find_span(knots, degree, x) result(returnVal)

implicit none

integer(kind=8) :: returnVal
real(kind=8), intent(in) :: knots(0:)
integer(kind=8), value :: degree
real(kind=8), value :: x
integer(kind=8) :: low
integer(kind=8) :: high
integer(kind=8) :: span

!________________________CommentBlock________________________!
!                                                            !
!    Determine the knot span index at location x, given the  !
!    B-Splines' knot sequence and polynomial degree. See     !
!    Algorithm A2.1 in [1].                                  !
!                                                            !
!    For a degree p, the knot span index i identifies the    !
!    indices [i-p:i] of all p+1 non-zero basis functions at a!
!    given location x.                                       !
!                                                            !
!    Parameters                                              !
!    ----------                                              !
!    knots : array_like                                      !
!        Knots sequence.                                     !
!                                                            !
!    degree : int                                            !
!        Polynomial degree of B-splines.                     !
!                                                            !
!    x : float                                               !
!        Location of interest.                               !
!                                                            !
!    Returns                                                 !
!    -------                                                 !
!    span : int                                              !
!        Knot span index.                                    !
!                                                            !
!                                                            !
!____________________________________________________________!
!Knot index at left/right boundary
low = degree
high = 0_8
high = size(knots,1) - 1_8 - degree
!Check if point is exactly on left/right boundary, or outside domain
if (x <= knots(low)) then
  returnVal = low
else if (x >= knots(high)) then
  returnVal = high - 1_8
  !Perform binary search
else
  span = FLOOR((low + high)/Real(2_8, 8),8)
  do while (x < knots(span) .or. x >= knots(span + 1_8))
    if (x < knots(span)) then
      high = span
    else
      low = span
    end if
    span = FLOOR((low + high)/Real(2_8, 8),8)
  end do
  returnVal = span
end if
return

end function find_span
!........................................

!........................................
pure  subroutine basis_funs(knots, degree, x, span, values) 

implicit none

real(kind=8), intent(in) :: knots(0:)
integer(kind=8), value :: degree
real(kind=8), value :: x
integer(kind=8), value :: span
real(kind=8), intent(inout) :: values(0:)
real(kind=8) :: left(0:degree-1)
real(kind=8) :: right(0:degree-1)
integer(kind=8) :: j
real(kind=8) :: saved
integer(kind=8) :: r
real(kind=8) :: temp

!________________________CommentBlock________________________!
!                                                            !
!    Compute the non-vanishing B-splines at location x,      !
!    given the knot sequence, polynomial degree and knot     !
!    span. See Algorithm A2.2 in [1].                        !
!                                                            !
!    Parameters                                              !
!    ----------                                              !
!    knots : array_like                                      !
!        Knots sequence.                                     !
!                                                            !
!    degree : int                                            !
!        Polynomial degree of B-splines.                     !
!                                                            !
!    x : float                                               !
!        Evaluation point.                                   !
!                                                            !
!    span : int                                              !
!        Knot span index.                                    !
!                                                            !
!    Results                                                 !
!    -------                                                 !
!    values : numpy.ndarray                                  !
!        Values of p+1 non-vanishing B-Splines at location x.!
!                                                            !
!    Notes                                                   !
!    -----                                                   !
!    The original Algorithm A2.2 in The NURBS Book [1] is here!
!    slightly improved by using 'left' and 'right' temporary !
!    arrays that are one element shorter.                    !
!                                                            !
!                                                            !
!____________________________________________________________!
values(0_8) = 1.0d0
do j = 0_8, degree-1_8, 1_8
left(j) = x - knots(span - j)
right(j) = knots(span + 1_8 + j) - x
saved = 0.0d0
do r = 0_8, j + 1_8-1_8, 1_8
  temp = values(r) / (right(r) + left(j - r))
  values(r) = saved + right(r) * temp
  saved = left(j - r) * temp
end do
values(j + 1_8) = saved
end do

end subroutine basis_funs
!........................................

!........................................
pure  subroutine basis_funs_1st_der(knots, degree, x, span, ders) 

implicit none

real(kind=8), intent(in) :: knots(0:)
integer(kind=8), value :: degree
real(kind=8), value :: x
integer(kind=8), value :: span
real(kind=8), intent(inout) :: ders(0:)
real(kind=8) :: values(0:degree-1)
integer(kind=8) :: degree_max
real(kind=8) :: saved
integer(kind=8) :: j
real(kind=8) :: temp

!__________________________CommentBlock__________________________!
!                                                                !
!    Compute the first derivative of the non-vanishing B-splines !
!    at location x, given the knot sequence, polynomial degree   !
!    and knot span.                                              !
!                                                                !
!    See function 's_bsplines_non_uniform__eval_deriv' in        !
!    Selalib's source file                                       !
!    'src/splines/sll_m_bsplines_non_uniform.F90'.               !
!                                                                !
!    Parameters                                                  !
!    ----------                                                  !
!    knots : array_like                                          !
!        Knots sequence.                                         !
!                                                                !
!    degree : int                                                !
!        Polynomial degree of B-splines.                         !
!                                                                !
!    x : float                                                   !
!        Evaluation point.                                       !
!                                                                !
!    span : int                                                  !
!        Knot span index.                                        !
!                                                                !
!    Results                                                     !
!    -------                                                     !
!    ders : numpy.ndarray                                        !
!        Derivatives of p+1 non-vanishing B-Splines at location x.!
!                                                                !
!                                                                !
!________________________________________________________________!
!Compute nonzero basis functions and knot differences for splines
!up to degree deg-1
degree_max = degree - 1_8
call basis_funs(knots, degree_max, x, span, values)
!Compute derivatives at x using formula based on difference of
!splines of degree deg-1
!-------
!j = 0
saved = degree * values(0_8) / (knots(span + 1_8) - knots(span + 1_8 - &
      degree))
ders(0_8) = -saved
!j = 1,...,degree-1
do j = 1_8, degree-1_8, 1_8
temp = saved
saved = degree * values(j) / (knots(span + j + 1_8) - knots(span + j + &
      1_8 - degree))
ders(j) = temp - saved
!j = degree
end do
ders(degree) = saved

end subroutine basis_funs_1st_der
!........................................

!........................................
pure  function eval_spline_1d_scalar(x, knots, degree, coeffs, der) &
      result(y)

implicit none

real(kind=8) :: y
real(kind=8), value :: x
real(kind=8), intent(in) :: knots(0:)
integer(kind=8), value :: degree
real(kind=8), intent(in) :: coeffs(0:)
integer(kind=8), value :: der
integer(kind=8) :: span
real(kind=8) :: basis(0:degree + 1_8-1)
integer(kind=8) :: j

span = find_span(knots, degree, x)
if (der == 0_8) then
call basis_funs(knots, degree, x, span, basis)
else if (der == 1_8) then
call basis_funs_1st_der(knots, degree, x, span, basis)
end if
y = 0.0d0
do j = 0_8, degree + 1_8-1_8, 1_8
y = y + coeffs(span - degree + j) * basis(j)
end do
return

end function eval_spline_1d_scalar
!........................................

!........................................
pure  subroutine eval_spline_1d_vector(x, knots, degree, coeffs, y, der &
      )

implicit none

real(kind=8), intent(in) :: x(0:)
real(kind=8), intent(in) :: knots(0:)
integer(kind=8), value :: degree
real(kind=8), intent(in) :: coeffs(0:)
real(kind=8), intent(inout) :: y(0:)
integer(kind=8), value :: der
real(kind=8) :: basis(0:degree + 1_8-1)
integer(kind=8) :: i
real(kind=8) :: xi
integer(kind=8) :: span
integer(kind=8) :: j

if (der == 0_8) then
do i = 0_8, size(x,1)-1_8, 1_8
xi = x(i)
span = find_span(knots, degree, xi)
call basis_funs(knots, degree, xi, span, basis)
y(i) = 0.0d0
do j = 0_8, degree + 1_8-1_8, 1_8
y(i) = y(i) + coeffs(span - degree + j) * basis(j)
end do
end do
else if (der == 1_8) then
do i = 0_8, size(x,1)-1_8, 1_8
xi = x(i)
span = find_span(knots, degree, xi)
call basis_funs(knots, degree, xi, span, basis)
call basis_funs_1st_der(knots, degree, xi, span, basis)
y(i) = 0.0d0
do j = 0_8, degree + 1_8-1_8, 1_8
y(i) = y(i) + coeffs(span - degree + j) * basis(j)
end do
end do
end if

end subroutine eval_spline_1d_vector
!........................................

!........................................
pure  function eval_spline_2d_scalar(x, y, kts1, deg1, kts2, deg2, &
      coeffs, der1, der2) result(z)

implicit none

real(kind=8) :: z
real(kind=8), value :: x
real(kind=8), value :: y
real(kind=8), intent(in) :: kts1(0:)
integer(kind=8), value :: deg1
real(kind=8), intent(in) :: kts2(0:)
integer(kind=8), value :: deg2
real(kind=8), intent(in) :: coeffs(0:,0:)
integer(kind=8), value :: der1
integer(kind=8), value :: der2
real(kind=8) :: basis1(0:deg1 + 1_8-1)
real(kind=8) :: basis2(0:deg2 + 1_8-1)
integer(kind=8) :: span1
integer(kind=8) :: span2
real(kind=8) :: theCoeffs(0:deg2 + 1_8-1,0:deg1 + 1_8-1)
integer(kind=8) :: i
integer(kind=8) :: j

span1 = find_span(kts1, deg1, x)
span2 = find_span(kts2, deg2, y)
if (der1 == 0_8) then
call basis_funs(kts1, deg1, x, span1, basis1)
else if (der1 == 1_8) then
call basis_funs_1st_der(kts1, deg1, x, span1, basis1)
end if
if (der2 == 0_8) then
call basis_funs(kts2, deg2, y, span2, basis2)
else if (der2 == 1_8) then
call basis_funs_1st_der(kts2, deg2, y, span2, basis2)
end if
theCoeffs(:, :) = coeffs(span2 - deg2:span2 + 1_8 - 1_8, span1 - deg1: &
      span1 + 1_8 - 1_8)
z = 0.0d0
do i = 0_8, deg1 + 1_8-1_8, 1_8
theCoeffs(0_8, i) = theCoeffs(0_8, i) * basis2(0_8)
do j = 1_8, deg2 + 1_8-1_8, 1_8
theCoeffs(0_8, i) = theCoeffs(0_8, i) + theCoeffs(j, i) * basis2(j)
end do
z = z + theCoeffs(0_8, i) * basis1(i)
end do
return

end function eval_spline_2d_scalar
!........................................

!........................................
pure  subroutine eval_spline_2d_cross(xVec, yVec, kts1, deg1, kts2, deg2 &
      , coeffs, z, der1, der2)

implicit none

real(kind=8), intent(in) :: xVec(0:)
real(kind=8), intent(in) :: yVec(0:)
real(kind=8), intent(in) :: kts1(0:)
integer(kind=8), value :: deg1
real(kind=8), intent(in) :: kts2(0:)
integer(kind=8), value :: deg2
real(kind=8), intent(in) :: coeffs(0:,0:)
real(kind=8), intent(inout) :: z(0:,0:)
integer(kind=8), value :: der1
integer(kind=8), value :: der2
real(kind=8) :: basis1(0:deg1 + 1_8-1)
real(kind=8) :: basis2(0:deg2 + 1_8-1)
real(kind=8) :: theCoeffs(0:deg2 + 1_8-1,0:deg1 + 1_8-1)
integer(kind=8) :: i
real(kind=8) :: x
integer(kind=8) :: span1
integer(kind=8) :: j
real(kind=8) :: y
integer(kind=8) :: span2
integer(kind=8) :: k
integer(kind=8) :: l

if (der1 == 0_8 .and. der2 == 0_8) then
do i = 0_8, size(xVec,1)-1_8, 1_8
x = xVec(i)
span1 = find_span(kts1, deg1, x)
call basis_funs(kts1, deg1, x, span1, basis1)
do j = 0_8, size(yVec,1)-1_8, 1_8
y = yVec(j)
span2 = find_span(kts2, deg2, y)
call basis_funs(kts2, deg2, y, span2, basis2)
theCoeffs(:, :) = coeffs(span2 - deg2:span2 + 1_8 - 1_8, span1 - deg1: &
      span1 + 1_8 - 1_8)
z(j, i) = 0.0d0
do k = 0_8, deg1 + 1_8-1_8, 1_8
theCoeffs(0_8, k) = theCoeffs(0_8, k) * basis2(0_8)
do l = 1_8, deg2 + 1_8-1_8, 1_8
theCoeffs(0_8, k) = theCoeffs(0_8, k) + theCoeffs(l, k) * basis2(l)
end do
z(j, i) = z(j, i) + theCoeffs(0_8, k) * basis1(k)
end do
end do
end do
else if (der1 == 0_8 .and. der2 == 1_8) then
do i = 0_8, size(xVec,1)-1_8, 1_8
x = xVec(i)
span1 = find_span(kts1, deg1, x)
call basis_funs(kts1, deg1, x, span1, basis1)
do j = 0_8, size(yVec,1)-1_8, 1_8
y = yVec(j)
span2 = find_span(kts2, deg2, y)
call basis_funs_1st_der(kts2, deg2, y, span2, basis2)
theCoeffs(:, :) = coeffs(span2 - deg2:span2 + 1_8 - 1_8, span1 - deg1: &
      span1 + 1_8 - 1_8)
z(j, i) = 0.0d0
do k = 0_8, deg1 + 1_8-1_8, 1_8
theCoeffs(0_8, k) = theCoeffs(0_8, k) * basis2(0_8)
do l = 1_8, deg2 + 1_8-1_8, 1_8
theCoeffs(0_8, k) = theCoeffs(0_8, k) + theCoeffs(l, k) * basis2(l)
end do
z(j, i) = z(j, i) + theCoeffs(0_8, k) * basis1(k)
end do
end do
end do
else if (der1 == 1_8 .and. der2 == 0_8) then
do i = 0_8, size(xVec,1)-1_8, 1_8
x = xVec(i)
span1 = find_span(kts1, deg1, x)
call basis_funs_1st_der(kts1, deg1, x, span1, basis1)
do j = 0_8, size(yVec,1)-1_8, 1_8
y = yVec(j)
span2 = find_span(kts2, deg2, y)
call basis_funs(kts2, deg2, y, span2, basis2)
theCoeffs(:, :) = coeffs(span2 - deg2:span2 + 1_8 - 1_8, span1 - deg1: &
      span1 + 1_8 - 1_8)
z(j, i) = 0.0d0
do k = 0_8, deg1 + 1_8-1_8, 1_8
theCoeffs(0_8, k) = theCoeffs(0_8, k) * basis2(0_8)
do l = 1_8, deg2 + 1_8-1_8, 1_8
theCoeffs(0_8, k) = theCoeffs(0_8, k) + theCoeffs(l, k) * basis2(l)
end do
z(j, i) = z(j, i) + theCoeffs(0_8, k) * basis1(k)
end do
end do
end do
else if (der1 == 1_8 .and. der2 == 1_8) then
do i = 0_8, size(xVec,1)-1_8, 1_8
x = xVec(i)
span1 = find_span(kts1, deg1, x)
call basis_funs_1st_der(kts1, deg1, x, span1, basis1)
do j = 0_8, size(yVec,1)-1_8, 1_8
y = yVec(j)
span2 = find_span(kts2, deg2, y)
call basis_funs_1st_der(kts2, deg2, y, span2, basis2)
theCoeffs(:, :) = coeffs(span2 - deg2:span2 + 1_8 - 1_8, span1 - deg1: &
      span1 + 1_8 - 1_8)
z(j, i) = 0.0d0
do k = 0_8, deg1 + 1_8-1_8, 1_8
theCoeffs(0_8, k) = theCoeffs(0_8, k) * basis2(0_8)
do l = 1_8, deg2 + 1_8-1_8, 1_8
theCoeffs(0_8, k) = theCoeffs(0_8, k) + theCoeffs(l, k) * basis2(l)
end do
z(j, i) = z(j, i) + theCoeffs(0_8, k) * basis1(k)
end do
end do
end do
end if

end subroutine eval_spline_2d_cross
!........................................

!........................................
pure  subroutine eval_spline_2d_vector(x, y, kts1, deg1, kts2, deg2, &
      coeffs, z, der1, der2)

implicit none

real(kind=8), intent(in) :: x(0:)
real(kind=8), intent(in) :: y(0:)
real(kind=8), intent(in) :: kts1(0:)
integer(kind=8), value :: deg1
real(kind=8), intent(in) :: kts2(0:)
integer(kind=8), value :: deg2
real(kind=8), intent(in) :: coeffs(0:,0:)
real(kind=8), intent(inout) :: z(0:)
integer(kind=8), value :: der1
integer(kind=8), value :: der2
real(kind=8) :: basis1(0:deg1 + 1_8-1)
real(kind=8) :: basis2(0:deg2 + 1_8-1)
real(kind=8) :: theCoeffs(0:deg2 + 1_8-1,0:deg1 + 1_8-1)
integer(kind=8) :: i
integer(kind=8) :: span1
integer(kind=8) :: span2
integer(kind=8) :: j
integer(kind=8) :: k

if (der1 == 0_8) then
if (der2 == 0_8) then
do i = 0_8, size(x,1)-1_8, 1_8
span1 = find_span(kts1, deg1, x(i))
span2 = find_span(kts2, deg2, y(i))
call basis_funs(kts1, deg1, x(i), span1, basis1)
call basis_funs(kts2, deg2, y(i), span2, basis2)
theCoeffs(:, :) = coeffs(span2 - deg2:span2 + 1_8 - 1_8, span1 - deg1: &
      span1 + 1_8 - 1_8)
z(i) = 0.0d0
do j = 0_8, deg1 + 1_8-1_8, 1_8
theCoeffs(0_8, j) = theCoeffs(0_8, j) * basis2(0_8)
do k = 1_8, deg2 + 1_8-1_8, 1_8
theCoeffs(0_8, j) = theCoeffs(0_8, j) + theCoeffs(k, j) * basis2(k)
end do
z(i) = z(i) + theCoeffs(0_8, j) * basis1(j)
end do
end do
else if (der2 == 1_8) then
do i = 0_8, size(x,1)-1_8, 1_8
span1 = find_span(kts1, deg1, x(i))
span2 = find_span(kts2, deg2, y(i))
call basis_funs(kts1, deg1, x(i), span1, basis1)
call basis_funs_1st_der(kts2, deg2, y(i), span2, basis2)
theCoeffs(:, :) = coeffs(span2 - deg2:span2 + 1_8 - 1_8, span1 - deg1: &
      span1 + 1_8 - 1_8)
z(i) = 0.0d0
do j = 0_8, deg1 + 1_8-1_8, 1_8
theCoeffs(0_8, j) = theCoeffs(0_8, j) * basis2(0_8)
do k = 1_8, deg2 + 1_8-1_8, 1_8
theCoeffs(0_8, j) = theCoeffs(0_8, j) + theCoeffs(k, j) * basis2(k)
end do
z(i) = z(i) + theCoeffs(0_8, j) * basis1(j)
end do
end do
end if
else if (der1 == 1_8) then
if (der2 == 0_8) then
do i = 0_8, size(x,1)-1_8, 1_8
span1 = find_span(kts1, deg1, x(i))
span2 = find_span(kts2, deg2, y(i))
call basis_funs_1st_der(kts1, deg1, x(i), span1, basis1)
call basis_funs(kts2, deg2, y(i), span2, basis2)
theCoeffs(:, :) = coeffs(span2 - deg2:span2 + 1_8 - 1_8, span1 - deg1: &
      span1 + 1_8 - 1_8)
z(i) = 0.0d0
do j = 0_8, deg1 + 1_8-1_8, 1_8
theCoeffs(0_8, j) = theCoeffs(0_8, j) * basis2(0_8)
do k = 1_8, deg2 + 1_8-1_8, 1_8
theCoeffs(0_8, j) = theCoeffs(0_8, j) + theCoeffs(k, j) * basis2(k)
end do
z(i) = z(i) + theCoeffs(0_8, j) * basis1(j)
end do
end do
else if (der2 == 1_8) then
do i = 0_8, size(x,1)-1_8, 1_8
span1 = find_span(kts1, deg1, x(i))
span2 = find_span(kts2, deg2, y(i))
call basis_funs_1st_der(kts1, deg1, x(i), span1, basis1)
call basis_funs_1st_der(kts2, deg2, y(i), span2, basis2)
theCoeffs(:, :) = coeffs(span2 - deg2:span2 + 1_8 - 1_8, span1 - deg1: &
      span1 + 1_8 - 1_8)
z(i) = 0.0d0
do j = 0_8, deg1 + 1_8-1_8, 1_8
theCoeffs(0_8, j) = theCoeffs(0_8, j) * basis2(0_8)
do k = 1_8, deg2 + 1_8-1_8, 1_8
theCoeffs(0_8, j) = theCoeffs(0_8, j) + theCoeffs(k, j) * basis2(k)
end do
z(i) = z(i) + theCoeffs(0_8, j) * basis1(j)
end do
end do
end if
end if

end subroutine eval_spline_2d_vector
!........................................

end module spline_eval_funcs
