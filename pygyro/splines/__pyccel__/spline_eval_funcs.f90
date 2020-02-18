module spline_eval_funcs

implicit none




contains

!........................................
pure function find_span(knots, degree, x) result(returnVal)

implicit none
integer(kind=4) :: returnVal  
real(kind=8), intent(in)  :: knots (0:)
integer(kind=4), intent(in)  :: degree 
real(kind=8), intent(in)  :: x 
integer(kind=4) :: low  
integer(kind=4) :: high  
integer(kind=4) :: span  

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
high = 0
high = -degree - 1 + size(knots,1)


!Check if point is exactly on left/right boundary, or outside domain
if (x <= knots(low)) then
  returnVal = low
else if (x >= knots(high)) then
  returnVal = high - 1
else
  !Perform binary search
  span = floor((high + low)/Real(2, 8))
  do while (x >= knots(span + 1) .or. x < knots(span)) 
    if (x < knots(span)) then
      high = span
    else
      low = span
    end if
    span = floor((high + low)/Real(2, 8))
  end do
  returnVal = span
end if
return
end function
!........................................

!........................................
pure subroutine basis_funs(knots, degree, x, span, values) 

implicit none
real(kind=8), intent(in)  :: knots (0:)
integer(kind=4), intent(in)  :: degree 
real(kind=8), intent(in)  :: x 
integer(kind=4), intent(in)  :: span 
real(kind=8), intent(inout)  :: values (0:)
real(kind=8) :: left (0:degree - 1) 
real(kind=8) :: right (0:degree - 1) 
integer(kind=4) :: j  
real(kind=8) :: saved  
integer(kind=4) :: r  
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

left = 0.0
right = 0.0


values(0) = 1.0d0
do j = 0, degree - 1, 1
left(j) = x - knots(-j + span)
right(j) = -x + knots(span + j + 1)
saved = 0.0d0
do r = 0, j, 1
  temp = values(r)/(left(j - r) + right(r))
  values(r) = saved + temp*right(r)
  saved = temp*left(j - r)
end do

values(j + 1) = saved


end do

end subroutine
!........................................

!........................................
pure subroutine basis_funs_1st_der(knots, degree, x, span, ders) 

implicit none
real(kind=8), intent(in)  :: knots (0:)
integer(kind=4), intent(in)  :: degree 
real(kind=8), intent(in)  :: x 
integer(kind=4), intent(in)  :: span 
real(kind=8), intent(inout)  :: ders (0:)
real(kind=8) :: values (0:degree - 1) 
integer(kind=4) :: degree_max  
real(kind=8) :: saved  
integer(kind=4) :: j  
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

values = 0.0
degree_max = degree - 1
call basis_funs(knots, degree_max, x, span, values)


!Compute derivatives at x using formula based on difference of
!splines of degree deg-1
!-------
!j = 0
saved = degree*(values(0)/(-knots(span + 1 - degree) + knots(span + 1)))
ders(0) = -saved
!j = 1,...,degree-1
do j = 1, degree - 1, 1
temp = saved
saved = degree*(values(j)/(-knots(span + j + 1 - degree) + knots(span + &
      j + 1)))
ders(j) = -saved + temp
end do

!j = degree
ders(degree) = saved
end subroutine
!........................................

!........................................
pure function eval_spline_1d_scalar(x, knots, degree, coeffs, der) &
      result(y)

implicit none
real(kind=8) :: y  
real(kind=8), intent(in)  :: x 
real(kind=8), intent(in)  :: knots (0:)
integer(kind=4), intent(in)  :: degree 
real(kind=8), intent(in)  :: coeffs (0:)
integer(kind=4), intent(in)  :: der 
integer(kind=4) :: span  
real(kind=8) :: basis (0:degree) 
integer(kind=4) :: j  

span = find_span(knots, degree, x)



basis = 0.0
if (der == 0 ) then
call basis_funs(knots, degree, x, span, basis)
else if (der == 1 ) then
call basis_funs_1st_der(knots, degree, x, span, basis)
end if
y = 0.0d0
do j = 0, degree, 1
y = y + basis(j)*coeffs(span - degree + j)
end do

return
end function
!........................................

!........................................
pure subroutine eval_spline_1d_vector(x, knots, degree, coeffs, y, der) 

implicit none
real(kind=8), intent(in)  :: x (0:)
real(kind=8), intent(in)  :: knots (0:)
integer(kind=4), intent(in)  :: degree 
real(kind=8), intent(in)  :: coeffs (0:)
real(kind=8), intent(inout)  :: y (0:)
integer(kind=4), intent(in)  :: der 
real(kind=8) :: basis (0:degree) 
integer(kind=4) :: i  
real(kind=8) :: xi  
integer(kind=4) :: span  
integer(kind=4) :: j  


basis = 0.0


if (der == 0 ) then
do i = 0, size(x,1) - 1, 1
xi = x(i)
span = find_span(knots, degree, xi)
call basis_funs(knots, degree, xi, span, basis)


y(i) = 0.0d0
do j = 0, degree, 1
y(i) = basis(j)*coeffs(span - degree + j) + y(i)
end do

end do

else if (der == 1 ) then
do i = 0, size(x,1) - 1, 1
xi = x(i)
span = find_span(knots, degree, xi)
call basis_funs(knots, degree, xi, span, basis)
call basis_funs_1st_der(knots, degree, xi, span, basis)


y(i) = 0.0d0
do j = 0, degree, 1
y(i) = basis(j)*coeffs(span - degree + j) + y(i)


end do

end do

end if
end subroutine
!........................................

!........................................
pure function eval_spline_2d_scalar(x, y, kts1, deg1, kts2, deg2, coeffs &
      , der1, der2) result(z)

implicit none
real(kind=8) :: z  
real(kind=8), intent(in)  :: x 
real(kind=8), intent(in)  :: y 
real(kind=8), intent(in)  :: kts1 (0:)
integer(kind=4), intent(in)  :: deg1 
real(kind=8), intent(in)  :: kts2 (0:)
integer(kind=4), intent(in)  :: deg2 
real(kind=8), intent(in)  :: coeffs (0:,0:)
integer(kind=4), intent(in)  :: der1 
integer(kind=4), intent(in)  :: der2 
real(kind=8) :: basis1 (0:deg1) 
real(kind=8) :: basis2 (0:deg2) 
integer(kind=4) :: span1  
integer(kind=4) :: span2  
real(kind=8) :: theCoeffs (0:deg2,0:deg1) 
integer(kind=4) :: i  
integer(kind=4) :: j  


basis1 = 0.0
basis2 = 0.0


span1 = find_span(kts1, deg1, x)
span2 = find_span(kts2, deg2, y)


if (der1 == 0 ) then
call basis_funs(kts1, deg1, x, span1, basis1)
else if (der1 == 1 ) then
call basis_funs_1st_der(kts1, deg1, x, span1, basis1)
end if
if (der2 == 0 ) then
call basis_funs(kts2, deg2, y, span2, basis2)
else if (der2 == 1 ) then
call basis_funs_1st_der(kts2, deg2, y, span2, basis2)
end if
theCoeffs = 0.0
theCoeffs(:, :) = coeffs(-deg2 + span2:span2, -deg1 + span1:span1)


z = 0.0d0
do i = 0, deg1, 1
theCoeffs(0, i) = basis2(0)*theCoeffs(0, i)
do j = 1, deg2, 1
theCoeffs(0, i) = basis2(j)*theCoeffs(j, i) + theCoeffs(0, i)
end do

z = z + basis1(i)*theCoeffs(0, i)
end do

return
end function
!........................................

!........................................
pure subroutine eval_spline_2d_cross(xVec, yVec, kts1, deg1, kts2, deg2, &
      coeffs, z, der1, der2)

implicit none
real(kind=8), intent(in)  :: xVec (0:)
real(kind=8), intent(in)  :: yVec (0:)
real(kind=8), intent(in)  :: kts1 (0:)
integer(kind=4), intent(in)  :: deg1 
real(kind=8), intent(in)  :: kts2 (0:)
integer(kind=4), intent(in)  :: deg2 
real(kind=8), intent(in)  :: coeffs (0:,0:)
real(kind=8), intent(inout)  :: z (0:,0:)
integer(kind=4), intent(in)  :: der1 
integer(kind=4), intent(in)  :: der2 
real(kind=8) :: basis1 (0:deg1) 
real(kind=8) :: basis2 (0:deg2) 
real(kind=8) :: theCoeffs (0:deg2,0:deg1) 
integer(kind=4) :: i  
real(kind=8) :: x  
integer(kind=4) :: span1  
integer(kind=4) :: j  
real(kind=8) :: y  
integer(kind=4) :: span2  
integer(kind=4) :: k  
integer(kind=4) :: l  


basis1 = 0.0
basis2 = 0.0
theCoeffs = 0.0


if (der1 == 0  .and. der2 == 0 ) then
do i = 0, size(xVec,1) - 1, 1
x = xVec(i)
span1 = find_span(kts1, deg1, x)
call basis_funs(kts1, deg1, x, span1, basis1)
do j = 0, size(yVec,1) - 1, 1
y = yVec(j)
span2 = find_span(kts2, deg2, y)
call basis_funs(kts2, deg2, y, span2, basis2)


theCoeffs(:, :) = coeffs(-deg2 + span2:span2, -deg1 + span1:span1)


z(j, i) = 0.0d0
do k = 0, deg1, 1
theCoeffs(0, k) = basis2(0)*theCoeffs(0, k)
do l = 1, deg2, 1
theCoeffs(0, k) = basis2(l)*theCoeffs(l, k) + theCoeffs(0, k)
end do

z(j, i) = basis1(k)*theCoeffs(0, k) + z(j, i)
end do

end do

end do

else if (der1 == 0  .and. der2 == 1 ) then
do i = 0, size(xVec,1) - 1, 1
x = xVec(i)
span1 = find_span(kts1, deg1, x)
call basis_funs(kts1, deg1, x, span1, basis1)
do j = 0, size(yVec,1) - 1, 1
y = yVec(j)
span2 = find_span(kts2, deg2, y)
call basis_funs_1st_der(kts2, deg2, y, span2, basis2)


theCoeffs(:, :) = coeffs(-deg2 + span2:span2, -deg1 + span1:span1)


z(j, i) = 0.0d0
do k = 0, deg1, 1
theCoeffs(0, k) = basis2(0)*theCoeffs(0, k)
do l = 1, deg2, 1
theCoeffs(0, k) = basis2(l)*theCoeffs(l, k) + theCoeffs(0, k)
end do

z(j, i) = basis1(k)*theCoeffs(0, k) + z(j, i)
end do

end do

end do

else if (der1 == 1  .and. der2 == 0 ) then
do i = 0, size(xVec,1) - 1, 1
x = xVec(i)
span1 = find_span(kts1, deg1, x)
call basis_funs_1st_der(kts1, deg1, x, span1, basis1)
do j = 0, size(yVec,1) - 1, 1
y = yVec(j)
span2 = find_span(kts2, deg2, y)
call basis_funs(kts2, deg2, y, span2, basis2)


theCoeffs(:, :) = coeffs(-deg2 + span2:span2, -deg1 + span1:span1)


z(j, i) = 0.0d0
do k = 0, deg1, 1
theCoeffs(0, k) = basis2(0)*theCoeffs(0, k)
do l = 1, deg2, 1
theCoeffs(0, k) = basis2(l)*theCoeffs(l, k) + theCoeffs(0, k)
end do

z(j, i) = basis1(k)*theCoeffs(0, k) + z(j, i)
end do

end do

end do

else if (der1 == 1  .and. der2 == 1 ) then
do i = 0, size(xVec,1) - 1, 1
x = xVec(i)
span1 = find_span(kts1, deg1, x)
call basis_funs_1st_der(kts1, deg1, x, span1, basis1)
do j = 0, size(yVec,1) - 1, 1
y = yVec(j)
span2 = find_span(kts2, deg2, y)
call basis_funs_1st_der(kts2, deg2, y, span2, basis2)


theCoeffs(:, :) = coeffs(-deg2 + span2:span2, -deg1 + span1:span1)


z(j, i) = 0.0d0
do k = 0, deg1, 1
theCoeffs(0, k) = basis2(0)*theCoeffs(0, k)
do l = 1, deg2, 1
theCoeffs(0, k) = basis2(l)*theCoeffs(l, k) + theCoeffs(0, k)
end do

z(j, i) = basis1(k)*theCoeffs(0, k) + z(j, i)


end do

end do

end do

end if
end subroutine
!........................................

!........................................
pure subroutine eval_spline_2d_vector(x, y, kts1, deg1, kts2, deg2, &
      coeffs, z, der1, der2)

implicit none
real(kind=8), intent(in)  :: x (0:)
real(kind=8), intent(in)  :: y (0:)
real(kind=8), intent(in)  :: kts1 (0:)
integer(kind=4), intent(in)  :: deg1 
real(kind=8), intent(in)  :: kts2 (0:)
integer(kind=4), intent(in)  :: deg2 
real(kind=8), intent(in)  :: coeffs (0:,0:)
real(kind=8), intent(inout)  :: z (0:)
integer(kind=4), intent(in)  :: der1 
integer(kind=4), intent(in)  :: der2 
real(kind=8) :: basis1 (0:deg1) 
real(kind=8) :: basis2 (0:deg2) 
real(kind=8) :: theCoeffs (0:deg2,0:deg1) 
integer(kind=4) :: i  
integer(kind=4) :: span1  
integer(kind=4) :: span2  
integer(kind=4) :: j  
integer(kind=4) :: k  


basis1 = 0.0
basis2 = 0.0
theCoeffs = 0.0


if (der1 == 0 ) then
if (der2 == 0 ) then
do i = 0, size(x,1) - 1, 1
span1 = find_span(kts1, deg1, x(i))
span2 = find_span(kts2, deg2, y(i))
call basis_funs(kts1, deg1, x(i), span1, basis1)
call basis_funs(kts2, deg2, y(i), span2, basis2)


theCoeffs(:, :) = coeffs(-deg2 + span2:span2, -deg1 + span1:span1)


z(i) = 0.0d0
do j = 0, deg1, 1
theCoeffs(0, j) = basis2(0)*theCoeffs(0, j)
do k = 1, deg2, 1
theCoeffs(0, j) = basis2(k)*theCoeffs(k, j) + theCoeffs(0, j)
end do

z(i) = basis1(j)*theCoeffs(0, j) + z(i)
end do

end do

else if (der2 == 1 ) then
do i = 0, size(x,1) - 1, 1
span1 = find_span(kts1, deg1, x(i))
span2 = find_span(kts2, deg2, y(i))
call basis_funs(kts1, deg1, x(i), span1, basis1)
call basis_funs_1st_der(kts2, deg2, y(i), span2, basis2)


theCoeffs(:, :) = coeffs(-deg2 + span2:span2, -deg1 + span1:span1)


z(i) = 0.0d0
do j = 0, deg1, 1
theCoeffs(0, j) = basis2(0)*theCoeffs(0, j)
do k = 1, deg2, 1
theCoeffs(0, j) = basis2(k)*theCoeffs(k, j) + theCoeffs(0, j)
end do

z(i) = basis1(j)*theCoeffs(0, j) + z(i)
end do

end do

end if
else if (der1 == 1 ) then
if (der2 == 0 ) then
do i = 0, size(x,1) - 1, 1
span1 = find_span(kts1, deg1, x(i))
span2 = find_span(kts2, deg2, y(i))
call basis_funs_1st_der(kts1, deg1, x(i), span1, basis1)
call basis_funs(kts2, deg2, y(i), span2, basis2)


theCoeffs(:, :) = coeffs(-deg2 + span2:span2, -deg1 + span1:span1)


z(i) = 0.0d0
do j = 0, deg1, 1
theCoeffs(0, j) = basis2(0)*theCoeffs(0, j)
do k = 1, deg2, 1
theCoeffs(0, j) = basis2(k)*theCoeffs(k, j) + theCoeffs(0, j)
end do

z(i) = basis1(j)*theCoeffs(0, j) + z(i)
end do

end do

else if (der2 == 1 ) then
do i = 0, size(x,1) - 1, 1
span1 = find_span(kts1, deg1, x(i))
span2 = find_span(kts2, deg2, y(i))
call basis_funs_1st_der(kts1, deg1, x(i), span1, basis1)
call basis_funs_1st_der(kts2, deg2, y(i), span2, basis2)


theCoeffs(:, :) = coeffs(-deg2 + span2:span2, -deg1 + span1:span1)


z(i) = 0.0d0
do j = 0, deg1, 1
theCoeffs(0, j) = basis2(0)*theCoeffs(0, j)
do k = 1, deg2, 1
theCoeffs(0, j) = basis2(k)*theCoeffs(k, j) + theCoeffs(0, j)
end do

z(i) = basis1(j)*theCoeffs(0, j) + z(i)
end do

end do

end if
end if
end subroutine
!........................................

end module