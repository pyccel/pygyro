module mod_context_1

implicit none



contains

! ........................................
integer(kind=4) function find_span(knots, degree, x)  result(returnVal)

implicit none
real(kind=8), intent(in)  :: knots (0:)
integer(kind=4), intent(in)  :: degree
real(kind=8), intent(in)  :: x
integer(kind=4) :: high
integer(kind=4) :: span
integer(kind=4) :: low

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
! Knot index at left/right boundary
low = degree
high = 0
high = -degree - 1 + size(knots,1)


! Check if point is exactly on left/right boundary, or outside
!  domain
if (x <= knots(low)) then
  returnVal = low
else if (x >= knots(high)) then
  returnVal = high - 1
else
  ! Perform binary search
  span = Int(0.5d0*(high + low))
  do while (x >= knots(span + 1) .or. x < knots(span))
    if (x < knots(span)) then
      high = span
    else
      low = span
    end if
    span = Int(0.5d0*(high + low))
  end do
  returnVal = span


end if
return


! ============================================================
! ==================
end function
! ........................................

! ........................................
subroutine basis_funs(knots, degree, x, span, values)

implicit none
real(kind=8), intent(in)  :: knots (0:)
integer(kind=4), intent(in)  :: degree
real(kind=8), intent(in)  :: x
integer(kind=4), intent(in)  :: span
real(kind=8), intent(inout)  :: values (0:)
real(kind=8) :: saved
real(kind=8), allocatable :: right (:)
integer(kind=4) :: j
real(kind=8), allocatable :: left (:)
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
!    The original Algorithm A2.2 in The NURBS Book [1] is her!
!e                                                           !
!    slightly improved by using 'left' and 'right' temporary !
!    arrays that are one element shorter.                    !
!                                                            !
!                                                            !
!____________________________________________________________!

allocate(left(0:degree - 1))
allocate(right(0:degree - 1))


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
! ........................................

! ........................................
subroutine basis_funs_1st_der(knots, degree, x, span, ders)

implicit none
real(kind=8), intent(in)  :: knots (0:)
integer(kind=4), intent(in)  :: degree
real(kind=8), intent(in)  :: x
integer(kind=4), intent(in)  :: span
real(kind=8), intent(inout)  :: ders (0:)
real(kind=8), allocatable :: values (:)
real(kind=8) :: saved
integer(kind=4) :: j
real(kind=8) :: temp

!_________________________CommentBlock_________________________!
!                                                              !
!    Compute the first derivative of the non-vanishing B-spli  !
!nes                                                           !
!    at location x, given the knot sequence, polynomial degre  !
!e                                                             !
!    and knot span.                                            !
!                                                              !
!    See function 's_bsplines_non_uniform__eval_deriv' in      !
!    Selalib's source file                                     !
!    'src/splines/sll_m_bsplines_non_uniform.F90'.             !
!                                                              !
!    Parameters                                                !
!    ----------                                                !
!    knots : array_like                                        !
!        Knots sequence.                                       !
!                                                              !
!    degree : int                                              !
!        Polynomial degree of B-splines.                       !
!                                                              !
!    x : float                                                 !
!        Evaluation point.                                     !
!                                                              !
!    span : int                                                !
!        Knot span index.                                      !
!                                                              !
!    Results                                                   !
!    -------                                                   !
!    ders : numpy.ndarray                                      !
!        Derivatives of p+1 non-vanishing B-Splines at location!
! x.                                                           !
!                                                              !
!                                                              !
!______________________________________________________________!
! Compute nonzero basis functions and knot differences for spl
! ines
! up to degree deg-1

allocate(values(0:degree - 1))
call basis_funs(knots, degree - 1, x, span, values)


! Compute derivatives at x using formula based on difference o
! f
! splines of degree deg-1
! -------
! j = 0
saved = degree*(values(0)/(knots(span + 1) - knots(span - degree + 1)))
ders(0) = -saved
! j = 1,...,degree-1
do j = 1, degree - 1, 1
  temp = saved
  saved = degree*(values(j)/(knots(span + j + 1) - knots(span + j - &
      degree + 1)))
  ders(j) = -saved + temp
end do

! j = degree
ders(degree) = saved
end subroutine
! ........................................

end module