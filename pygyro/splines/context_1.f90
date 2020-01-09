module context_1

implicit none




contains

!........................................
subroutine find_span(n0_knots, knots, degree, x, returnVal) 

  implicit none
  integer(kind=4), intent(out)  :: returnVal 
  integer(kind=4), intent(in)  :: n0_knots 
  real(kind=8), intent(in)  :: knots (0:n0_knots - 1)
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
end subroutine
!........................................

!........................................
subroutine basis_funs(n0_knots, knots, degree, x, span, n0_values, &
      values)

  implicit none
  integer(kind=4), intent(in)  :: n0_knots 
  real(kind=8), intent(in)  :: knots (0:n0_knots - 1)
  integer(kind=4), intent(in)  :: degree 
  real(kind=8), intent(in)  :: x 
  integer(kind=4), intent(in)  :: span 
  integer(kind=4), intent(in)  :: n0_values 
  real(kind=8), intent(inout)  :: values (0:n0_values - 1)
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

end module