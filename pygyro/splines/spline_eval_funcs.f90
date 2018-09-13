module mod_spline_eval_funcs

use mod_context_1, only: find_span
use mod_context_1, only: basis_funs
use mod_context_1, only: basis_funs_1st_der
implicit none



contains

! ........................................
subroutine eval_spline_1d_scalar(x, n0_knots, knots, degree, n0_coeffs, &
      coeffs, der, y)

  implicit none
  integer(kind=4) :: j
  integer(kind=4), intent(in)  :: n0_knots
  real(kind=8), intent(out)  :: y
  integer(kind=4), intent(in)  :: der
  real(kind=8), intent(in)  :: coeffs (0:n0_coeffs - 1)
  integer(kind=4), intent(in)  :: n0_coeffs
  real(kind=8), allocatable :: basis (:)
  integer(kind=4) :: span
  real(kind=8), intent(in)  :: knots (0:n0_knots - 1)
  real(kind=8), intent(in)  :: x
  integer(kind=4), intent(in)  :: degree

  span = find_span(knots, degree, x)



  allocate(basis(0:degree))
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


end subroutine
! ........................................

! ........................................
subroutine eval_spline_1d_vector(n0_x, x, n0_knots, knots, degree, &
      n0_coeffs, coeffs, n0_y, y, der)

  implicit none
  real(kind=8), intent(in)  :: knots (0:n0_knots - 1)
  integer(kind=4), intent(in)  :: n0_knots
  integer(kind=4), intent(in)  :: n0_x
  real(kind=8), allocatable :: basis (:)
  integer(kind=4) :: span
  integer(kind=4) :: j
  integer(kind=4) :: i
  real(kind=8), intent(inout)  :: y (0:n0_y - 1)
  real(kind=8), intent(in)  :: x (0:n0_x - 1)
  real(kind=8), intent(in)  :: coeffs (0:n0_coeffs - 1)
  integer(kind=4), intent(in)  :: n0_y
  integer(kind=4), intent(in)  :: n0_coeffs
  integer(kind=4), intent(in)  :: degree
  integer(kind=4), intent(in)  :: der


  if (der == 0 ) then
    allocate(basis(0:degree))
    do i = 0, size(x,1) - 1, 1
      span = find_span(knots, degree, x(i))
      call basis_funs(knots, degree, x(i), span, basis)


      y(i) = 0.0d0
      do j = 0, degree, 1
        y(i) = basis(j)*coeffs(span - degree + j) + y(i)
      end do

    end do

  else if (der == 1 ) then
    allocate(basis(0:degree))
    do i = 0, size(x,1) - 1, 1
      span = find_span(knots, degree, x(i))
      call basis_funs(knots, degree, x(i), span, basis)
      call basis_funs_1st_der(knots, degree, x(i), span, basis)


      y(i) = 0.0d0
      do j = 0, degree, 1
        y(i) = basis(j)*coeffs(span - degree + j) + y(i)
      end do

    end do

  end if
  return


end subroutine
! ........................................

! ........................................
subroutine eval_spline_2d_scalar(x, y, n0_kts1, kts1, deg1, n0_kts2, &
      kts2, deg2, n0_coeffs, n1_coeffs, coeffs, der1, der2, z)

  implicit none
  real(kind=8), allocatable :: basis1 (:)
  integer(kind=4), intent(in)  :: n1_coeffs
  real(kind=8), intent(in)  :: kts1 (0:n0_kts1 - 1)
  real(kind=8), allocatable :: theCoeffs (:,:)
  integer(kind=4), intent(in)  :: der1
  integer(kind=4) :: span2
  integer(kind=4), intent(in)  :: der2
  real(kind=8), intent(in)  :: x
  integer(kind=4) :: i
  real(kind=8), intent(out)  :: z
  integer(kind=4) :: j
  integer(kind=4) :: span1
  real(kind=8), intent(in)  :: y
  real(kind=8), intent(in)  :: kts2 (0:n0_kts2 - 1)
  integer(kind=4), intent(in)  :: deg2
  integer(kind=4), intent(in)  :: deg1
  real(kind=8), allocatable :: basis2 (:)
  integer(kind=4), intent(in)  :: n0_kts1
  integer(kind=4), intent(in)  :: n0_coeffs
  integer(kind=4), intent(in)  :: n0_kts2
  real(kind=8), intent(in)  :: coeffs (0:n0_coeffs - 1,0:n1_coeffs - 1)

  span1 = find_span(kts1, deg1, x)
  span2 = find_span(kts2, deg2, y)



  allocate(basis1(0:deg1))
  allocate(basis2(0:deg2))
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
  allocate(theCoeffs(0:deg2, 0:deg1))
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


end subroutine
! ........................................

! ........................................
subroutine eval_spline_2d_cross(n0_xVec, xVec, n0_yVec, yVec, n0_kts1, &
      kts1, deg1, n0_kts2, kts2, deg2, n0_coeffs, n1_coeffs, coeffs, &
      n0_z, n1_z, z, der1, der2)

  implicit none
  real(kind=8), allocatable :: basis1 (:)
  integer(kind=4), intent(in)  :: n1_coeffs
  real(kind=8), intent(in)  :: kts1 (0:n0_kts1 - 1)
  real(kind=8), intent(in)  :: xVec (0:n0_xVec - 1)
  integer(kind=4), intent(in)  :: der2
  integer(kind=4) :: span2
  integer(kind=4), intent(in)  :: der1
  real(kind=8) :: y
  integer(kind=4), intent(in)  :: n0_yVec
  integer(kind=4) :: i
  real(kind=8), intent(in)  :: coeffs (0:n0_coeffs - 1,0:n1_coeffs - 1)
  real(kind=8), intent(in)  :: yVec (0:n0_yVec - 1)
  integer(kind=4) :: l
  integer(kind=4) :: j
  real(kind=8), intent(in)  :: kts2 (0:n0_kts2 - 1)
  integer(kind=4), intent(in)  :: deg2
  integer(kind=4) :: k
  integer(kind=4), intent(in)  :: n0_kts1
  integer(kind=4), intent(in)  :: n0_xVec
  integer(kind=4), intent(in)  :: n0_kts2
  real(kind=8) :: x
  real(kind=8), allocatable :: theCoeffs (:,:)
  integer(kind=4), intent(in)  :: n0_z
  real(kind=8), intent(inout)  :: z (0:n0_z - 1,0:n1_z - 1)
  integer(kind=4), intent(in)  :: n1_z
  integer(kind=4) :: span1
  integer(kind=4), intent(in)  :: deg1
  real(kind=8), allocatable :: basis2 (:)
  integer(kind=4), intent(in)  :: n0_coeffs


  allocate(basis1(0:deg1))
  allocate(basis2(0:deg2))
  allocate(theCoeffs(0:deg2, 0:deg1))


  if (der1 == 0  .and. der2 == 0 ) then
    do i = 0, n0_xVec - 1, 1
      x = xVec(i)
      span1 = find_span(kts1, deg1, x)
      call basis_funs(kts1, deg1, x, span1, basis1)
      do j = 0, n0_yVec - 1, 1
        y = yVec(j)
        span2 = find_span(kts2, deg2, y)
        call basis_funs(kts2, deg2, y, span2, basis2)


        theCoeffs(:, :) = coeffs(-deg2 + span2:span2, -deg1 + span1: &
      span1)


        z(j, i) = 0.0d0
        do k = 0, deg1, 1
          theCoeffs(0, k) = basis2(0)*theCoeffs(0, k)
          do l = 1, deg2, 1
            theCoeffs(0, k) = basis2(l)*theCoeffs(l, k) + theCoeffs(0, k &
      )
          end do

          z(j, i) = basis1(k)*theCoeffs(0, k) + z(j, i)
        end do

      end do

    end do

  else if (der1 == 0  .and. der2 == 1 ) then
    do i = 0, n0_xVec - 1, 1
      x = xVec(i)
      span1 = find_span(kts1, deg1, x)
      call basis_funs(kts1, deg1, x, span1, basis1)
      do j = 0, n0_yVec - 1, 1
        y = yVec(j)
        span2 = find_span(kts2, deg2, y)
        call basis_funs_1st_der(kts2, deg2, y, span2, basis2)


        theCoeffs(:, :) = coeffs(-deg2 + span2:span2, -deg1 + span1: &
      span1)


        z(j, i) = 0.0d0
        do k = 0, deg1, 1
          theCoeffs(0, k) = basis2(0)*theCoeffs(0, k)
          do l = 1, deg2, 1
            theCoeffs(0, k) = basis2(l)*theCoeffs(l, k) + theCoeffs(0, k &
      )
          end do

          z(j, i) = basis1(k)*theCoeffs(0, k) + z(j, i)
        end do

      end do

    end do

  else if (der1 == 1  .and. der2 == 0 ) then
    do i = 0, n0_xVec - 1, 1
      x = xVec(i)
      span1 = find_span(kts1, deg1, x)
      call basis_funs_1st_der(kts1, deg1, x, span1, basis1)
      do j = 0, n0_yVec - 1, 1
        y = yVec(j)
        span2 = find_span(kts2, deg2, y)
        call basis_funs(kts2, deg2, y, span2, basis2)


        theCoeffs(:, :) = coeffs(-deg2 + span2:span2, -deg1 + span1: &
      span1)


        z(j, i) = 0.0d0
        do k = 0, deg1, 1
          theCoeffs(0, k) = basis2(0)*theCoeffs(0, k)
          do l = 1, deg2, 1
            theCoeffs(0, k) = basis2(l)*theCoeffs(l, k) + theCoeffs(0, k &
      )
          end do

          z(j, i) = basis1(k)*theCoeffs(0, k) + z(j, i)
        end do

      end do

    end do

  else if (der1 == 1  .and. der2 == 1 ) then
    do i = 0, n0_xVec - 1, 1
      x = xVec(i)
      span1 = find_span(kts1, deg1, x)
      call basis_funs_1st_der(kts1, deg1, x, span1, basis1)
      do j = 0, n0_yVec - 1, 1
        y = yVec(j)
        span2 = find_span(kts2, deg2, y)
        call basis_funs_1st_der(kts2, deg2, y, span2, basis2)


        theCoeffs(:, :) = coeffs(-deg2 + span2:span2, -deg1 + span1: &
      span1)


        z(j, i) = 0.0d0
        do k = 0, deg1, 1
          theCoeffs(0, k) = basis2(0)*theCoeffs(0, k)
          do l = 1, deg2, 1
            theCoeffs(0, k) = basis2(l)*theCoeffs(l, k) + theCoeffs(0, k &
      )
          end do

          z(j, i) = basis1(k)*theCoeffs(0, k) + z(j, i)


        end do

      end do

    end do

  end if
end subroutine
! ........................................

! ........................................
subroutine eval_spline_2d_vector(n0_x, x, n0_y, y, n0_kts1, kts1, deg1, &
      n0_kts2, kts2, deg2, n0_coeffs, n1_coeffs, coeffs, n0_z, z, der1, &
      der2)

  implicit none
  real(kind=8), allocatable :: basis1 (:)
  integer(kind=4), intent(in)  :: n1_coeffs
  real(kind=8), allocatable :: theCoeffs (:,:)
  integer(kind=4), intent(in)  :: der1
  integer(kind=4) :: span2
  integer(kind=4), intent(in)  :: der2
  integer(kind=4), intent(in)  :: n0_kts2
  integer(kind=4) :: i
  real(kind=8), intent(in)  :: coeffs (0:n0_coeffs - 1,0:n1_coeffs - 1)
  real(kind=8), intent(in)  :: kts1 (0:n0_kts1 - 1)
  real(kind=8), intent(in)  :: x (0:n0_x - 1)
  real(kind=8), intent(in)  :: kts2 (0:n0_kts2 - 1)
  integer(kind=4), intent(in)  :: deg2
  integer(kind=4) :: k
  integer(kind=4), intent(in)  :: n0_kts1
  integer(kind=4), intent(in)  :: n0_y
  integer(kind=4), intent(in)  :: n0_x
  integer(kind=4) :: j
  integer(kind=4), intent(in)  :: n0_z
  real(kind=8), intent(inout)  :: z (0:n0_z - 1)
  real(kind=8), intent(in)  :: y (0:n0_y - 1)
  integer(kind=4) :: span1
  integer(kind=4), intent(in)  :: deg1
  real(kind=8), allocatable :: basis2 (:)
  integer(kind=4), intent(in)  :: n0_coeffs


  allocate(theCoeffs(0:deg2, 0:deg1))


  if (der1 == 0 ) then
    if (der2 == 0 ) then
      allocate(basis1(0:deg1))
      allocate(basis2(0:deg2))
      do i = 0, size(x,1) - 1, 1
        span1 = find_span(kts1, deg1, x(i))
        span2 = find_span(kts2, deg2, y(i))
        call basis_funs(kts1, deg1, x(i), span1, basis1)
        call basis_funs(kts2, deg2, y(i), span2, basis2)


        theCoeffs(:, :) = coeffs(-deg2 + span2:span2, -deg1 + span1: &
      span1)


        z(i) = 0.0d0
        do j = 0, deg1, 1
          theCoeffs(0, j) = basis2(0)*theCoeffs(0, j)
          do k = 1, deg2, 1
            theCoeffs(0, j) = basis2(k)*theCoeffs(k, j) + theCoeffs(0, j &
      )
          end do

          z(i) = basis1(j)*theCoeffs(0, j) + z(i)
        end do

      end do

    else if (der2 == 1 ) then
      allocate(basis1(0:deg1))
      allocate(basis2(0:deg2))
      do i = 0, size(x,1) - 1, 1
        span1 = find_span(kts1, deg1, x(i))
        span2 = find_span(kts2, deg2, y(i))
        call basis_funs(kts1, deg1, x(i), span1, basis1)
        call basis_funs_1st_der(kts2, deg2, y(i), span2, basis2)


        theCoeffs(:, :) = coeffs(-deg2 + span2:span2, -deg1 + span1: &
      span1)


        z(i) = 0.0d0
        do j = 0, deg1, 1
          theCoeffs(0, j) = basis2(0)*theCoeffs(0, j)
          do k = 1, deg2, 1
            theCoeffs(0, j) = basis2(k)*theCoeffs(k, j) + theCoeffs(0, j &
      )
          end do

          z(i) = basis1(j)*theCoeffs(0, j) + z(i)
        end do

      end do

    end if
  else if (der1 == 1 ) then
    if (der2 == 0 ) then
      allocate(basis1(0:deg1))
      allocate(basis2(0:deg2))
      do i = 0, size(x,1) - 1, 1
        span1 = find_span(kts1, deg1, x(i))
        span2 = find_span(kts2, deg2, y(i))
        call basis_funs_1st_der(kts1, deg1, x(i), span1, basis1)
        call basis_funs(kts2, deg2, y(i), span2, basis2)


        theCoeffs(:, :) = coeffs(-deg2 + span2:span2, -deg1 + span1: &
      span1)


        z(i) = 0.0d0
        do j = 0, deg1, 1
          theCoeffs(0, j) = basis2(0)*theCoeffs(0, j)
          do k = 1, deg2, 1
            theCoeffs(0, j) = basis2(k)*theCoeffs(k, j) + theCoeffs(0, j &
      )
          end do

          z(i) = basis1(j)*theCoeffs(0, j) + z(i)
        end do

      end do

    else if (der2 == 1 ) then
      allocate(basis1(0:deg1))
      allocate(basis2(0:deg2))
      do i = 0, size(x,1) - 1, 1
        span1 = find_span(kts1, deg1, x(i))
        span2 = find_span(kts2, deg2, y(i))
        call basis_funs_1st_der(kts1, deg1, x(i), span1, basis1)
        call basis_funs_1st_der(kts2, deg2, y(i), span2, basis2)


        theCoeffs(:, :) = coeffs(-deg2 + span2:span2, -deg1 + span1: &
      span1)


        z(i) = 0.0d0
        do j = 0, deg1, 1
          theCoeffs(0, j) = basis2(0)*theCoeffs(0, j)
          do k = 1, deg2, 1
            theCoeffs(0, j) = basis2(k)*theCoeffs(k, j) + theCoeffs(0, j &
      )
          end do

          z(i) = basis1(j)*theCoeffs(0, j) + z(i)
        end do

      end do

    end if
  end if
end subroutine
! ........................................

end module