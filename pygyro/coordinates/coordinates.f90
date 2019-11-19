module coordinates

implicit none




contains

!........................................
pure subroutine logical_to_pseudocart(r, theta, x, y) 

implicit none
real(kind=8), intent(out)  :: x 
real(kind=8), intent(out)  :: y 
real(kind=8), intent(in)  :: r 
real(kind=8), intent(in)  :: theta 

x = r*cos(theta)
y = r*sin(theta)


return
end subroutine
!........................................

!........................................
pure subroutine pseudocart_to_logical(x, y, r, theta) 

implicit none
real(kind=8), intent(out)  :: theta 
real(kind=8), intent(out)  :: r 
real(kind=8), intent(in)  :: x 
real(kind=8), intent(in)  :: y 

r = sqrt(x*x + y*y)
theta = atan2(y,x)
if (theta < 0) then
theta = 2*3.14159265358979d0 + theta
end if
return
end subroutine
!........................................

!........................................
pure subroutine function_logical_to_pseudocart(f_r, f_theta, r, theta, &
      g_x, g_y)

implicit none
real(kind=8), intent(out)  :: g_x 
real(kind=8), intent(out)  :: g_y 
real(kind=8), intent(in)  :: f_r 
real(kind=8), intent(in)  :: f_theta 
real(kind=8), intent(in)  :: r 
real(kind=8), intent(in)  :: theta 

g_x = f_r*cos(theta) + (-r)*(f_theta*sin(theta))
g_y = f_r*sin(theta) + r*(f_theta*cos(theta))
return
end subroutine
!........................................

!........................................
pure subroutine function_pseudocart_to_logical(f_x, f_y, r, theta, g_r, &
      g_theta)

implicit none
real(kind=8), intent(out)  :: g_r 
real(kind=8), intent(out)  :: g_theta 
real(kind=8), intent(in)  :: f_x 
real(kind=8), intent(in)  :: f_y 
real(kind=8), intent(in)  :: r 
real(kind=8), intent(in)  :: theta 

g_r = f_x*cos(theta) + f_y*sin(theta)
g_theta = (f_x*(-sin(theta)) + f_y*cos(theta))/r


return
end subroutine
!........................................

end module
