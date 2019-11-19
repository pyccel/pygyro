from pyccel.decorators import pure, external, types
from numpy import cos, sin, sqrt, arctan2, pi

@external
@pure
@types('double','double')
def logical_to_pseudocart(r,theta):
    x = r * cos( theta )
    y = r * sin( theta )

    return x,y

@external
@pure
@types('double','double')
def pseudocart_to_logical(x,y):
    r = sqrt( x * x + y * y )
    theta = arctan2( y, x )
    #if theta < 0:
    #   theta = theta + 2 * pi

    return r, theta

@external
@pure
@types('double','double','double','double')
def function_logical_to_pseudocart(f_r,f_theta,r,theta):
    g_x = cos(theta) * f_r - r * sin(theta) * f_theta
    g_y = sin(theta) * f_r + r * cos(theta) * f_theta
    return g_x, g_y

@external
@pure
@types('double','double','double','double')
def function_pseudocart_to_logical(f_x,f_y,r,theta):
    g_r     =    cos(theta) * f_x + sin(theta) * f_y
    g_theta = (- sin(theta) * f_x + cos(theta) * f_y) / r

    return g_r, g_theta
