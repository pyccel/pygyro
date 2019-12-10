import numpy                as np

def pytest_generate_tests(metafunc):
    if 'param_df_poly' in metafunc.fixturenames:
        param_df_poly = []
        if metafunc.config.getoption('short'):
            for npts,tol in zip([10,20,50],[0.002,0.0003,2e-6]):
                for _ in range( 2 ):
                    coeffs = np.random.random(4)*3
                    param_df_poly.append( (npts, coeffs,tol) )
        else:
            for npts,tol in zip([10,20,50],[0.002,0.0003,2e-6]):
                for _ in range( 5 ):
                    coeffs = np.random.random(4)*3
                    param_df_poly.append( (npts, coeffs,tol) )
        metafunc.parametrize("param_df_poly", param_df_poly)
    if 'param_poisson_dirichlet' in metafunc.fixturenames:
        if metafunc.config.getoption('short'):
            param_poisson_dirichlet = [(1,4,0.3),(1,32,0.01),(2,6,0.1),
                                            (2,32,0.1),(3,9,0.03)]
        else:
            param_poisson_dirichlet = [(1,4,0.3),(1,32,0.01),(2,6,0.1),
                                          (2,32,0.1),(3,9,0.03),(3,32,0.02),
                                          (4,10,0.02),(4,40,0.02),(5,14,0.01),
                                          (5,64,0.01)]
        metafunc.parametrize("param_poisson_dirichlet", param_poisson_dirichlet)
    if 'param_poisson_neumann' in metafunc.fixturenames:
        if metafunc.config.getoption('short'):
            param_poisson_neumann = [(1,4,10),(1,32,0.09),(2,6,1e-12),
                      (3,9,1e-12)]
        else:
            param_poisson_neumann = [(1,4,10),(1,32,0.09),(2,6,1e-12),
                      (2,32,1e-12),(3,9,1e-12),(3,32,1e-12),
                      (4,10,1e-12),(4,40,1e-11),(5,14,1e-12),
                      (5,64,1e-11)]
        metafunc.parametrize("param_poisson_neumann", param_poisson_neumann)
    if 'param_grad' in metafunc.fixturenames:
        if metafunc.config.getoption('short'):
            param_grad = [(1,4,0.9),(1,32,0.07),(2,6,0.3),
                      (2,32,0.05),(3,10,0.2)]
        else:
            param_grad = [(1,4,0.9),(1,32,0.07),(2,6,0.3),
                      (2,32,0.05),(3,10,0.2),(3,32,0.04),
                      (4,10,0.2),(4,40,0.03),(5,14,0.09),
                      (5,64,0.02)]
        metafunc.parametrize("param_grad", param_grad)
    if 'param_grad_r' in metafunc.fixturenames:
        if metafunc.config.getoption('short'):
            param_grad_r = [(1,32,0.2),(1,128,0.02),(2,32,0.09),
                      (3,32,0.08)]
        else:
            param_grad_r = [(1,32,0.2),(1,256,0.02),(2,32,0.09),
                      (2,256,0.02),(3,32,0.08),(3,256,0.009),
                      (4,32,0.07),(4,256,0.007),(5,32,0.06),
                      (5,256,0.006)]
        metafunc.parametrize("param_grad_r", param_grad_r)
    if 'param_sin_sin' in metafunc.fixturenames:
        if metafunc.config.getoption('short'):
            param_sin_sin = [(1,4,2),(1,32,0.2),(1,64,0.03),(2,6,1.1),
                              (2,32,0.03),(3,10,0.3),(3,32,0.02)]
        else:
            param_sin_sin = [(1,4,2),(1,32,0.2),(1,64,0.03),(2,6,1.1),
                              (2,32,0.03),(3,10,0.3),(3,32,0.02),
                              (4,10,0.3),(4,40,0.008),(5,14,0.09),
                              (5,64,0.003)]
        metafunc.parametrize("param_sin_sin", param_sin_sin)
    if 'param_fft' in metafunc.fixturenames:
        if metafunc.config.getoption('short'):
            param_fft = [(1,4),(1,32),(1,64),(2,6),
                          (2,32),(3,10),(3,32)]
        else:
            param_fft = [(1,4),(1,32),(1,64),(2,6),
                          (2,32),(3,10),(3,32),
                          (4,10),(4,40),(5,14),
                          (5,64)]
        metafunc.parametrize("param_fft", param_fft)
