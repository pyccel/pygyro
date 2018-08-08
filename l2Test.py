from mpi4py import MPI
from pygyro.initialisation.setups       import setupCylindricalGrid

comm = MPI.COMM_WORLD

npts = [20,20,10,8]

dt=1
tEnd = 10

tN = tEnd//dt

distribFunc = setupCylindricalGrid(npts   = npts,
                            layout = 'flux_surface',
                            comm   = comm,
                            allocateSaveMemory = True)

fluxAdv = fluxSurfaceAdvection(grid.eta_grid, grid.get2DSpline())
vParAdv = vParallelAdvection(grid.eta_grid, grid.getSpline(3))
polAdv = poloidalAdvection(grid.eta_grid, grid.getSpline(slice(1,None,-1)))

halfStep = dt*0.5

layout_poisson   = {'v_parallel': [0,2,1],
                    'mode_solve': [2,1,0]}
layout_advection = {'dphi'      : [0,1,2],
                    'poloidal'  : [2,1,0],
                    'v_parallel': [0,2,1]}

nprocs = distribFunc.currentLayout.nprocs

remapperPhi = LayoutSwapper( comm, [layout_poisson, layout_advection],
                            [nprocs,nprocs[0]], distribFunc.eta_grid[:3],
                            'mode_solve' )
remapperRho = getLayoutHandler( comm, layout_poisson, nprocs, distribFunc.eta_grid[:3] )

phi = Grid(distribFunc.eta_grid[:3],distribFunc.getSpline(slice(0,3)),
            remapperPhi,'mode_solve',comm,dtype=np.complex128)
rho = Grid(distribFunc.eta_grid[:3],distribFunc.getSpline(slice(0,3)),
            remapperRho,'v_parallel',comm,dtype=np.complex128)

density = DensityFinder(6,distribFunc.getSpline(0))

QNSolver = QuasiNeutralitySolver(distribFunc.eta_grid[:3],7,distribFunc.getSpline(0))

for ti in range(tN):
    # Find phi from f^n by solving QN eq
    density.getPerturbedRho(distribFunc,rho)
    QNSolver.getModes(rho)
    rho.setLayout('mode_solve')
    QNSolver.solveEquation(phi,rho)
    phi.setLayout('v_parallel')
    QNSolver.findPotential(phi)
    
    # Compute f^n+.5 using lie splitting
    distribFunc.saveGridValues()
    for i,r in grid.getCoords(0):
        for j,v in grid.getCoords(1):
            fluxAdv.step(grid.get2DSlice([i,j]),halfStep,v)
    grid.setLayout('v_parallel')
    for i,r in grid.getCoords(0):
        for j,z in grid.getCoords(1):
            for k,q in grid.getCoords(2):
                vParAdv.step(grid.get1DSlice([i,j,k]),halfStep,0,r)
    grid.setLayout('poloidal')
    for i,v in grid.getCoords(0):
        for j,z in grid.getCoords(1):
            polAdv.step(grid.get2DSlice([i,j]),halfStep,phi,v)
