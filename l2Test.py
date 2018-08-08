from mpi4py import MPI

import numpy    as np

from pygyro.model.layout                import LayoutSwapper, getLayoutHandler
from pygyro.model.grid                  import Grid
from pygyro.initialisation.setups       import setupCylindricalGrid
from pygyro.advection.advection         import FluxSurfaceAdvection, VParallelAdvection, PoloidalAdvection, ParallelGradient
from pygyro.poisson.poisson_solver      import DensityFinder, QuasiNeutralitySolver

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

npts = [20,20,10,8]

tEnd = 10
tN = 10

dt=tEnd/tN

halfStep = dt*0.5

distribFunc = setupCylindricalGrid(npts   = npts,
                            layout = 'v_parallel',
                            comm   = comm,
                            allocateSaveMemory = True)

fluxAdv = FluxSurfaceAdvection(distribFunc.eta_grid, distribFunc.get2DSpline(),
                                distribFunc.getLayout('flux_surface'),halfStep)
vParAdv = VParallelAdvection(distribFunc.eta_grid, distribFunc.getSpline(3))
polAdv = PoloidalAdvection(distribFunc.eta_grid, distribFunc.getSpline(slice(1,None,-1)))
parGrad = ParallelGradient(distribFunc.getSpline(1),distribFunc.eta_grid)
parGradVals = np.empty([npts[2],npts[1]])

layout_poisson   = {'v_parallel_2d': [0,2,1],
                    'mode_solve'   : [1,2,0]}
layout_advection = {'poloidal'     : [2,1,0],
                    'intermediate' : [0,1,2],
                    'v_parallel_1d': [0,2,1]}

nprocs = distribFunc.getLayout(distribFunc.currentLayout).nprocs[:-1]

remapperPhi = LayoutSwapper( comm, [layout_poisson, layout_advection],
                            [nprocs,nprocs[0]], distribFunc.eta_grid[:3],
                            'mode_solve' )
remapperRho = getLayoutHandler( comm, layout_poisson, nprocs, distribFunc.eta_grid[:3] )

phi = Grid(distribFunc.eta_grid[:3],distribFunc.getSpline(slice(0,3)),
            remapperPhi,'mode_solve',comm,dtype=np.complex128)
rho = Grid(distribFunc.eta_grid[:3],distribFunc.getSpline(slice(0,3)),
            remapperRho,'v_parallel_2d',comm,dtype=np.complex128)

density = DensityFinder(6,distribFunc.getSpline(0))

QNSolver = QuasiNeutralitySolver(distribFunc.eta_grid[:3],7,distribFunc.getSpline(0),
                                chi=0)

l2Phi = np.zeros(tN+1)

r = phi.eta_grid[0]
q = phi.eta_grid[1]
z = phi.eta_grid[2]
dr = np.array([r[0], *(r[:-1]+r[1:]), r[-1]])*0.5
dq = q[1]-q[0]
dz = z[1]-z[0]

rCalc = (r[phi.getLayout('v_parallel_2d').starts[0]:phi.getLayout('v_parallel_2d').ends[0]])[:,None,None]
drCalc = (dr[phi.getLayout('v_parallel_2d').starts[0]:phi.getLayout('v_parallel_2d').ends[0]])[:,None,None]

for ti in range(tN):
    # Find phi from f^n by solving QN eq
    distribFunc.setLayout('v_parallel')
    density.getPerturbedRho(distribFunc,rho)
    QNSolver.getModes(rho)
    rho.setLayout('mode_solve')
    phi.setLayout('mode_solve')
    QNSolver.solveEquation(phi,rho)
    phi.setLayout('v_parallel_2d')
    rho.setLayout('v_parallel_2d')
    QNSolver.findPotential(phi)
    
    # Calculate diagnostic quantity |phi|_2
    l2[ti]=np.sum(phi._f**2*drCalc*dq*dz*rCalc)
    
    # Compute f^n+1/2 using lie splitting
    distribFunc.setLayout('flux_surface')
    distribFunc.saveGridValues()
    for i,r in distribFunc.getCoords(0):
        for j,v in distribFunc.getCoords(1):
            fluxAdv.step(distribFunc.get2DSlice([i,j]),halfStep,v)
    distribFunc.setLayout('v_parallel')
    phi.setLayout('v_parallel_1d')
    for i,r in distribFunc.getCoords(0):
        parGrad.parallel_gradient(phi.get2DSlice([i]),i,parGradVals)
        for j,z in distribFunc.getCoords(1):
            for k,q in distribFunc.getCoords(2):
                vParAdv.step(distribFunc.get1DSlice([i,j,k]),halfStep,parGradVals[j,k],r)
    distribFunc.setLayout('poloidal')
    for i,v in distribFunc.getCoords(0):
        for j,z in distribFunc.getCoords(1):
            polAdv.step(distribFunc.get2DSlice([i,j]),halfStep,phi,v)
    
    # Find phi from f^n+1/2 by solving QN eq again
    distribFunc.setLayout('v_parallel')
    density.getPerturbedRho(distribFunc,rho)
    QNSolver.getModes(rho)
    rho.setLayout('mode_solve')
    phi.setLayout('mode_solve')
    QNSolver.solveEquation(phi,rho)
    phi.setLayout('v_parallel')
    rho.setLayout('v_parallel')
    QNSolver.findPotential(phi)
    
    # Compute f^n+1 using strang splitting
    distribFunc.restoreGridValues() # restored from flux_surface layout
    for i,r in distribFunc.getCoords(0):
        for j,v in distribFunc.getCoords(1):
            fluxAdv.step(distribFunc.get2DSlice([i,j]),halfStep,v)
    distribFunc.setLayout('v_parallel')
    phi.setLayout('v_parallel_1d')
    for i,r in distribFunc.getCoords(0):
        parGrad.parallel_gradient(phi.get2DSlice([i]),i,parGradVals)
        for j,z in distribFunc.getCoords(1):
            for k,q in distribFunc.getCoords(2):
                vParAdv.step(distribFunc.get1DSlice([i,j,k]),halfStep,0,r)
    distribFunc.setLayout('poloidal')
    phi.setLayout('poloidal')
    for i,v in distribFunc.getCoords(0):
        for j,z in distribFunc.getCoords(1):
            polAdv.step(distribFunc.get2DSlice([i,j]),dt,phi,v)
    distribFunc.setLayout('v_parallel')
    for i,r in distribFunc.getCoords(0):
        for j,z in distribFunc.getCoords(1):
            for k,q in distribFunc.getCoords(2):
                vParAdv.step(distribFunc.get1DSlice([i,j,k]),halfStep,0,r)
    distribFunc.setLayout('flux_surface')
    for i,r in distribFunc.getCoords(0):
        for j,v in distribFunc.getCoords(1):
            fluxAdv.step(distribFunc.get2DSlice([i,j]),halfStep,v)

# Find phi from f^n by solving QN eq
distribFunc.setLayout('v_parallel')
density.getPerturbedRho(distribFunc,rho)
QNSolver.getModes(rho)
rho.setLayout('mode_solve')
phi.setLayout('mode_solve')
QNSolver.solveEquation(phi,rho)
phi.setLayout('v_parallel_2d')
rho.setLayout('v_parallel_2d')
QNSolver.findPotential(phi)

# Calculate diagnostic quantity |phi|_2
l2[tN]=np.sum(phi._f**2*drCalc*dqCalc*dzCalc)

l2Result=np.zeros(tN+1)

comm.Reduce(l2,l2Result,op=MPI.SUM, root=0)

if (rank==0):
    l2Result=np.sqrt(l2Result)
    
    t=np.linspace(0,tEnd,tN+1)
    
    plt.figure()
    plt.plot(t,l2Result)
    plt.xlabel('time [s]')
    plt.ylabel('$|\phi|_2$')
    plt.show()


