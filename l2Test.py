from mpi4py                 import MPI
from matplotlib             import rc
from glob                   import glob

import numpy                as np
import matplotlib.pyplot    as plt
import argparse
import os
import h5py
#~ import cProfile, pstats, io

from pygyro.model.layout                    import LayoutSwapper, getLayoutHandler
from pygyro.model.grid                      import Grid
from pygyro.initialisation.setups           import setupCylindricalGrid, setupFromFile
from pygyro.advection.advection             import FluxSurfaceAdvection, VParallelAdvection, PoloidalAdvection, ParallelGradient
from pygyro.poisson.poisson_solver          import DensityFinder, QuasiNeutralitySolver
from pygyro.splines.splines                 import Spline2D
from pygyro.splines.spline_interpolators    import SplineInterpolator2D
from pygyro.utilities.savingTools           import setupSave


parser = argparse.ArgumentParser(description='Process foldername')
parser.add_argument('tEnd', metavar='tEnd',nargs=1,type=int,
                   help='end time')
parser.add_argument('-f', dest='foldername',nargs=1,type=str,
                    default=[""],
                   help='the name of the folder from which to load and in which to save')
parser.add_argument('-r', dest='rDegree',nargs=1,type=int,
                    default=[3],
                   help='Degree of spline in r')
parser.add_argument('-q', dest='qDegree',nargs=1,type=int,
                    default=[3],
                   help='Degree of spline in theta')
parser.add_argument('-z', dest='zDegree',nargs=1,type=int,
                    default=[3],
                   help='Degree of spline in z')
parser.add_argument('-v', dest='vDegree',nargs=1,type=int,
                    default=[3],
                   help='Degree of spline in v')

args = parser.parse_args()
foldername = args.foldername[0]

loadable = False

rDegree = args.rDegree[0]
qDegree = args.qDegree[0]
zDegree = args.zDegree[0]
vDegree = args.vDegree[0]

tEnd = args.tEnd[0]

if (len(foldername)>0):
    print("To load from ",foldername)
    if (os.path.isdir(foldername)):
        loadable = True
else:
    foldername = None

if (loadable):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    filename = "{0}/initParams.h5".format(foldername)
    save_file = h5py.File(filename,'r',driver='mpio',comm=comm)
    group = save_file['constants']
    
    npts = save_file.attrs['npts']
    dt = save_file.attrs['dt']
    
    halfStep = dt*0.5
    
    save_file.close()
    
    list_of_files = glob("{0}/grid_*".format(foldername))
    if (len(list_of_files)==0):
        tN = int(tEnd//dt)
        t=0
    else:
        filename = max(list_of_files)
        tStart = int(filename.split('_')[-1].split('.')[0])
        
        tN = int((tEnd-tStart)//dt)
        t = tStart
        
    distribFunc = setupFromFile(foldername,comm=comm,
                                allocateSaveMemory = True,
                                layout = 'v_parallel')
else:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    npts = [255,512,32,128]

    dt=2

    tN = int(tEnd//dt)

    halfStep = dt*0.5

    distribFunc = setupCylindricalGrid(npts   = npts,
                                layout = 'v_parallel',
                                comm   = comm,
                                allocateSaveMemory = True)
    
    foldername = setupSave(rDegree,qDegree,zDegree,vDegree,npts,dt,foldername)
    print("Saving in ",foldername)
    t = 0

fluxAdv = FluxSurfaceAdvection(distribFunc.eta_grid, distribFunc.get2DSpline(),
                                distribFunc.getLayout('flux_surface'),halfStep)
vParAdv = VParallelAdvection(distribFunc.eta_grid, distribFunc.getSpline(3))
polAdv = PoloidalAdvection(distribFunc.eta_grid, distribFunc.getSpline(slice(1,None,-1)))
parGrad = ParallelGradient(distribFunc.getSpline(1),distribFunc.eta_grid)
parGradVals = np.empty([npts[2],npts[1]])

layout_poisson   = {'v_parallel_2d': [0,2,1],
                    'mode_solve'   : [1,2,0]}
layout_vpar      = {'v_parallel_1d': [0,2,1]}
layout_poloidal  = {'poloidal'     : [2,1,0]}

nprocs = distribFunc.getLayout(distribFunc.currentLayout).nprocs[:2]

remapperPhi = LayoutSwapper( comm, [layout_poisson, layout_vpar, layout_poloidal],
                            [nprocs,nprocs[0],nprocs[1]], distribFunc.eta_grid[:3],
                            'mode_solve' )
remapperRho = getLayoutHandler( comm, layout_poisson, nprocs, distribFunc.eta_grid[:3] )

phi = Grid(distribFunc.eta_grid[:3],distribFunc.getSpline(slice(0,3)),
            remapperPhi,'mode_solve',comm,dtype=np.complex128)
rho = Grid(distribFunc.eta_grid[:3],distribFunc.getSpline(slice(0,3)),
            remapperRho,'v_parallel_2d',comm,dtype=np.complex128)
phiSplines = [Spline2D(*distribFunc.getSpline(slice(1,None,-1))) for i in range(phi.getLayout('poloidal').shape[0])]
interpolator = SplineInterpolator2D(*distribFunc.getSpline(slice(1,None,-1)))

density = DensityFinder(6,distribFunc.getSpline(3))

QNSolver = QuasiNeutralitySolver(distribFunc.eta_grid[:3],7,distribFunc.getSpline(0),
                                chi=0)


saveStep = 5

l2Phi = np.zeros([2,saveStep])
l2Result=np.zeros(saveStep)

r = phi.eta_grid[0]
q = phi.eta_grid[1]
z = phi.eta_grid[2]
dr = np.array([r[0], *(r[:-1]+r[1:]), r[-1]])*0.5
dq = q[1]-q[0]
dz = z[2]-z[1]

rCalc = (r[phi.getLayout('v_parallel_2d').starts[0]:phi.getLayout('v_parallel_2d').ends[0]])[:,None,None]
drCalc = (dr[phi.getLayout('v_parallel_2d').starts[0]:phi.getLayout('v_parallel_2d').ends[0]])[:,None,None]

phi_filename = "{0}/phiDat.dat".format(foldername)
if (not os.path.exists(phi_filename)):
    phiFile = h5py.File(phi_filename,'w',driver='mpio',comm=comm)
    dset = phiFile.create_dataset("dset",(tN+1, 2), float)
    phiFile.close()
else:
    phiFile = h5py.File(phi_filename,'r+',driver='mpio',comm=comm)
    dset = phiFile['/dset']
    assert(dset.size==(tEnd//dt+1)*2)
    phiFile.close()

#Setup profiling tools
#~ pr = cProfile.Profile()
#~ pr.enable()

print("ready")

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
    
    if (ti%saveStep==0 and ti!=0):
        distribFunc.writeH5Dataset(foldername,t)
        
        comm.Reduce(l2Phi[1,:],l2Result,op=MPI.SUM, root=0)
        l2Result = np.sqrt(l2Result)
        phiFile = h5py.File(phi_filename,'r+',driver='mpio',comm=comm)
        if (rank == 0):
            n = int(t/dt)
            dset = phiFile['/dset']
            dset[n-saveStep:n,0]=l2Phi[0,:]
            dset[n-saveStep:n,1]=l2Result
        phiFile.close()
    
    # Calculate diagnostic quantity |phi|_2
    l2Phi[0,ti%saveStep]=t
    l2Phi[1,ti%saveStep]=np.sum(np.real(phi._f*phi._f.conj()*drCalc*dq*dz*rCalc))
    
    print("t=",t)
    
    t+=dt
    
    # Compute f^n+1/2 using lie splitting
    distribFunc.setLayout('flux_surface')
    distribFunc.saveGridValues()
    for i,r in distribFunc.getCoords(0):
        for j,v in distribFunc.getCoords(1):
            fluxAdv.step(distribFunc.get2DSlice([i,j]),j)
    distribFunc.setLayout('v_parallel')
    phi.setLayout('v_parallel_1d')
    for i,r in distribFunc.getCoords(0):
        parGrad.parallel_gradient(np.real(phi.get2DSlice([i])),i,parGradVals)
        for j,z in distribFunc.getCoords(1):
            for k,q in distribFunc.getCoords(2):
                vParAdv.step(distribFunc.get1DSlice([i,j,k]),halfStep,parGradVals[j,k],r)
    distribFunc.setLayout('poloidal')
    phi.setLayout('poloidal')
    for j,z in distribFunc.getCoords(1):
        interpolator.compute_interpolant(np.real(phi.get2DSlice([j])),phiSplines[j])
        polAdv.step(distribFunc.get2DSlice([0,j]),halfStep,phiSplines[j],distribFunc.getCoordVals(0)[0])
    for i,v in enumerate(distribFunc.getCoordVals(0)[1:],1):
        for j,z in distribFunc.getCoords(1):
            polAdv.step(distribFunc.get2DSlice([i,j]),halfStep,phiSplines[j],v)
    
    # Find phi from f^n+1/2 by solving QN eq again
    distribFunc.setLayout('v_parallel')
    density.getPerturbedRho(distribFunc,rho)
    QNSolver.getModes(rho)
    rho.setLayout('mode_solve')
    phi.setLayout('mode_solve')
    QNSolver.solveEquation(phi,rho)
    phi.setLayout('v_parallel_2d')
    rho.setLayout('v_parallel_2d')
    QNSolver.findPotential(phi)
    
    # Compute f^n+1 using strang splitting
    distribFunc.restoreGridValues() # restored from flux_surface layout
    for i,r in distribFunc.getCoords(0):
        for j,v in distribFunc.getCoords(1):
            fluxAdv.step(distribFunc.get2DSlice([i,j]),j)
    distribFunc.setLayout('v_parallel')
    phi.setLayout('v_parallel_1d')
    for i,r in distribFunc.getCoords(0):
        parGrad.parallel_gradient(np.real(phi.get2DSlice([i])),i,parGradVals)
        for j,z in distribFunc.getCoords(1):
            for k,q in distribFunc.getCoords(2):
                vParAdv.step(distribFunc.get1DSlice([i,j,k]),halfStep,0,r)
    distribFunc.setLayout('poloidal')
    phi.setLayout('poloidal')
    for j,z in distribFunc.getCoords(1):
        interpolator.compute_interpolant(np.real(phi.get2DSlice([j])),phiSplines[j])
        polAdv.step(distribFunc.get2DSlice([0,j]),halfStep,phiSplines[j],distribFunc.getCoordVals(0)[0])
    for i,v in enumerate(distribFunc.getCoordVals(0)[1:],1):
        for j,z in distribFunc.getCoords(1):
            polAdv.step(distribFunc.get2DSlice([i,j]),halfStep,phiSplines[j],v)
    distribFunc.setLayout('v_parallel')
    for i,r in distribFunc.getCoords(0):
        for j,z in distribFunc.getCoords(1):
            for k,q in distribFunc.getCoords(2):
                vParAdv.step(distribFunc.get1DSlice([i,j,k]),halfStep,0,r)
    distribFunc.setLayout('flux_surface')
    for i,r in distribFunc.getCoords(0):
        for j,v in distribFunc.getCoords(1):
            fluxAdv.step(distribFunc.get2DSlice([i,j]),j)

if (tN%saveStep==0):
    comm.Reduce(l2Phi[1,:],l2Result,op=MPI.SUM, root=0)
    l2Result = np.sqrt(l2Result)
    phiFile = h5py.File(phi_filename,'r+',driver='mpio',comm=comm)
    if (rank == 0):
        n = int(t/dt)
        dset = phiFile['/dset']
        dset[n-saveStep:n,0]=l2Phi[0,:]
        dset[n-saveStep:n,1]=l2Result
    phiFile.close()

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
l2Phi[0,tN%saveStep]=t
l2Phi[1,tN%saveStep]=np.sum(np.real(phi._f*phi._f.conj()*drCalc*dq*dz*rCalc))

distribFunc.writeH5Dataset(foldername,t)

comm.Reduce(l2Phi[1,:],l2Result,op=MPI.SUM, root=0)
l2Result = np.sqrt(l2Result)
phiFile = h5py.File(phi_filename,'r+',driver='mpio',comm=comm)
if (rank == 0):
    nE = int(tEnd/dt+1)
    nS = int(nE-1-(tEnd/dt)%saveStep)
    dset = phiFile['/dset']
    dset[nS:nE,0]=l2Phi[0,:(nE-nS)]
    dset[nS:nE,1]=l2Result[:(nE-nS)]
phiFile.close()

#End profiling and print results
#~ pr.disable()
#~ s = io.StringIO()
#~ ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
#~ ps.print_stats()
#~ print(s.getvalue(), file=open("profile/l2Test{}.txt".format(rank), "w"))

