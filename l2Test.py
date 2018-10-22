from mpi4py                 import MPI
import time
setup_time_start = time.clock()

from glob                   import glob

import numpy                as np
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
from pygyro.diagnostics.norms               import l2, l1, nParticles
from pygyro.diagnostics.energy              import KineticEnergy

loop_start = 0
loop_time = 0
diagnostic_start = 0
diagnostic_time = 0
output_start = 0
output_time = 0

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
parser.add_argument('-s', dest='saveStep',nargs=1,type=int,
                    default=[5],
                   help='Number of time steps between writing output')

def my_print(rank,*args,**kwargs):
    if (rank==0):
        print(time.clock(),*args,**kwargs,file=open("out{}.txt".format(MPI.COMM_WORLD.Get_size()),"a"))

args = parser.parse_args()
foldername = args.foldername[0]

loadable = False

rDegree = args.rDegree[0]
qDegree = args.qDegree[0]
zDegree = args.zDegree[0]
vDegree = args.vDegree[0]

saveStep = args.saveStep[0]
saveStepCut = saveStep-1

tEnd = args.tEnd[0]

if (len(foldername)>0):
    print("To load from ",foldername)
    if (os.path.isdir(foldername) and os.path.exists("{0}/initParams.h5".format(foldername))):
        loadable = True
else:
    foldername = None

if (loadable):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    my_print(rank,"ready to setup from loadable")

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
        
    my_print(rank,"setting up from ",t)
    distribFunc = setupFromFile(foldername,comm=comm,
                                allocateSaveMemory = True,
                                layout = 'v_parallel')
else:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    my_print(rank,"ready to setup new")

    npts = [256,512,32,128]
    # npts = [128,256,32,64]

    dt=2

    tN = int(tEnd//dt)

    halfStep = dt*0.5

    my_print(rank,"about to setup")

    distribFunc = setupCylindricalGrid(npts   = npts,
                                layout = 'v_parallel',
                                comm   = comm,
                                allocateSaveMemory = True)

    my_print(rank,"setup done, saving initParams")
    
    foldername = setupSave(rDegree,qDegree,zDegree,vDegree,npts,dt,foldername)
    print("Saving in ",foldername)
    t = 0

my_print(rank,"conditional setup done")

fluxAdv = FluxSurfaceAdvection(distribFunc.eta_grid, distribFunc.get2DSpline(),
                                distribFunc.getLayout('flux_surface'),halfStep)
my_print(rank,"flux adv init done")
vParAdv = VParallelAdvection(distribFunc.eta_grid, distribFunc.getSpline(3),'periodic')
my_print(rank,"v par adv init done")
polAdv = PoloidalAdvection(distribFunc.eta_grid, distribFunc.getSpline(slice(1,None,-1)))
my_print(rank,"pol adv init done")
parGradVals = np.empty([distribFunc.getLayout(distribFunc.currentLayout).shape[0],npts[2],npts[1]])
my_print(rank,"par grad vals done")

layout_poisson   = {'v_parallel_2d': [0,2,1],
                    'mode_solve'   : [1,2,0]}
layout_vpar      = {'v_parallel_1d': [0,2,1]}
layout_poloidal  = {'poloidal'     : [2,1,0]}

nprocs = distribFunc.getLayout(distribFunc.currentLayout).nprocs[:2]
my_print(rank,"layout params ready")

remapperPhi = LayoutSwapper( comm, [layout_poisson, layout_vpar, layout_poloidal],
                            [nprocs,nprocs[0],nprocs[1]], distribFunc.eta_grid[:3],
                            'mode_solve' )
my_print(rank,"remapper1 done")
remapperRho = getLayoutHandler( comm, layout_poisson, nprocs, distribFunc.eta_grid[:3] )
my_print(rank,"remappers done")

phi = Grid(distribFunc.eta_grid[:3],distribFunc.getSpline(slice(0,3)),
            remapperPhi,'mode_solve',comm,dtype=np.complex128)
my_print(rank,"phi done")
rho = Grid(distribFunc.eta_grid[:3],distribFunc.getSpline(slice(0,3)),
            remapperRho,'v_parallel_2d',comm,dtype=np.complex128)
my_print(rank,"rho done")
phiSplines = [Spline2D(*distribFunc.getSpline(slice(1,None,-1))) for i in range(phi.getLayout('poloidal').shape[0])]
my_print(rank,"phi spline ready")
interpolator = SplineInterpolator2D(*distribFunc.getSpline(slice(1,None,-1)))
my_print(rank,"interp done")

density = DensityFinder(6,distribFunc.getSpline(3),distribFunc.eta_grid)
my_print(rank,"df ready")

QNSolver = QuasiNeutralitySolver(distribFunc.eta_grid[:3],7,distribFunc.getSpline(0),
                                chi=0)
my_print(rank,"QN ready")
parGrad = ParallelGradient(distribFunc.getSpline(1),distribFunc.eta_grid,remapperPhi.getLayout('v_parallel_1d'))

my_print(rank,"par grad ready")

# 0 - time
# 1 - l2 (phi)
# 2 - l2 (f)
# 3 - l1 (f)
# 4 - n particles
# 5 - min
# 6 - max
# 7 - KE
diagnostics = np.zeros([8,saveStep])
l2PhiResult=np.zeros(saveStep)
l2GridResult=np.zeros(saveStep)
l1Result=np.zeros(saveStep)
nPartResult=np.zeros(saveStep)
min_val=np.zeros(saveStep)
max_val=np.zeros(saveStep)
KE_val=np.zeros(saveStep)

l2_phi_class = l2(phi.eta_grid,phi.getLayout('v_parallel_2d'))
l1class = l1(distribFunc.eta_grid,distribFunc.getLayout('v_parallel'))
l2_grid_class = l2(distribFunc.eta_grid,distribFunc.getLayout('v_parallel'))
npart = nParticles(distribFunc.eta_grid,distribFunc.getLayout('v_parallel'))
KEclass = KineticEnergy(distribFunc.eta_grid,distribFunc.getLayout('v_parallel'))

my_print(rank,"diagnostics ready")


phi_filename = "{0}/phiDat.txt".format(foldername)

#Setup profiling tools
#~ pr = cProfile.Profile()
#~ pr.enable()

my_print(rank,"ready for setup")

setup_time = time.clock()-setup_time_start

# Find phi from f^n by solving QN eq
distribFunc.setLayout('v_parallel')
density.getPerturbedRho(distribFunc,rho)
my_print(rank,"pert rho")
QNSolver.getModes(rho)
my_print(rank,"got modes")
rho.setLayout('mode_solve')
phi.setLayout('mode_solve')
my_print(rank,"ready to solve")
QNSolver.solveEquation(phi,rho)
my_print(rank,"solved")
phi.setLayout('v_parallel_2d')
rho.setLayout('v_parallel_2d')
my_print(rank,"ready to inv fourier")
QNSolver.findPotential(phi)
my_print(rank,"got phi")

output_start=time.clock()
distribFunc.writeH5Dataset(foldername,t)
my_print(rank,"grid printed")
phi.writeH5Dataset(foldername,t,"phi")
my_print(rank,"phi printed")
output_time+=(time.clock()-output_start)

print("t=",t)

loop_start=time.clock()

for ti in range(tN):
    loop_time+=(time.clock()-loop_start)
    
    diagnostic_start=time.clock()
    # Calculate diagnostic quantities
    diagnostics[0,ti%saveStep]=t
    diagnostics[1,ti%saveStep]=l2_phi_class.l2NormSquared(phi)
    diagnostics[2,ti%saveStep]=l2_grid_class.l2NormSquared(distribFunc)
    diagnostics[3,ti%saveStep]=l1class.l1Norm(distribFunc)
    diagnostics[4,ti%saveStep]=npart.getN(distribFunc)
    diagnostics[5,ti%saveStep]=distribFunc.getMin()
    diagnostics[6,ti%saveStep]=distribFunc.getMax()
    diagnostics[7,ti%saveStep]=KEclass.getKE(distribFunc)
    
    t+=dt
    diagnostic_time+=(time.clock()-diagnostic_start)
    
    if (ti%saveStep==saveStepCut):
        my_print(rank,"save time")
        output_start=time.clock()
        #distribFunc.writeH5Dataset(foldername,t)
        #phi.writeH5Dataset(foldername,t,"phi")
        
        comm.Reduce(diagnostics[1,:],l2PhiResult,op=MPI.SUM, root=0)
        comm.Reduce(diagnostics[2,:],l2GridResult,op=MPI.SUM, root=0)
        comm.Reduce(diagnostics[3,:],l1Result,op=MPI.SUM, root=0)
        comm.Reduce(diagnostics[4,:],nPartResult,op=MPI.SUM, root=0)
        comm.Reduce(diagnostics[5,:],min_val,op=MPI.MIN, root=0)
        comm.Reduce(diagnostics[6,:],max_val,op=MPI.MAX, root=0)
        comm.Reduce(diagnostics[7,:],KE_val,op=MPI.SUM, root=0)
        if (rank == 0):
            l2PhiResult = np.sqrt(l2PhiResult)
            l2GridResult = np.sqrt(l2GridResult)
            phiFile = open(phi_filename,"a")
            for i in range(saveStep):
                print("{t:10g}   {l2P:16.10e}   {l2G:16.10e}   {l1:16.10e}   {np:16.10e}   {minim:16.10e}   {maxim:16.10e}   {ke:16.10e}".
                        format(t=diagnostics[0,i],l2P=l2PhiResult[i],l2G=l2GridResult[i],
                                l1=l1Result[i], np = nPartResult[i],
                                minim=min_val[i],maxim=max_val[i],ke=KE_val[i]),
                        file=phiFile)
            phiFile.close()
        output_time+=(time.clock()-output_start)
    
    my_print(rank,"t=",t)
    loop_start=time.clock()
    
    # Compute f^n+1/2 using lie splitting
    distribFunc.setLayout('flux_surface')
    distribFunc.saveGridValues()
    for i,r in distribFunc.getCoords(0):
        for j,v in distribFunc.getCoords(1):
            fluxAdv.step(distribFunc.get2DSlice([i,j]),j)
    distribFunc.setLayout('v_parallel')
    phi.setLayout('v_parallel_1d')
    for i,r in distribFunc.getCoords(0):
        parGrad.parallel_gradient(np.real(phi.get2DSlice([i])),i,parGradVals[0])
        for j,z in distribFunc.getCoords(1):
            for k,q in distribFunc.getCoords(2):
                vParAdv.step(distribFunc.get1DSlice([i,j,k]),halfStep,parGradVals[i,j,k],r)
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
        parGrad.parallel_gradient(np.real(phi.get2DSlice([i])),i,parGradVals[i])
        for j,z in distribFunc.getCoords(1):
            for k,q in distribFunc.getCoords(2):
                vParAdv.step(distribFunc.get1DSlice([i,j,k]),halfStep,parGradVals[i,j,k],r)
    distribFunc.setLayout('poloidal')
    phi.setLayout('poloidal')
    # For v[0]
    for j,z in distribFunc.getCoords(1):
        interpolator.compute_interpolant(np.real(phi.get2DSlice([j])),phiSplines[j])
        polAdv.step(distribFunc.get2DSlice([0,j]),halfStep,phiSplines[j],distribFunc.getCoordVals(0)[0])
    # For v[1:], splitting avoids calling interpolator more often than necessary
    for i,v in enumerate(distribFunc.getCoordVals(0)[1:],1):
        for j,z in distribFunc.getCoords(1):
            polAdv.step(distribFunc.get2DSlice([i,j]),halfStep,phiSplines[j],v)
    distribFunc.setLayout('v_parallel')
    for i,r in distribFunc.getCoords(0):
        for j,z in distribFunc.getCoords(1):
            for k,q in distribFunc.getCoords(2):
                vParAdv.step(distribFunc.get1DSlice([i,j,k]),halfStep,parGradVals[i,j,k],r)
    distribFunc.setLayout('flux_surface')
    for i,r in distribFunc.getCoords(0):
        for j,v in distribFunc.getCoords(1):
            fluxAdv.step(distribFunc.get2DSlice([i,j]),j)
    
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

loop_time+=(time.clock()-loop_start)

diagnostic_start=time.clock()
# Calculate diagnostic quantities
diagnostics[0,tN%saveStep]=t
diagnostics[1,tN%saveStep]=l2_phi_class.l2NormSquared(phi)
diagnostics[2,tN%saveStep]=l2_grid_class.l2NormSquared(distribFunc)
diagnostics[3,tN%saveStep]=l1class.l1Norm(distribFunc)
diagnostics[4,tN%saveStep]=npart.getN(distribFunc)
diagnostics[5,tN%saveStep]=distribFunc.getMin()
diagnostics[6,tN%saveStep]=distribFunc.getMax()
diagnostics[7,tN%saveStep]=KEclass.getKE(distribFunc)
diagnostic_time+=(time.clock()-diagnostic_start)

output_start=time.clock()

comm.Reduce(diagnostics[1,:],l2PhiResult,op=MPI.SUM, root=0)
comm.Reduce(diagnostics[2,:],l2GridResult,op=MPI.SUM, root=0)
comm.Reduce(diagnostics[3,:],l1Result,op=MPI.SUM, root=0)
comm.Reduce(diagnostics[4,:],nPartResult,op=MPI.SUM, root=0)
comm.Reduce(diagnostics[5,:],min_val,op=MPI.MIN, root=0)
comm.Reduce(diagnostics[6,:],max_val,op=MPI.MAX, root=0)
comm.Reduce(diagnostics[7,:],KE_val,op=MPI.SUM, root=0)
if (rank == 0):
    l2PhiResult = np.sqrt(l2PhiResult)
    l2GridResult = np.sqrt(l2GridResult)
    phiFile = open(phi_filename,"a")
    for i in range(tN%saveStep+1):
        print("{t:10g}   {l2P:16.10e}   {l2G:16.10e}   {l1:16.10e}   {np:16.10e}   {minim:16.10e}   {maxim:16.10e}   {ke:16.10e}".
                    format(t=diagnostics[0,i],l2P=l2PhiResult[i],l2G=l2GridResult[i],
                            l1=l1Result[i], np = nPartResult[i],
                            minim=min_val[i],maxim=max_val[i],ke=KE_val[i]),
                    file=phiFile)
    phiFile.close()


distribFunc.writeH5Dataset(foldername,t)
phi.writeH5Dataset(foldername,t,"phi")
output_time+=(time.clock()-output_start)

#End profiling and print results
#~ pr.disable()
#~ s = io.StringIO()
#~ ps = pstats.Stats(pr, stream=s).sort_stats('time')
#~ ps.print_stats()
#~ print(s.getvalue(), file=open("profile/l2Test{}.txt".format(rank), "w"))


print("{loop:16.10e}   {output:16.10e}   {setup:16.10e}   {diagnostic:16.10e}".
            format(loop=loop_time,output=output_time,setup=setup_time,
            diagnostic=diagnostic_time),
        file=open("timing/{}_l2Test{}.txt".format(MPI.COMM_WORLD.Get_size(),rank), "w"))
