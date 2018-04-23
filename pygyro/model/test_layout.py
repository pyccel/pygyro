from mpi4py import MPI
import numpy as np
import pytest

from .layout        import LayoutManager
from .process_grid  import compute_2d_process_grid

def define_f(Eta1,Eta2,Eta3,Eta4,layout,f):
    nEta1=len(Eta1)
    nEta2=len(Eta2)
    nEta3=len(Eta3)
    nEta4=len(Eta4)
    inv_dims=[0,0,0,0]
    for i,j in enumerate(layout.dims_order):
        inv_dims[j]=i
    
    # The loop ordering for the tests is not optimal but allows all layouts to use the same function
    
    for i,eta1 in enumerate(Eta1[layout.starts[inv_dims[0]]:layout.ends[inv_dims[0]]]):
        # get global index
        I=i+layout.starts[inv_dims[0]]
        for j,eta2 in enumerate(Eta2[layout.starts[inv_dims[1]]:layout.ends[inv_dims[1]]]):
            # get global index
            J=j+layout.starts[inv_dims[1]]
            for k,eta3 in enumerate(Eta3[layout.starts[inv_dims[2]]:layout.ends[inv_dims[2]]]):
                # get global index
                K=k+layout.starts[inv_dims[2]]
                for l,eta4 in enumerate(Eta4[layout.starts[inv_dims[3]]:layout.ends[inv_dims[3]]]):
                    # get global index
                    L=l+layout.starts[inv_dims[3]]
                    indices=[0,0,0,0]
                    indices[inv_dims[0]]=i
                    indices[inv_dims[1]]=j
                    indices[inv_dims[2]]=k
                    indices[inv_dims[3]]=l
                    
                    # set value using global indices
                    f[indices[0],indices[1],indices[2],indices[3]]= \
                            I*nEta4*nEta3*nEta2+J*nEta4*nEta3+K*nEta4+L

def compare_f(Eta1,Eta2,Eta3,Eta4,layout,f):
    nEta1=len(Eta1)
    nEta2=len(Eta2)
    nEta3=len(Eta3)
    nEta4=len(Eta4)
    inv_dims=[0,0,0,0]
    for i,j in enumerate(layout.dims_order):
        inv_dims[j]=i
    
    # The loop ordering for the tests is not optimal but allows all layouts to use the same function
    
    for i,eta1 in enumerate(Eta1[layout.starts[inv_dims[0]]:layout.ends[inv_dims[0]]]):
        # get global index
        I=i+layout.starts[inv_dims[0]]
        for j,eta2 in enumerate(Eta2[layout.starts[inv_dims[1]]:layout.ends[inv_dims[1]]]):
            # get global index
            J=j+layout.starts[inv_dims[1]]
            for k,eta3 in enumerate(Eta3[layout.starts[inv_dims[2]]:layout.ends[inv_dims[2]]]):
                # get global index
                K=k+layout.starts[inv_dims[2]]
                for l,eta4 in enumerate(Eta4[layout.starts[inv_dims[3]]:layout.ends[inv_dims[3]]]):
                    # get global index
                    L=l+layout.starts[inv_dims[3]]
                    indices=[0,0,0,0]
                    indices[inv_dims[0]]=i
                    indices[inv_dims[1]]=j
                    indices[inv_dims[2]]=k
                    indices[inv_dims[3]]=l
                    
                    # ensure value is as expected from function define_f()
                    assert(f[indices[0],indices[1],indices[2],indices[3]]== \
                        float(I*nEta4*nEta3*nEta2+J*nEta4*nEta3+K*nEta4+L))

@pytest.mark.parallel
def test_LayoutSwap():
    nprocs = compute_2d_process_grid( [40,20,10,30], MPI.COMM_WORLD.Get_size() )
    
    eta_grids=[np.linspace(0,1,40),
               np.linspace(0,6.28318531,20),
               np.linspace(0,10,10),
               np.linspace(0,10,30)]
    
    layouts = {'flux_surface': [0,3,1,2],
               'v_parallel'  : [0,2,1,3],
               'poloidal'    : [3,2,1,0]}
    remapper = LayoutManager( MPI.COMM_WORLD, layouts, nprocs, eta_grids )

    fsLayout = remapper.getLayout('flux_surface')
    vLayout = remapper.getLayout('v_parallel')
    pLayout = remapper.getLayout('poloidal')
    
    f_fs = np.empty(fsLayout.shape)
    f_v = np.empty(vLayout.shape)
    f_p = np.empty(pLayout.shape)
    
    define_f(eta_grids[0],eta_grids[1],eta_grids[2],eta_grids[3],fsLayout,f_fs)
    
    remapper.transpose(source=f_fs,
                       dest=f_v,
                       source_name='flux_surface',
                       dest_name='v_parallel')
    
    compare_f(eta_grids[0],eta_grids[1],eta_grids[2],eta_grids[3],vLayout,f_v)

@pytest.mark.parallel
def test_IncompatibleLayoutError():
    nprocs = compute_2d_process_grid( [10,10,10,10], MPI.COMM_WORLD.Get_size() )
    if (nprocs[0]==nprocs[1]): return
    
    eta_grids=[np.linspace(0,1,10),
               np.linspace(0,6.28318531,10),
               np.linspace(0,10,10),
               np.linspace(0,10,10)]
    with pytest.raises(RuntimeError):
        layouts = {'flux_surface': [0,3,1,2],
                   'v_parallel'  : [0,2,1,3],
                   'poloidal'    : [3,2,1,0],
                   'broken'      : [1,0,2,3]}
        LayoutManager( MPI.COMM_WORLD, layouts, nprocs, eta_grids )

@pytest.mark.parallel
def test_CompatibleLayouts():
    eta_grids=[np.linspace(0,1,10),
               np.linspace(0,6.28318531,10),
               np.linspace(0,10,10),
               np.linspace(0,10,10)]
    
    nprocs = compute_2d_process_grid( [10,10,10,10], MPI.COMM_WORLD.Get_size() )
    
    layouts = {'flux_surface': [0,3,1,2],
               'v_parallel'  : [0,2,1,3],
               'poloidal'    : [3,2,1,0]}
    LayoutManager( MPI.COMM_WORLD, layouts, nprocs, eta_grids )

@pytest.mark.parallel
def test_BadStepWarning():
    nprocs = compute_2d_process_grid( [10,10,20,15], MPI.COMM_WORLD.Get_size() )
    if (nprocs[0]==nprocs[1]): return
    
    eta_grids=[np.linspace(0,1,10),
               np.linspace(0,6.28318531,10),
               np.linspace(0,10,20),
               np.linspace(0,10,15)]
    
    layouts = {'flux_surface': [0,3,1,2],
               'v_parallel'  : [0,2,1,3],
               'poloidal'    : [3,2,1,0]}
    remapper = LayoutManager( MPI.COMM_WORLD, layouts, nprocs, eta_grids )

    myLayout = remapper.getLayout('flux_surface')
    endLayout = remapper.getLayout('poloidal')
    
    with pytest.warns(UserWarning):
        remapper.transpose(source=np.empty(myLayout.shape),
                           dest=np.empty(endLayout.shape),
                           source_name='flux_surface',
                           dest_name='poloidal')
