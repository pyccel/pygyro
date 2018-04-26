from mpi4py import MPI
import numpy as np
import pytest

from .layout        import LayoutManager, Layout
from .process_grid  import compute_2d_process_grid, compute_2d_process_grid_from_max

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

@pytest.mark.serial
def test_Layout_DimsOrder():
    eta_grids=[np.linspace(0,1,40),
               np.linspace(0,6.28318531,20),
               np.linspace(0,10,10),
               np.linspace(0,10,30)]
    
    l = Layout('test', [2,3], [0,3,2,1], eta_grids, [0,0] )
    assert(l.dims_order==[0,3,2,1])
    assert(l.inv_dims_order==[0,3,2,1])
    
    l = Layout('test', [2,3], [0,2,3,1], eta_grids, [0,0] )
    assert(l.dims_order==[0,2,3,1])
    assert(l.inv_dims_order==[0,3,1,2])
    
    l = Layout('test', [2,3], [2,1,0,3], eta_grids, [0,0] )
    assert(l.dims_order==[2,1,0,3])
    assert(l.inv_dims_order==[2,1,0,3])

@pytest.mark.parallel
def test_OddLayoutPaths():
    comm = MPI.COMM_WORLD
    nprocs = compute_2d_process_grid_from_max( 10 , 20 , comm.Get_size() )
    
    eta_grids=[np.linspace(0,1,40),
               np.linspace(0,6.28318531,20),
               np.linspace(0,10,10),
               np.linspace(0,10,30)]
    
    layouts = {'0123': [0,1,2,3],
               '0321': [0,3,2,1],
               '1320': [1,3,2,0],
               '1302': [1,3,0,2],
               '2301': [2,3,0,1]}
    remapper = LayoutManager( comm, layouts, nprocs, eta_grids )
    
    layout1=remapper.getLayout('1320')
    assert(layout1.name=='1320')
    layout2=remapper.getLayout('1302')
    
    fStart = np.empty(remapper.bufferSize)
    fEnd = np.empty(remapper.bufferSize)
    f1_s = np.split(fStart,[layout1.size])[0].reshape(layout1.shape)
    f1_e = np.split(fEnd  ,[layout1.size])[0].reshape(layout1.shape)
    f2_s = np.split(fStart,[layout2.size])[0].reshape(layout2.shape)
    f2_e = np.split(fEnd  ,[layout2.size])[0].reshape(layout2.shape)
    
    define_f(eta_grids[0],eta_grids[1],eta_grids[2],eta_grids[3],layout1,f1_s)
    
    remapper.in_place_transpose(data=fStart,
                       source_name='1320',
                       dest_name='1302')
    
    compare_f(eta_grids[0],eta_grids[1],eta_grids[2],eta_grids[3],layout2,f2_s)
    
    remapper.transpose(source=f2_s,
                       dest  =fEnd,
                       source_name='1302',
                       dest_name='1320')
    
    compare_f(eta_grids[0],eta_grids[1],eta_grids[2],eta_grids[3],layout1,f1_e)

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
    
    fStart = np.empty(max(fsLayout.size,vLayout.size,pLayout.size))
    fEnd = np.empty(max(fsLayout.size,vLayout.size,pLayout.size))
    f_fs_s = np.split(fStart,[fsLayout.size])[0].reshape(fsLayout.shape)
    assert(not f_fs_s.flags['OWNDATA'])
    f_v_s  = np.split(fStart,[ vLayout.size])[0].reshape( vLayout.shape)
    assert(not f_v_s.flags['OWNDATA'])
    f_p_s  = np.split(fStart,[ pLayout.size])[0].reshape( pLayout.shape)
    assert(not f_p_s.flags['OWNDATA'])
    f_fs_e = np.split(fEnd  ,[fsLayout.size])[0].reshape(fsLayout.shape)
    assert(not f_fs_e.flags['OWNDATA'])
    f_v_e  = np.split(fEnd  ,[ vLayout.size])[0].reshape( vLayout.shape)
    assert(not f_v_e.flags['OWNDATA'])
    f_p_e  = np.split(fEnd  ,[ pLayout.size])[0].reshape( pLayout.shape)
    assert(not f_p_e.flags['OWNDATA'])
    
    define_f(eta_grids[0],eta_grids[1],eta_grids[2],eta_grids[3],fsLayout,f_fs_s)
    
    remapper.transpose(source=f_fs_s,
                       dest=fEnd,
                       source_name='flux_surface',
                       dest_name='v_parallel')
    
    compare_f(eta_grids[0],eta_grids[1],eta_grids[2],eta_grids[3],fsLayout,f_fs_s)
    compare_f(eta_grids[0],eta_grids[1],eta_grids[2],eta_grids[3],vLayout,f_v_e)
    
    remapper.transpose(source = f_v_e,
                       dest=fStart,
                       source_name='v_parallel',
                       dest_name='poloidal')
    
    compare_f(eta_grids[0],eta_grids[1],eta_grids[2],eta_grids[3],vLayout,f_v_e)
    compare_f(eta_grids[0],eta_grids[1],eta_grids[2],eta_grids[3],pLayout,f_p_s)
    
    remapper.transpose(source = f_p_s,
                       dest=fEnd,
                       source_name='poloidal',
                       dest_name='v_parallel')
    
    compare_f(eta_grids[0],eta_grids[1],eta_grids[2],eta_grids[3],pLayout,f_p_s)
    compare_f(eta_grids[0],eta_grids[1],eta_grids[2],eta_grids[3],vLayout,f_v_e)
    
    remapper.transpose(source = f_v_e,
                       dest = fStart,
                       source_name='v_parallel',
                       dest_name='flux_surface')
    
    compare_f(eta_grids[0],eta_grids[1],eta_grids[2],eta_grids[3],vLayout,f_v_e)
    compare_f(eta_grids[0],eta_grids[1],eta_grids[2],eta_grids[3],fsLayout,f_fs_s)

@pytest.mark.parallel
def test_in_place_LayoutSwap():
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
    
    f = np.empty(max(fsLayout.size,vLayout.size,pLayout.size))
    f_fs = np.split(f,[fsLayout.size])[0].reshape(fsLayout.shape)
    assert(not f_fs.flags['OWNDATA'])
    f_v  = np.split(f,[ vLayout.size])[0].reshape( vLayout.shape)
    assert(not f_v.flags['OWNDATA'])
    f_p  = np.split(f,[ pLayout.size])[0].reshape( pLayout.shape)
    assert(not f_p.flags['OWNDATA'])
    
    define_f(eta_grids[0],eta_grids[1],eta_grids[2],eta_grids[3],fsLayout,f_fs)
    
    remapper.in_place_transpose(data=f,
                       source_name='flux_surface',
                       dest_name='v_parallel')
    
    compare_f(eta_grids[0],eta_grids[1],eta_grids[2],eta_grids[3],vLayout,f_v)
    
    remapper.in_place_transpose(data = f,
                       source_name='v_parallel',
                       dest_name='poloidal')
    
    compare_f(eta_grids[0],eta_grids[1],eta_grids[2],eta_grids[3],pLayout,f_p)
    
    remapper.in_place_transpose(data = f,
                       source_name='poloidal',
                       dest_name='v_parallel')
    
    compare_f(eta_grids[0],eta_grids[1],eta_grids[2],eta_grids[3],vLayout,f_v)
    
    remapper.in_place_transpose(data = f,
                       source_name='v_parallel',
                       dest_name='flux_surface')
    
    compare_f(eta_grids[0],eta_grids[1],eta_grids[2],eta_grids[3],fsLayout,f_fs)
    
    assert(not f_fs.flags['OWNDATA'])
    assert(not f_v.flags['OWNDATA'])
    assert(not f_p.flags['OWNDATA'])

@pytest.mark.parallel
def test_IncompatibleLayoutError():
    nprocs = compute_2d_process_grid( [10,10,10,10], MPI.COMM_WORLD.Get_size() )
    
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
def test_in_place_BadStepWarning():
    nprocs = compute_2d_process_grid( [10,10,20,15], MPI.COMM_WORLD.Get_size() )
    
    eta_grids=[np.linspace(0,1,10),
               np.linspace(0,6.28318531,10),
               np.linspace(0,10,20),
               np.linspace(0,10,15)]
    
    layouts = {'flux_surface': [0,3,1,2],
               'v_parallel'  : [0,2,1,3],
               'poloidal'    : [3,2,1,0]}
    remapper = LayoutManager( MPI.COMM_WORLD, layouts, nprocs, eta_grids )
    
    fsLayout = remapper.getLayout('flux_surface')
    vLayout = remapper.getLayout('v_parallel')
    pLayout = remapper.getLayout('poloidal')
    
    f = np.empty(max(fsLayout.size,vLayout.size,pLayout.size))
    f_fs = np.split(f,[fsLayout.size])[0].reshape(fsLayout.shape)
    f_v  = np.split(f,[ vLayout.size])[0].reshape( vLayout.shape)
    f_p  = np.split(f,[ pLayout.size])[0].reshape( pLayout.shape)
    
    
    
    with pytest.warns(UserWarning):
        remapper.in_place_transpose(data = f,
                           source_name='flux_surface',
                           dest_name='poloidal')

@pytest.mark.parallel
def test_BadStepWarning():
    nprocs = compute_2d_process_grid( [10,10,20,15], MPI.COMM_WORLD.Get_size() )
    
    eta_grids=[np.linspace(0,1,10),
               np.linspace(0,6.28318531,10),
               np.linspace(0,10,20),
               np.linspace(0,10,15)]
    
    layouts = {'flux_surface': [0,3,1,2],
               'v_parallel'  : [0,2,1,3],
               'poloidal'    : [3,2,1,0]}
    remapper = LayoutManager( MPI.COMM_WORLD, layouts, nprocs, eta_grids )
    
    fsLayout = remapper.getLayout('flux_surface')
    vLayout = remapper.getLayout('v_parallel')
    pLayout = remapper.getLayout('poloidal')
    
    fStart = np.empty(max(fsLayout.size,vLayout.size,pLayout.size))
    fEnd = np.empty(max(fsLayout.size,vLayout.size,pLayout.size))
    f_fs_s = np.split(fStart,[fsLayout.size])[0].reshape(fsLayout.shape)
    assert(not f_fs_s.flags['OWNDATA'])
    f_v_s  = np.split(fStart,[ vLayout.size])[0].reshape( vLayout.shape)
    assert(not f_v_s.flags['OWNDATA'])
    f_p_s  = np.split(fStart,[ pLayout.size])[0].reshape( pLayout.shape)
    assert(not f_p_s.flags['OWNDATA'])
    f_fs_e = np.split(fEnd  ,[fsLayout.size])[0].reshape(fsLayout.shape)
    assert(not f_fs_e.flags['OWNDATA'])
    f_v_e  = np.split(fEnd  ,[ vLayout.size])[0].reshape( vLayout.shape)
    assert(not f_v_e.flags['OWNDATA'])
    f_p_e  = np.split(fEnd  ,[ pLayout.size])[0].reshape( pLayout.shape)
    assert(not f_p_e.flags['OWNDATA'])
    
    
    
    with pytest.warns(UserWarning):
        remapper.transpose(source = f_fs_s,
                           dest = fEnd,
                           source_name='flux_surface',
                           dest_name='poloidal')

@pytest.mark.parallel
def test_copy():
    nprocs = compute_2d_process_grid( [10,10,20,15], MPI.COMM_WORLD.Get_size() )
    
    eta_grids=[np.linspace(0,1,10),
               np.linspace(0,6.28318531,10),
               np.linspace(0,10,20),
               np.linspace(0,10,15)]
    
    layouts = {'flux_surface': [0,3,1,2],
               'v_parallel'  : [0,2,1,3],
               'poloidal'    : [3,2,1,0]}
    remapper = LayoutManager( MPI.COMM_WORLD, layouts, nprocs, eta_grids )
    
    fsLayout = remapper.getLayout('flux_surface')
    vLayout = remapper.getLayout('v_parallel')
    pLayout = remapper.getLayout('poloidal')
    
    fStart = np.empty(max(fsLayout.size,vLayout.size,pLayout.size))
    fEnd = np.empty(max(fsLayout.size,vLayout.size,pLayout.size))
    f_fs_s = np.split(fStart,[fsLayout.size])[0].reshape(fsLayout.shape)
    assert(not f_fs_s.flags['OWNDATA'])
    f_v_s  = np.split(fStart,[ vLayout.size])[0].reshape( vLayout.shape)
    assert(not f_v_s.flags['OWNDATA'])
    f_p_s  = np.split(fStart,[ pLayout.size])[0].reshape( pLayout.shape)
    assert(not f_p_s.flags['OWNDATA'])
    f_fs_e = np.split(fEnd  ,[fsLayout.size])[0].reshape(fsLayout.shape)
    assert(not f_fs_e.flags['OWNDATA'])
    f_v_e  = np.split(fEnd  ,[ vLayout.size])[0].reshape( vLayout.shape)
    assert(not f_v_e.flags['OWNDATA'])
    f_p_e  = np.split(fEnd  ,[ pLayout.size])[0].reshape( pLayout.shape)
    assert(not f_p_e.flags['OWNDATA'])
    
    define_f(eta_grids[0],eta_grids[1],eta_grids[2],eta_grids[3],fsLayout,f_fs_s)
    
    remapper.transpose(source=f_fs_s,
                       dest=fEnd,
                       source_name='flux_surface',
                       dest_name='flux_surface')
    
    compare_f(eta_grids[0],eta_grids[1],eta_grids[2],eta_grids[3],fsLayout,f_fs_s)
    compare_f(eta_grids[0],eta_grids[1],eta_grids[2],eta_grids[3],fsLayout,f_fs_e)
    
    remapper.in_place_transpose(data=fEnd,
                       source_name='flux_surface',
                       dest_name='flux_surface')
