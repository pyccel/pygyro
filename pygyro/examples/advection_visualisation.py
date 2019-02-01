comm = MPI.COMM_WORLD
rank = comm.Get_rank()

npts = [20,20,10,8]
drawRank = 1
grid = setupCylindricalGrid(npts   = npts,
                            layout = 'flux_surface',
                            eps    = 0.1,
                            comm   = comm,
                            plotThread = True,
                            drawRank = drawRank)

plt.ion()

plot = SlicePlotterNd(grid,False,comm,drawingRank=drawRank,drawRankInGrid=False)

if (rank!=drawRank):
    
    dt=1
    halfStep = dt*0.5
    
    fluxAdv = FluxSurfaceAdvection(grid.eta_grid, grid.get2DSpline(),
                                grid.getLayout('flux_surface'),halfStep)
    vParAdv = VParallelAdvection(grid.eta_grid, grid.getSpline(3))
    polAdv = PoloidalAdvection(grid.eta_grid, grid.getSpline(slice(1,None,-1)))
    
    phi = Spline2D(grid.getSpline(1),grid.getSpline(0))
    phiVals = np.empty([npts[1],npts[0]])
    #phiVals[:]=3
    phiVals[:]=3*grid.eta_grid[0]**2
    #phiVals[:]=10*eta_vals[0]
    interp = SplineInterpolator2D(grid.getSpline(1),grid.getSpline(0))
    
    interp.compute_interpolant(phiVals,phi)

if (rank==drawRank):
    p.show()
else:
    while (p.calculation_complete()):
        for i,r in grid.getCoords(0):
            for j,v in grid.getCoords(1):
                fluxAdv.step(grid.get2DSlice([i,j]),j)
        
        print("rank ",rank," has completed flux step 1")
            
        grid.setLayout('v_parallel')
        
        for i,r in grid.getCoords(0):
            for j,z in grid.getCoords(1):
                for k,q in grid.getCoords(2):
                    vParAdv.step(grid.get1DSlice([i,j,k]),halfStep,0,r)
        
        print("rank ",rank," has completed v parallel step 1")
        
        grid.setLayout('poloidal')
        
        for i,v in grid.getCoords(0):
            for j,z in grid.getCoords(1):
                polAdv.step(grid.get2DSlice([i,j]),dt,phi,v)
        
        print("rank ",rank," has completed poloidal step")
        
        grid.setLayout('v_parallel')
        
        for i,r in grid.getCoords(0):
            for j,z in grid.getCoords(1):
                for k,q in grid.getCoords(2):
                    vParAdv.step(grid.get1DSlice([i,j,k]),halfStep,0,r)
        
        print("rank ",rank," has completed v parallel step 2")
        
        grid.setLayout('flux_surface')
        
        for i,r in grid.getCoords(0):
            for j,v in grid.getCoords(1):
                fluxAdv.step(grid.get2DSlice([i,j]),j)
        
        print("rank ",rank," has completed flux step 2")
        
        plot.updateDraw()
