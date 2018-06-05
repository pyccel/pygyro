from mpi4py                 import MPI
from math                   import pi
from enum                   import IntEnum
from matplotlib.widgets     import Button
import matplotlib.pyplot    as plt
import numpy                as np

from .discrete_slider import DiscreteSlider
from ..model.grid     import Grid

class Dimension(IntEnum):
    ETA1 = 0,
    ETA2 = 1,
    ETA3 = 2,
    ETA4 = 3

class SlicePlotter4d(object):
    def __init__(self,grid: Grid, autoUpdate: bool = True,
                 comm: MPI.Comm = MPI.COMM_WORLD, drawingRank: int = 0,
                 drawRankInGrid: bool = True):
        self.grid = grid
        self.autoUpdate = autoUpdate
        self.drawRankInGrid = drawRankInGrid
        self.closed=False
        
        # get MPI values
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.drawRank = drawingRank
        
        # get initial values for the sliders
        self.vVal = grid.nGlobalCoords[Dimension.ETA4]//2
        self.zVal = 0
        
        self.minimum=grid.getMin(drawingRank)
        self.maximum=grid.getMax(drawingRank)
        
        if (not drawRankInGrid):
            otherRank = (self.drawRank+1)%self.comm.Get_size()
            if (self.rank == self.drawRank):
                layout1=comm.recv(None, source=otherRank, tag=77)
                layout2=comm.recv(None, source=otherRank, tag=78)
                layout3=comm.recv(None, source=otherRank, tag=79)
                self._layouts = {'flux_surface':layout1, 'poloidal': layout2, 'v_parallel': layout3}
                self.nprocs = comm.recv(None, source=otherRank, tag=80)
            elif (self.rank == otherRank):
                comm.send(grid._layout_manager.getLayout('flux_surface'), dest=self.drawRank, tag=77)
                comm.send(grid._layout_manager.getLayout('poloidal'), dest=self.drawRank, tag=78)
                comm.send(grid._layout_manager.getLayout('v_parallel'), dest=self.drawRank, tag=79)
                comm.send(grid._layout_manager.nProcs, dest=self.drawRank, tag=80)
        else:
            self.nprocs = grid._layout_manager.nProcs
            self._layouts = {'flux_surface':grid._layout_manager.getLayout('flux_surface'),
                             'poloidal': grid._layout_manager.getLayout('poloidal'),
                             'v_parallel': grid._layout_manager.getLayout('v_parallel')}
        
        if (self.rank==self.drawRank):
            self.fig = plt.figure()
            self.fig.canvas.mpl_connect('close_event', self.handle_close)
            self.ax = self.fig.add_axes([0.1, 0.25, 0.7, 0.7],)
            self.sliderax1 = self.fig.add_axes([0.1, 0.02, 0.6, 0.03],)
            self.sliderax2 = self.fig.add_axes([0.1, 0.12, 0.6, 0.03],)
            self.colorbarax2 = self.fig.add_axes([0.85, 0.15, 0.03, 0.8],)
            
            self.calcText = plt.text(0.5, 0.5, 'calculating...',
                                     horizontalalignment='center',
                                     verticalalignment='center',
                                     transform=self.ax.transAxes)
            self.calcText.set_zorder(20)

            # add sliders and remember their values
            self.vPar = DiscreteSlider(self.sliderax1, r'$v_\parallel$',
                    valinit=grid.eta_grid[Dimension.ETA4][self.vVal],values=grid.eta_grid[Dimension.ETA4])
            self.Z = DiscreteSlider(self.sliderax2, 'z', valinit=grid.eta_grid[Dimension.ETA3][self.zVal],
                    values=grid.eta_grid[Dimension.ETA3])
            self.vPar.on_changed(self.updateV)
            self.Z.on_changed(self.updateZ)
            
            self.buttonax = plt.axes([0.9, 0.01, 0.09, 0.075])
            self._button = Button(self.buttonax, 'Step')
            self._button.on_clicked(self.step)
            
            # get coordinate values
            theta = np.repeat(np.append(self.grid.eta_grid[Dimension.ETA2],2*pi),
                    self.grid.nGlobalCoords[Dimension.ETA1]) \
                    .reshape(
                    self.grid.nGlobalCoords[Dimension.ETA2]+1,
                    self.grid.nGlobalCoords[Dimension.ETA1])
            r = np.tile(self.grid.eta_grid[Dimension.ETA1],
                        self.grid.nGlobalCoords[Dimension.ETA2]+1) \
                        .reshape(self.grid.nGlobalCoords[Dimension.ETA2]+1,
                                self.grid.nGlobalCoords[Dimension.ETA1])
            
            self.x=r*np.cos(theta)
            self.y=r*np.sin(theta)
        
        self.plotFigure()

    def updateV(self, value):
        if (self.autoUpdate):
            # alert non-0 ranks that v has been updated and the drawing must be updated
            MPI.COMM_WORLD.bcast(1,root=self.drawRank)
            self.vVal=(np.abs(value-self.grid.eta_grid[Dimension.ETA4])).argmin()
            # broadcast new value
            self.vVal=MPI.COMM_WORLD.bcast(self.vVal,root=self.drawRank)
            # update plot
            self.updateDraw()
            self.fig.canvas.start_event_loop(-1)
        elif (self.rank==self.drawRank):
            self.vVal=(np.abs(value-self.grid.eta_grid[Dimension.ETA4])).argmin()

    def updateZ(self, value):
        if (self.autoUpdate):
            # alert non-0 ranks that z has been updated and the drawing must be updated
            MPI.COMM_WORLD.bcast(2,root=self.drawRank)
            self.zVal=(np.abs(value-self.grid.eta_grid[Dimension.ETA3])).argmin()
            # broadcast new value
            self.zVal=MPI.COMM_WORLD.bcast(self.zVal,root=self.drawRank)
            # update plot
            self.updateDraw()
            self.fig.canvas.start_event_loop(-1)
        elif (self.rank==self.drawRank):
            self.zVal=(np.abs(value-self.grid.eta_grid[Dimension.ETA3])).argmin()
    
    def updateDraw(self):
        if (not self.autoUpdate):
            # broadcast new v value
            self.vVal=MPI.COMM_WORLD.bcast(self.vVal,root=self.drawRank)
            
            # broadcast new z value
            self.zVal=MPI.COMM_WORLD.bcast(self.zVal,root=self.drawRank)
            
            self.minimum=self.grid.getMin(self.drawRank)
            self.maximum=self.grid.getMax(self.drawRank)
            
        self.plotFigure()
    
    def plotFigure(self):
        # get the slice with v and z values as indicated by the sliders
        d = {2 : self.zVal, 3 : self.vVal}
        if (self.rank==self.drawRank):
            layout, starts, mpi_data, theSlice = self.grid.getSliceFromDict(d,self.comm,self.drawRank)
            
            layout = self._layouts[layout.name]
            
            baseShape = layout.shape
            
            if (layout.inv_dims_order[2]<2 and layout.inv_dims_order[3]<2):
                idx = np.where(starts==0)[-1]
                myShape = baseShape.copy()
                myShape[0]=1
                myShape[1]=1
                theSlice=np.squeeze(theSlice.reshape(myShape))
            else:
                if (self.drawRankInGrid):
                    splitSlice = np.split(theSlice,starts[1:])
                else:
                    splitSlice = np.split(theSlice,starts[2:])
                    mpi_data=mpi_data[1:]
                concatReady = [[None for i in range(self.nprocs[1])] for j in range(self.nprocs[0])]
                for i,chunk in enumerate(splitSlice):
                    coords=mpi_data[i]
                    myShape=baseShape.copy()
                    myShape[0]=layout.mpi_lengths(0)[coords[0]]
                    myShape[1]=layout.mpi_lengths(1)[coords[1]]
                    if (chunk.size==0):
                        myShape[min(layout.inv_dims_order[2:4])]=0
                        myShape[max(layout.inv_dims_order[2:4])]=1
                    else:
                        myShape[layout.inv_dims_order[2]]=1
                        myShape[layout.inv_dims_order[3]]=1
                    concatReady[coords[0]][coords[1]]=chunk.reshape(myShape)
                
                concat1 = [np.concatenate(concat,axis=1) for concat in concatReady]
                theSlice = np.squeeze(np.concatenate(concat1,axis=0))
            
            if (self.grid._layout.dims_order.index(0)<self.grid._layout.dims_order.index(1)):
                theSlice = np.append(theSlice, theSlice[:,0,None],axis=1).T
            else:
                theSlice = np.append(theSlice, theSlice[None,0,:],axis=0)
            
            # remove the old plot
            self.ax.clear()
            self.colorbarax2.clear()
            #,vmin=self.minimum,vmax=self.maximum
            self.plot = self.ax.pcolormesh(self.x,self.y,theSlice,cmap="jet",vmin=self.minimum,vmax=self.maximum)
            self.fig.colorbar(self.plot,cax=self.colorbarax2)
            
            self.ax.set_xlabel("x [m]")
            self.ax.set_ylabel("y [m]")
            
            self.fig.canvas.draw()
            
        else:
            self.grid.getSliceFromDict(d,self.comm,self.drawRank)
    
    def show(self):
        # set-up non-0 ranks as listeners so they can react to interactions with the plot on rank 0
        if (self.rank!=0):
            stop=1
            while (stop!=0):
                stop=MPI.COMM_WORLD.bcast(1,root=self.drawRank)
                if (stop==1):
                    self.vVal=MPI.COMM_WORLD.bcast(self.vVal,root=self.drawRank)
                    self.updateDraw()
                elif (stop==2):
                    self.zVal=MPI.COMM_WORLD.bcast(self.zVal,root=self.drawRank)
                    self.updateDraw()
        else:
            plt.show()
    
    def listen(self):
        self.autoUpdate = True
        # set-up non-0 ranks as listeners so they can react to interactions with the plot on rank 0
        if (self.rank!=self.drawRank):
            stop=1
            while (stop!=0):
                stop=MPI.COMM_WORLD.bcast(1,root=self.drawRank)
                if (stop==1):
                    self.vVal=MPI.COMM_WORLD.bcast(self.vVal,root=self.drawRank)
                    self.updateDraw()
                elif (stop==2):
                    self.zVal=MPI.COMM_WORLD.bcast(self.zVal,root=self.drawRank)
                    self.updateDraw()
                elif (stop==3):
                    self.autoUpdate = False
                    stop=0
                elif (stop==0):
                    return 0
        else:
            self.calcText.set_visible(False)
            plt.ioff()
            self.fig.canvas.start_event_loop(-1)
            if (self.closed):
                return 0
            else:
                self.calcText = plt.text(0.5, 0.5, 'calculating...',
                                         horizontalalignment='center',
                                         verticalalignment='center',
                                         transform=self.ax.transAxes,
                                         bbox=dict(color='white'))
                self.fig.canvas.draw()
    
    def handle_close(self,_evt):
        self.fig.canvas.stop_event_loop()
        # broadcast 0 to break non-0 ranks listen loop
        MPI.COMM_WORLD.bcast(0,root=self.drawRank)
        self.closed=True
    
    def step( self, _evt ):
        plt.ion()
        # broadcast 3 to break non-0 ranks listen loop
        MPI.COMM_WORLD.bcast(3,root=self.drawRank)
        self.autoUpdate = False
        self.fig.canvas.stop_event_loop()
        

class SlicePlotter3d(object):
    def __init__(self,grid: Grid, *args, **kwargs):
        self.comm = kwargs.pop('comm',MPI.COMM_WORLD)
        self.drawRank = kwargs.pop('drawingRank',0)
        
        self.grid = grid
        
        self.nprocs = grid._layout_manager.nProcs
        
        # use arguments to get coordinate values (including omitted coordinate)
        if (len(args)==0):
            # if no arguments are provided then assume (r,θ,z)
            self.xVals=Dimension.ETA1
            self.x=grid.eta_grid[Dimension.ETA1]
            self.yVals=Dimension.ETA2
            self.y=np.append(grid.eta_grid[Dimension.ETA2],2*pi)
            self.zVals=Dimension.ETA3
            self.omit=Dimension.ETA4
            self.omitVal=self.grid.nGlobalCoords[Dimension.ETA4]//2
        else:
            assert(len(args)==3)
            # use arguments to get x values
            if (args[0]=='r'):
                self.xVals=Dimension.ETA1
                self.x=grid.eta_grid[Dimension.ETA1]
            elif (args[0]=='q' or args[0]=='theta'):
                self.xVals=Dimension.ETA2
                self.x=np.append(grid.eta_grid[Dimension.ETA2],2*pi)
            elif (args[0]=='z'):
                self.xVals=Dimension.ETA3
                self.x=grid.eta_grid[Dimension.ETA3]
            elif (args[0]=='v'):
                self.xVals=Dimension.ETA4
                self.x=grid.eta_grid[Dimension.ETA4]
            else:
                raise TypeError("%s is not a valid dimension" % self.xVals)
            
            # use arguments to get y values
            if (args[1]=='r'):
                self.yVals=Dimension.ETA1
                self.y=grid.eta_grid[Dimension.ETA1]
            elif (args[1]=='q' or args[1]=='theta'):
                self.yVals=Dimension.ETA2
                self.y=np.append(grid.eta_grid[Dimension.ETA2],2*pi)
            elif (args[1]=='z'):
                self.yVals=Dimension.ETA3
                self.y=grid.eta_grid[Dimension.ETA3]
            elif (args[1]=='v'):
                self.yVals=Dimension.ETA4
                self.y=grid.eta_grid[Dimension.ETA4]
            else:
                raise TypeError("%s is not a valid dimension" % self.yVals)
            
            # use arguments to get z values
            if (args[2]=='r'):
                self.zVals=Dimension.ETA1
            elif (args[2]=='q' or args[2]=='theta'):
                self.zVals=Dimension.ETA2
            elif (args[2]=='z'):
                self.zVals=Dimension.ETA3
            elif (args[2]=='v'):
                self.zVals=Dimension.ETA4
            else:
                raise TypeError("%s is not a valid dimension" % self.zVals)
            
            # get unused dimensions identifier
            self.omit=(set(Dimension)-set([self.xVals,self.yVals,self.zVals])).pop()
            
            # fix unused dimension at an arbitrary (but measured) value
            if (self.omit==Dimension.ETA1):
                self.omitVal=self.grid.nGlobalCoords[Dimension.ETA1]//2
            elif (self.omit==Dimension.ETA2):
                self.omitVal=self.grid.nGlobalCoords[Dimension.ETA2]//2
            elif (self.omit==Dimension.ETA3):
                self.omitVal=self.grid.nGlobalCoords[Dimension.ETA3]//2
            elif (self.omit==Dimension.ETA4):
                self.omitVal=self.grid.nGlobalCoords[Dimension.ETA4]//2
        
        # save x and y grid values
        nx=len(self.x)
        ny=len(self.y)
        self.x = np.repeat(self.x,ny).reshape(nx,ny)
        self.y = np.tile(self.y,nx).reshape(nx,ny)
        
        # if (x,y) are (r,θ) or (θ,r) then print in polar coordinates
        self.polar=False
        if (self.xVals==Dimension.ETA1 and self.yVals==Dimension.ETA2):
            x=self.x*np.cos(self.y)
            y=self.x*np.sin(self.y)
            self.x=x
            self.y=y
            self.polar=True
        elif (self.yVals==Dimension.ETA1 and self.xVals==Dimension.ETA2):
            y=self.x*np.cos(self.y)
            x=self.x*np.sin(self.y)
            self.x=x
            self.y=y
            self.polar=True
        
        # get MPI values
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        
        # get max and min values of f to avoid colorbar jumps
        self.minimum=grid.getMin(0,self.omit,self.omitVal)
        self.maximum=grid.getMax(0,self.omit,self.omitVal)
        
        # on rank 0 set-up the graph
        if (self.rank==0):
            self.fig = plt.figure()
            self.fig.canvas.mpl_connect('close_event', self.handle_close)
            self.ax = self.fig.add_axes([0.1, 0.25, 0.7, 0.7],)
            self.sliderax1 = self.fig.add_axes([0.1, 0.02, 0.6, 0.03],)
            self.colorbarax2 = self.fig.add_axes([0.85, 0.1, 0.03, 0.8],)

            # add slider and remember its value
            self.initVal = 0
            if (self.zVals==Dimension.ETA1):
                self.slider = DiscreteSlider(self.sliderax1, "r",
                        valinit=grid.eta_grid[Dimension.ETA1][0],values=grid.eta_grid[Dimension.ETA1])
            elif (self.zVals==Dimension.ETA2):
                self.slider = DiscreteSlider(self.sliderax1, r'$\theta$',
                        valinit=grid.eta_grid[Dimension.ETA2][0],values=grid.eta_grid[Dimension.ETA2])
            elif (self.zVals==Dimension.ETA3):
                self.slider = DiscreteSlider(self.sliderax1, "z", valinit=grid.eta_grid[Dimension.ETA3][0],
                        values=grid.eta_grid[Dimension.ETA3])
            elif (self.zVals==Dimension.ETA4):
                self.initVal = self.grid.nGlobalCoords[Dimension.ETA4]//2
                self.slider = DiscreteSlider(self.sliderax1, r'$v_\parallel$', 
                        valinit=grid.eta_grid[Dimension.ETA4][self.grid.nGlobalCoords[Dimension.ETA4]//2],
                        values=grid.eta_grid[Dimension.ETA4])
            self.slider.on_changed(self.updateVal)
            if (not self.polar):
                # add x-axis label
                if (self.xVals==Dimension.ETA1):
                    self.ax.set_xlabel("r [m]")
                elif (self.xVals==Dimension.ETA2):
                    self.ax.set_xlabel(r'$\theta$ [rad]')
                elif (self.xVals==Dimension.ETA3):
                    self.ax.set_xlabel("z [m]")
                elif (self.xVals==Dimension.ETA4):
                    self.ax.set_xlabel(r'v [$ms^{-1}$]')
                
                # add y-axis label
                if (self.yVals==Dimension.ETA1):
                    self.ax.set_ylabel("r [m]")
                elif (self.yVals==Dimension.ETA2):
                    self.ax.set_ylabel(r'$\theta$ [rad]')
                elif (self.yVals==Dimension.ETA3):
                    self.ax.set_ylabel("z [m]")
                elif (self.yVals==Dimension.ETA4):
                    self.ax.set_ylabel(r'v [$ms^{-1}$]')
            else:            
                self.ax.set_xlabel("x [m]")
                self.ax.set_ylabel("y [m]")
        else:
            # on other ranks save the initial value for the slider
            if (self.zVals==Dimension.ETA4):
                self.initVal = self.grid.nGlobalCoords[Dimension.ETA4]//2
            else:
                self.initVal = 0
            
        self.plotFigure()
    
    def updateVal(self, value):
        # alert non-0 ranks that they must receive the new value and update the drawing
        MPI.COMM_WORLD.bcast(1,root=0)
        if (self.zVals==Dimension.ETA1):
            self.initVal=(np.abs(value-self.grid.eta_grid[Dimension.ETA1])).argmin()
        elif (self.zVals==Dimension.ETA2):
            self.initVal=(np.abs(value-self.grid.eta_grid[Dimension.ETA2])).argmin()
        elif (self.zVals==Dimension.ETA3):
            self.initVal=(np.abs(value-self.grid.eta_grid[Dimension.ETA3])).argmin()
        elif (self.zVals==Dimension.ETA4):
            self.initVal=(np.abs(value-self.grid.eta_grid[Dimension.ETA4])).argmin()
        # broadcast new value
        self.initVal=MPI.COMM_WORLD.bcast(self.initVal,root=0)
        # update plot
        self.updateDraw()
    
    def updateDraw(self):
        self.plotFigure()
    
    def plotFigure(self):
        # get slice by passing dictionary containing fixed dimensions and their values
        d = {self.zVals : self.initVal, self.omit : self.omitVal}
        
        if (self.rank==self.drawRank):
            layout, starts, mpi_data, theSlice = self.grid.getSliceFromDict(d,self.comm,self.drawRank)
            baseShape = layout.shape
            
            if (layout.inv_dims_order[self.zVals]<2 and layout.inv_dims_order[self.omit]<2):
                idx = np.where(starts==0)[-1]
                myShape = baseShape.copy()
                myShape[0]=1
                myShape[1]=1
                theSlice=np.squeeze(theSlice.reshape(myShape))
            else:
                splitSlice = np.split(theSlice,starts[1:])
                concatReady = [[None for i in range(self.nprocs[1])] for j in range(self.nprocs[0])]
                for i,chunk in enumerate(splitSlice):
                    coords=mpi_data[i]
                    myShape=baseShape.copy()
                    myShape[0]=layout.mpi_lengths(0)[coords[0]]
                    myShape[1]=layout.mpi_lengths(1)[coords[1]]
                    if (chunk.size==0):
                        myShape[min(layout.inv_dims_order[self.zVals],
                                    layout.inv_dims_order[self.omit])]=0
                        myShape[max(layout.inv_dims_order[self.zVals],
                                    layout.inv_dims_order[self.omit])]=1
                    else:
                        myShape[layout.inv_dims_order[self.zVals]]=1
                        myShape[layout.inv_dims_order[self.omit]]=1
                    concatReady[coords[0]][coords[1]]=chunk.reshape(myShape)
                
                concat1 = [np.concatenate(concat,axis=1) for concat in concatReady]
                theSlice = np.squeeze(np.concatenate(concat1,axis=0))
            
            if (layout.inv_dims_order[self.xVals]>layout.inv_dims_order[self.yVals]):
                if (Dimension.ETA2 == self.xVals):
                    theSlice = np.append(theSlice, theSlice[:,0,None],axis=1).T
                elif (Dimension.ETA2 == self.yVals):
                    theSlice = np.append(theSlice, theSlice[None,0,:],axis=0).T
                else:
                    theSlice = theSlice.T
            else:
                if (Dimension.ETA2 == self.xVals):
                    theSlice = np.append(theSlice, theSlice[None,0,:],axis=0)
                elif (Dimension.ETA2 == self.yVals):
                    theSlice = np.append(theSlice, theSlice[:,0,None],axis=1)
            
            self.fig.canvas.draw()
            
            if (hasattr(self,'plot')):
                # remove the old plot
                del self.plot
                del self.ax.get_children()[0]
            self.colorbarax2.clear()
            self.plot = self.ax.pcolormesh(self.x,self.y,theSlice,vmin=self.minimum,vmax=self.maximum,cmap="jet")
            self.fig.colorbar(self.plot,cax=self.colorbarax2)
            
            self.fig.canvas.draw()
            
        else:
            self.grid.getSliceFromDict(d,self.comm,self.drawRank)
    
    def show(self):
        # set-up non-0 ranks as listeners so they can react to interactions with the plot on rank 0
        if (self.rank!=0):
            stop=1
            while (stop!=0):
                stop=MPI.COMM_WORLD.bcast(1,root=0)
                if (stop==1):
                    self.initVal=MPI.COMM_WORLD.bcast(self.initVal,root=0)
                    self.updateDraw()
        plt.show()
    
    def handle_close(self,_evt):
        # broadcast 0 to break non-0 ranks listen loop
        MPI.COMM_WORLD.bcast(0,root=0)

class Plotter2d(object):
    def __init__(self,grid: Grid, *args, **kwargs):
        self.comm = kwargs.pop('comm',MPI.COMM_WORLD)
        self.drawRank = kwargs.pop('drawingRank',0)
        
        self.nprocs = grid._layout_manager.nProcs
        
        # use arguments to get x values
        assert(len(args)==2)
        if (args[0]=='r'):
            self.xVals=Dimension.ETA1
            self.x=grid.eta_grid[Dimension.ETA1]
        elif (args[0]=='q' or args[0]=='theta'):
            self.xVals=Dimension.ETA2
            self.x=np.append(grid.eta_grid[Dimension.ETA2],2*pi)
        elif (args[0]=='z'):
            self.xVals=Dimension.ETA3
            self.x=grid.eta_grid[Dimension.ETA3]
        elif (args[0]=='v'):
            self.xVals=Dimension.ETA4
            self.x=grid.eta_grid[Dimension.ETA4]
        else:
            raise TypeError("%s is not a valid dimension" % self.xVals)
        
        # use arguments to get y values
        if (args[1]=='r'):
            self.yVals=Dimension.ETA1
            self.y=grid.eta_grid[Dimension.ETA1]
        elif (args[1]=='q' or args[0]=='theta'):
            self.yVals=Dimension.ETA2
            self.y=np.append(grid.eta_grid[Dimension.ETA2],2*pi)
        elif (args[1]=='z'):
            self.yVals=Dimension.ETA3
            self.y=grid.eta_grid[Dimension.ETA3]
        elif (args[1]=='v'):
            self.yVals=Dimension.ETA4
            self.y=grid.eta_grid[Dimension.ETA4]
        else:
            raise TypeError("%s is not a valid dimension" % self.yVals)
        self.grid = grid
        
        # get unused dimensions identifiers
        omitted=(set(Dimension)-set([self.xVals,self.yVals]))
        self.omit1=omitted.pop()
        self.omit2=omitted.pop()
        
        # fix first unused dimension at an arbitrary (but measured) value
        if (self.omit1==Dimension.ETA1):
            self.omitVal1=self.grid.nGlobalCoords[Dimension.ETA1]//2
        elif (self.omit1==Dimension.ETA2):
            self.omitVal1=self.grid.nGlobalCoords[Dimension.ETA2]//2
        elif (self.omit1==Dimension.ETA3):
            self.omitVal1=self.grid.nGlobalCoords[Dimension.ETA3]//2
        elif (self.omit1==Dimension.ETA4):
            self.omitVal1=self.grid.nGlobalCoords[Dimension.ETA4]//2
        
        # fix second unused dimension at an arbitrary (but measured) value
        if (self.omit2==Dimension.ETA1):
            self.omitVal2=self.grid.nGlobalCoords[Dimension.ETA1]//2
        elif (self.omit2==Dimension.ETA2):
            self.omitVal2=self.grid.nGlobalCoords[Dimension.ETA2]//2
        elif (self.omit2==Dimension.ETA3):
            self.omitVal2=self.grid.nGlobalCoords[Dimension.ETA3]//2
        elif (self.omit2==Dimension.ETA4):
            self.omitVal2=self.grid.nGlobalCoords[Dimension.ETA4]//2
        
        # get MPI vals
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        
        # save x and y grid values
        nx=len(self.x)
        ny=len(self.y)
        self.x = np.repeat(self.x,ny).reshape(nx,ny)
        self.y = np.tile(self.y,nx).reshape(nx,ny)
        
        # if (x,y) are (r,θ) or (θ,r) then print in polar coordinates
        self.polar=False
        if (self.xVals==Dimension.ETA1 and self.yVals==Dimension.ETA2):
            x=self.x*np.cos(self.y)
            y=self.x*np.sin(self.y)
            self.x=x
            self.y=y
            self.polar=True
        elif (self.yVals==Dimension.ETA1 and self.xVals==Dimension.ETA2):
            y=self.x*np.cos(self.y)
            x=self.x*np.sin(self.y)
            self.x=x
            self.y=y
            self.polar=True
        
        # on rank 0 set-up the graph
        if (self.rank==0):
            self.fig = plt.figure()
            self.ax = self.fig.add_axes([0.1, 0.15, 0.7, 0.7],)
            self.colorbarax2 = self.fig.add_axes([0.85, 0.1, 0.03, 0.8],)
            if (not self.polar):
                # add x-axis label
                if (self.xVals==Dimension.ETA1):
                    self.ax.set_xlabel("r [m]")
                elif (self.xVals==Dimension.ETA2):
                    self.ax.set_xlabel(r'$\theta$ [rad]')
                elif (self.xVals==Dimension.ETA3):
                    self.ax.set_xlabel("z [m]")
                elif (self.xVals==Dimension.ETA4):
                    self.ax.set_xlabel(r'v [$ms^{-1}$]')
                # add y-axis label
                if (self.yVals==Dimension.ETA1):
                    self.ax.set_ylabel("r [m]")
                elif (self.yVals==Dimension.ETA2):
                    self.ax.set_ylabel(r'$\theta$ [rad]')
                elif (self.yVals==Dimension.ETA3):
                    self.ax.set_ylabel("z [m]")
                elif (self.yVals==Dimension.ETA4):
                    self.ax.set_ylabel(r'v [$ms^{-1}$]')
        
        self.plotFigure()
    
    def plotFigure(self):
        # get slice by passing dictionary containing fixed dimensions and their values
        d = {self.omit1 : self.omitVal1, self.omit2 : self.omitVal2}
        
        if (self.rank==self.drawRank):
            layout, starts, mpi_data, theSlice = self.grid.getSliceFromDict(d,self.comm,self.drawRank)
            baseShape = layout.shape
            
            if (layout.inv_dims_order[self.omit1]<2 and layout.inv_dims_order[self.omit2]<2):
                idx = np.where(starts==0)[-1]
                myShape = baseShape.copy()
                myShape[0]=1
                myShape[1]=1
                theSlice=np.squeeze(theSlice.reshape(myShape))
            else:
                splitSlice = np.split(theSlice,starts[1:])
                concatReady = [[None for i in range(self.nprocs[1])] for j in range(self.nprocs[0])]
                for i,chunk in enumerate(splitSlice):
                    coords=mpi_data[i]
                    myShape=baseShape.copy()
                    myShape[0]=layout.mpi_lengths(0)[coords[0]]
                    myShape[1]=layout.mpi_lengths(1)[coords[1]]
                    if (chunk.size==0):
                        myShape[min(layout.inv_dims_order[self.omit1],
                                    layout.inv_dims_order[self.omit2])]=0
                        myShape[max(layout.inv_dims_order[self.omit1],
                                    layout.inv_dims_order[self.omit2])]=1
                    else:
                        myShape[layout.inv_dims_order[self.omit1]]=1
                        myShape[layout.inv_dims_order[self.omit2]]=1
                    concatReady[coords[0]][coords[1]]=chunk.reshape(myShape)
                
                concat1 = [np.concatenate(concat,axis=1) for concat in concatReady]
                theSlice = np.squeeze(np.concatenate(concat1,axis=0))
            
            self.fig.canvas.draw()
            
            # remove the old plot
            self.ax.clear()
            self.colorbarax2.clear()
            
            if (layout.inv_dims_order[self.xVals]>layout.inv_dims_order[self.yVals]):
                if (Dimension.ETA2 == self.xVals):
                    theSlice = np.append(theSlice, theSlice[:,0,None],axis=1)
                elif (Dimension.ETA2 == self.yVals):
                    theSlice = np.append(theSlice, theSlice[None,0,:],axis=0)
                self.plot = self.ax.pcolormesh(self.x,self.y,theSlice.T,cmap="jet")
            else:
                if (Dimension.ETA2 == self.xVals):
                    theSlice = np.append(theSlice, theSlice[None,0,:],axis=0)
                elif (Dimension.ETA2 == self.yVals):
                    theSlice = np.append(theSlice, theSlice[:,0,None],axis=1)
                self.plot = self.ax.pcolormesh(self.x,self.y,theSlice,cmap="jet")
            self.fig.colorbar(self.plot,cax=self.colorbarax2)
            
            if (not self.polar):
                # add x-axis label
                if (self.xVals==Dimension.ETA1):
                    self.ax.set_xlabel("r [m]")
                elif (self.xVals==Dimension.ETA2):
                    self.ax.set_xlabel(r'$\theta$ [rad]')
                elif (self.xVals==Dimension.ETA3):
                    self.ax.set_xlabel("z [m]")
                elif (self.xVals==Dimension.ETA4):
                    self.ax.set_xlabel(r'v [$ms^{-1}$]')
                # add y-axis label
                if (self.yVals==Dimension.ETA1):
                    self.ax.set_ylabel("r [m]")
                elif (self.yVals==Dimension.ETA2):
                    self.ax.set_ylabel(r'$\theta$ [rad]')
                elif (self.yVals==Dimension.ETA3):
                    self.ax.set_ylabel("z [m]")
                elif (self.yVals==Dimension.ETA4):
                    self.ax.set_ylabel(r'v [$ms^{-1}$]')
            else:
                self.ax.set_xlabel("x [m]")
                self.ax.set_ylabel("y [m]")
            self.fig.canvas.draw()
            
        else:
            self.grid.getSliceFromDict(d,self.comm,self.drawRank)
    
    def show(self):
        plt.show()

