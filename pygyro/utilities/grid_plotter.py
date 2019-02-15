from mpi4py                     import MPI
from math                       import pi
from enum                       import IntEnum
from matplotlib.widgets         import Button, CheckButtons
from matplotlib.gridspec        import GridSpec, GridSpecFromSubplotSpec
from matplotlib                 import rc
from mpl_toolkits.axes_grid1    import make_axes_locatable
import matplotlib.pyplot        as plt
import numpy                    as np
import sched,time

from .discrete_slider import DiscreteSlider
from .bounded_slider import BoundedSlider
from ..model.grid     import Grid

class SlicePlotterNd(object):
    def __init__(self,grid: Grid, xDimension: int, yDimension: int,
                polar: bool, sliderDimensions: list = [], sliderNames = [], **kwargs):
        self.comm = kwargs.pop('comm',MPI.COMM_WORLD)
        self.comm_size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.drawRank = kwargs.pop('drawingRank',0)
        
        self.grid = grid
        
        self.nprocs = grid._layout_manager.nProcs
        
        # use arguments to get coordinate values (including omitted coordinate)
        self.xDim = xDimension
        self.yDim = yDimension
        self.sDims = sliderDimensions
        self.nSliders = len(self.sDims)
        self.sliderVals = [0]*self.nSliders
        assert(len(sliderNames)==self.nSliders)
        self.x = grid.eta_grid[self.xDim]
        self.y = grid.eta_grid[self.yDim]
        
        # Get omitted dimensions
        self.oDims = list(set(range(len(grid.eta_grid)))-set([*self.sDims,self.xDim,self.yDim]))
        
        # Get fixed values in non-plotted dimensions
        if ('fixValues' in kwargs):
            self.omitVals = kwargs.pop('fixValues')
            assert(len(self.omitVals)==len(self.oDims))
        else:
            self.omitVals = [self.grid.nGlobalCoords[d]//2 for d in self.oDims]
        # Create a dictionary from the omitted values to feed to the grid
        self.selectDict = dict(zip([*self.oDims,*self.sDims],[*self.omitVals,*self.sliderVals]))
        self.selectInfo = np.array([0 for r in range(3*len(self.sliderVals))])
        self.nDims = len(grid.eta_grid)
        
        # Select how to access the printed values
        self.access_pattern = [0 for i in range(len(grid.eta_grid))]
        self.access_pattern[self.xDim] = slice(0,len(self.x))
        self.access_pattern[self.yDim] = slice(0,len(self.y))
        self.memory_requirements = [1 for i in range(len(grid.eta_grid))]
        self.memory_requirements[self.xDim] = grid.eta_grid[self.xDim].size
        self.memory_requirements[self.yDim] = grid.eta_grid[self.yDim].size
        
        ## Shift the x/y values to centre the plotted squares on the data
        
        # Get sorted values to avoid wrong dx values (found at periodic boundaries)
        xSorted = np.sort(self.x)
        ySorted = np.sort(self.y)
        
        dx = xSorted[1:] - xSorted[:-1]
        dy = ySorted[1:] - ySorted[:-1]
        
        # The corner of the square should be shifted by (-dx/2,-dy/2) for the value
        # to be found in the middle
        shift = [dx[0]*0.5, *(dx[:-1]+dx[1:])*0.25, dx[-1]*0.5]
        # The values are reordered to the original ordering to shift the relevant values
        shift = [shift[i] for i in np.argsort(self.x)]
        self.x = self.x - shift
        
        # Ditto for y
        shift = [dy[0]*0.5, *(dy[:-1]+dy[1:])*0.25, dy[-1]*0.5]
        shift = [shift[i] for i in np.argsort(self.y)]
        self.y = self.y - shift
        
        # save x and y grid values
        nx=len(self.x)
        ny=len(self.y)
        
        # if (x,y) are (r,θ) then print in polar coordinates
        self.polar=polar
        if (polar):
            self.x = np.repeat(self.x,ny+1).reshape(nx,ny+1)
            self.y = np.tile([*self.y,self.y[0]],nx).reshape(nx,ny+1)
            x=self.x*np.cos(self.y)
            y=self.x*np.sin(self.y)
            self.x=x
            self.y=y
            
            self.xLab = "x"
            self.yLab = "y"
        else:
            self.x = np.repeat(self.x,ny).reshape(nx,ny)
            self.y = np.tile(self.y,nx).reshape(nx,ny)
            
            self.xLab = "r"
            self.yLab = r'$\theta$ [rad]'
        
        # get max and min values of f to avoid colorbar jumps
        minimum=grid.getMin(self.drawRank,self.oDims,self.omitVals)
        maximum=grid.getMax(self.drawRank,self.oDims,self.omitVals)
        
        # on rank 0 set-up the graph
        if (self.rank==self.drawRank):
            plt.ion()
            self.open=True
            self.fig = plt.figure()
            self.fig.canvas.mpl_connect('close_event', self.handle_close)
            #Grid spec in Grid spec to guarantee size of plot
            gs_orig = GridSpec(1, 2, width_ratios = [9,1])
            gs = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_orig[0], height_ratios = [3,1],hspace=0.3)
            gs_plot = GridSpecFromSubplotSpec(1, 1, subplot_spec=gs[0])
            gs_slider = GridSpecFromSubplotSpec(self.nSliders, 1, subplot_spec=gs[1],hspace=1)
            gs_buttons = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_orig[1])
            
            

            self.ax = self.fig.add_subplot(gs_plot[0])
            divider = make_axes_locatable(self.ax)
            self.colorbarax = divider.append_axes("right",size="5%",pad=0.05)
            self.slider_axes = [self.fig.add_subplot(gs_slider[i]) for i in range(self.nSliders)]
            self.button_axes = [self.fig.add_subplot(gs_buttons[i]) for i in range(4)]
            # ~ rc('font',size=30)
            self.button_axes[2].axis('off')
            self.button_axes[3].axis('off')
            self.button_axes[3].text(0,0.66,'Memory\nrequired:',verticalalignment='center',fontsize='medium')
            self.buttons = [Button(self.button_axes[0],u"\u25b6"),
                            Button(self.button_axes[1],u"\u25b6\u258e"),
                            CheckButtons(self.button_axes[2],['Fix\n max/min\n colour'],[True]),
                            self.button_axes[3].text(0,0.33,'txt',verticalalignment='center',fontsize='large')]
            self.playing = False
            self.mpi_playing = False
            self.buttons[0].on_clicked(self.play_pause)
            self.buttons[1].on_clicked(self.stepForward)
            self.buttons[2].on_clicked(self.fixBounds)
            self.buttons[2].labels[0].set_fontsize('large')
            self.buttons[2].rectangles[0].set_width(0.2)
            dat = self.buttons[2].lines[0][0].get_data()
            self.buttons[2].lines[0][0].set_data([0.05,0.25],dat[1])
            self.buttons[2].lines[0][1].set_data([0.25,0.05],dat[1])
            
            self.sliders = []
            
            for i in range(self.nSliders):
                # add slider and remember its value
                slider = BoundedSlider(self.slider_axes[i], sliderNames[i],
                        values=grid.eta_grid[self.sDims[i]])
                slider.grid_dimension = self.sDims[i]
                slider.on_changed(lambda val,dim=self.sDims[i]: self.updateVal(val,dim))
                slider.on_mem_changed(lambda idx=i,dim=self.sDims[i]: self.updateMem(idx,dim))
                s_min,s_max = slider.reset_bounds()
                self.selectDict[slider.grid_dimension] = range(s_min,s_max+1)
                self.memory_requirements[self.sDims[i]] = slider.get_next_n_poss_pts()
                self.sliders.append(slider)
            
            gs_orig.tight_layout(self.fig,pad=1.0)
            
            self.plotParams = {'vmin':minimum,'vmax':maximum, 'cmap':"jet"}
            
            self.setMemText()
            
            self.completedRanks = [i==self.drawRank for i in range(self.comm_size)]
        
        self.getData()
        self.action = 1
        
        if (self.rank == self.drawRank):
            self.plotFigure()
            self.prepare_data_reception()
    
    def setLabels(self,xLab,yLab):
        self.xLab=xLab
        self.yLab=yLab
        self.useLabels()
    
    def useLabels(self):
        if (self.rank==self.drawRank):
            # add x-axis label
            self.ax.set_xlabel(self.xLab)
            # add y-axis label
            self.ax.set_ylabel(self.yLab)
    
    def updateVal(self, value, dimension):
        self.fig.canvas.stop_event_loop()
        if (self.playing):
            self.access_pattern[dimension] = value - self.selectDict[dimension].start
        else:
            self.comm.bcast(4,root=0)
            self.getData()
        self.plotFigure()
        self.fig.canvas.start_event_loop(timeout = 1)
    
    def updateMem(self,idx,dim):
        self.memory_requirements[dim] = self.sliders[idx].get_next_n_poss_pts()
        self.setMemText()
    
    def setMemText(self):
        requirements = np.prod(self.memory_requirements)*8.0
        units = int(np.log2(requirements)/10)
        if (units == 0):
            units = 'B'
        elif (units==1):
            requirements/=2**10
            units = 'KB'
        elif (units==2):
            requirements/=2**20
            units = 'MB'
        else:
            requirements/=2**30
            units = 'GB'
        self.buttons[3].set_text('{:.4g} {}'.format(requirements,units))
    
    def handle_close(self,_evt):
        assert(self.rank==self.drawRank)
        self.action = 0
        self.open=False
        if (self.playing):
            self.prepare_data_reception()
            self.fig.canvas.stop_event_loop()
        else:
            self.fig.canvas.stop_event_loop()
            self.comm.bcast(self.action,root=0)
    
    def prepare_data_reception(self):
        self.fig.canvas.stop_event_loop()
        for j in range(1,self.comm_size):
            i = self.comm.recv(tag=2510)
            self.completedRanks[i] = True
        self.comm.Barrier()
        self.getData()
        self.comm.bcast(self.action,root=0)
        if (self.action==1):
            self.mpi_playing = False
        self.plotFigure()
        self.fig.canvas.start_event_loop(timeout = 1)
    
    def play_pause(self,evt):
        self.playing = not self.playing
        for slider in self.sliders:
            slider.toggle_active_bounds()
            slider.reset_bounds()
        if (self.playing):
            self.buttons[0].label.set_text(u"\u258e\u258e")
            self.action = 5
            self.buttons[1].set_active(False)
            if (not self.mpi_playing):
                self.mpi_playing = True
                if (self.action!=2):
                    self.comm.bcast(self.action,root=0)
            self.action = 3
            for i,d in enumerate(self.sDims):
                self.access_pattern[d] = self.sliders[i].idx - self.sliders[i].fix_min
                self.selectInfo[i*3+1] = self.sliders[i].fix_min
                self.selectInfo[i*3+2] = self.sliders[i].fix_max
            self.getData()
        else:
            self.buttons[0].label.set_text(u"\u25b6")
            self.action = 1
            self.buttons[1].set_active(True)
            for d in self.sDims:
                self.access_pattern[d] = 0
        self.fig.canvas.draw()
    
    def stepForward(self,evt):
        self.buttons[1].set_active(False)
        self.action = 2
        self.fig.canvas.stop_event_loop()
        self.comm.bcast(self.action,root=0)
        self.mpi_playing = True
        self.fig.canvas.start_event_loop(timeout = 1)
    
    def fixBounds(self,evt):
        if ('vmin' in self.plotParams):
            self.plotParams.pop('vmin')
            self.plotParams.pop('vmax')
        else:
            # get max and min values of f to avoid colorbar jumps
            minimum=self.grid.getMin(self.drawRank,self.oDims,self.omitVals)
            maximum=self.grid.getMax(self.drawRank,self.oDims,self.omitVals)
            self.plotParams['vmin'] = minimum
            self.plotParams['vmax'] = maximum
    
    def getData(self):
        if (self.rank!=self.drawRank):
            self.comm.Bcast(self.selectInfo,self.drawRank)
            for i in range(self.nSliders):
                self.selectDict[self.selectInfo[i*3]] = range(self.selectInfo[i*3+1],self.selectInfo[i*2+2])
            self.grid.getBlockFromDict(self.selectDict,self.comm,self.drawRank)
        else:
            if (self.playing):
                for i,slider in enumerate(self.sliders):
                    s_min,s_max = slider.reset_bounds()
                    self.selectDict[slider.grid_dimension] = range(s_min,s_max+1)
                    self.selectInfo[i*3] = slider.grid_dimension
                    self.selectInfo[i*2+1] = s_min
                    self.selectInfo[i*2+2] = s_max+1
            else:
                for i,slider in enumerate(self.sliders):
                    val = slider.idx
                    self.selectDict[slider.grid_dimension] = range(val,val+1)
                    self.selectInfo[i*3] = slider.grid_dimension
                    self.selectInfo[i*2+1] = val
                    self.selectInfo[i*2+2] = val+1
            self.comm.Bcast(self.selectInfo,self.drawRank)
            
            layout, starts, mpi_data, theSlice = self.grid.getBlockFromDict(self.selectDict,self.comm,self.drawRank)
            
            split = np.split(theSlice,starts[1:])
            
            nprocs = [max([mpi_data[i][j] for i in range(len(mpi_data))])+1 for j in range(len(mpi_data[0]))]
            
            concatReady = np.ndarray(tuple(nprocs),dtype=object)
            
            for l,ranks in enumerate(mpi_data):
                layout_shape = list(layout.shape)
                visited=False
                for j,d in enumerate(ranks):
                    layout_shape[j]=layout.mpi_lengths(j)[d]
                
                n_ranks = len(ranks)
                for k in self.selectDict.keys():
                    j = layout.inv_dims_order[k]
                    if (j<n_ranks):
                        start = layout.mpi_starts(j)[ranks[j]]
                        end = start + layout.mpi_lengths(j)[ranks[j]]
                        if (isinstance(self.selectDict[k],int)):
                            val = self.selectDict[k]
                            if start <= val and end > val:
                                layout_shape[j]=1
                            else:
                                layout_shape[j]=0
                        else:
                            range_start = self.selectDict[k].start
                            range_stop  = self.selectDict[k].stop
                            range_start = max(range_start - start,0)
                            range_stop  = min(range_stop - start, layout.mpi_lengths(j)[ranks[j]])
                            layout_shape[j] = max(range_stop-range_start,0)
                    else:
                        if (isinstance(self.selectDict[k],int)):
                            layout_shape[j]=1
                        else:
                            layout_shape[j]=self.selectDict[k].stop-self.selectDict[k].start
                concatReady[tuple(ranks)]=split[l].reshape(layout_shape)
            
            zone = [range(n) for n in nprocs]
            zone.pop()
            
            for i in range(len(nprocs)-1,0,-1):
                toConcat = np.ndarray(tuple(nprocs[:i]),dtype=object)
                
                coords = [0 for n in zone]
                for d in range(len(zone)):
                    for j in range(nprocs[d]):
                        toConcat[tuple(coords)]=np.concatenate(concatReady[tuple(coords)].tolist(),axis=i)
                        coords[d]+=1
                concatReady = toConcat
                zone.pop()
            
            toConcat = np.ndarray(tuple(nprocs[:i]),dtype=object)
            
            self._data = np.concatenate(concatReady.tolist(),axis=0)
            
            self._data = self._data.transpose(layout.dims_order)
    
    def plotFigure(self):
        assert(self.rank==self.drawRank)
        
        theSlice = self._data[tuple(self.access_pattern)]
        
        # Plot the new data, adding an extra row of data for polar plots
        # (to avoid having a missing segment), and transposing the data
        # if the storage dimensions are ordered differently to the plotting
        # dimensions
        if (self.xDim>self.yDim):
            if (self.polar):
                theSlice = np.append(theSlice, theSlice[None,0,:],axis=0)
            theSlice = theSlice.T
        else:
            if (self.polar):
                theSlice = np.append(theSlice, theSlice[:,0,None],axis=1)
            theSlice = theSlice
        
        # remove the old plot
        self.ax.clear()
        self.colorbarax.clear()
        
        # Plot the new data, adding an extra row of data for polar plots
        # (to avoid having a missing segment), and transposing the data
        # if the storage dimensions are ordered differently to the plotting
        # dimensions
        self.plot = self.ax.pcolormesh(self.x,self.y,theSlice,**self.plotParams)
        self.fig.colorbar(self.plot,cax=self.colorbarax)
        
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        
        self.fig.canvas.draw()
        
    def show(self):
        while (self.open):
            self.checkProgress()
            self.fig.canvas.start_event_loop(timeout = 1)
    
    def calculation_complete(self):
        assert(self.rank!=self.drawRank)
        self.comm.send(self.rank,self.drawRank,tag=2510)
        self.comm.Barrier()
        self.getData()
        return self.runPauseLoopSlave()
    
    def runPauseLoopSlave(self):
        action = 1
        while(action==1):
            ac=0
            action = self.comm.bcast(ac,root=0)
            if action == 0:
                # close
                return False
            elif action == 1:
                # pause
                pass
            elif action == 2:
                # step
                pass
            elif action == 3:
                # play
                pass
            elif action == 4:
                # collect data (in pause)
                self.getData()
                action = 1
            elif action == 5:
                # first play
                self.getData()
                action = 3
                pass
        return True
    
    def checkProgress(self):
        assert(self.rank==self.drawRank)
        if (self.comm.Iprobe(tag=2510)):
            if (self.action == 2):
                self.action = 1
                self.prepare_data_reception()
                self.mpi_playing = False
                self.buttons[1].set_active(True)
            else:
                self.prepare_data_reception()
            for i in range(len(self.completedRanks)):
                self.completedRanks[i] = (i==self.drawRank)
        if (not self.mpi_playing):
            self.comm.bcast(self.action,root=0)

