from src.Utilities.DiscreteSlider import DiscreteSlider
from src import Grid
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
import warnings

class SlicePlotter4d(object):
    def __init__(self,grid: Grid):
        self.grid = grid
        
        # get MPI values
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        
        # get initial values for the sliders
        self.vVal = grid.vVals[len(grid.vVals)//2]
        self.zVal = grid.zVals[0]
        
        # get max and min values of f to avoid colorbar jumps
        self.minimum=grid.getMin()
        self.maximum=grid.getMax()
        
        if (self.rank==0):
            self.fig = plt.figure()
            self.fig.canvas.mpl_connect('close_event', self.handle_close)
            self.ax = self.fig.add_axes([0.1, 0.25, 0.7, 0.7],)
            self.sliderax1 = self.fig.add_axes([0.1, 0.02, 0.6, 0.03],)
            self.sliderax2 = self.fig.add_axes([0.1, 0.12, 0.6, 0.03],)
            self.colorbarax2 = self.fig.add_axes([0.85, 0.1, 0.03, 0.8],)

            # add sliders and remember their values
            self.vPar = DiscreteSlider(self.sliderax1, r'$v_\parallel$', valinit=self.vVal,values=grid.vVals)
            self.Z = DiscreteSlider(self.sliderax2, 'z', valinit=self.zVal, values=grid.zVals)
            self.vPar.on_changed(self.updateV)
            self.Z.on_changed(self.updateZ)
            
            # get coordinate values
            theta = np.repeat(np.append(self.grid.thetaVals,self.grid.thetaVals[0]),
                    len(self.grid.rVals)).reshape(len(self.grid.thetaVals)+1,len(self.grid.rVals))
            r = np.tile(self.grid.rVals,len(self.grid.thetaVals)+1).reshape(len(self.grid.thetaVals)+1,len(self.grid.rVals))
            
            self.x=r*np.cos(theta)
            self.y=r*np.sin(theta)
        
        self.plotFigure()

    def updateV(self, value):
        # alert non-0 ranks that v has been updated and the drawing must be updated
        MPI.COMM_WORLD.bcast(1,root=0)
        self.vVal=value
        # broadcast new value
        self.vVal=MPI.COMM_WORLD.bcast(self.vVal,root=0)
        # update plot
        self.updateDraw()

    def updateZ(self, value):
        # alert non-0 ranks that z has been updated and the drawing must be updated
        MPI.COMM_WORLD.bcast(2,root=0)
        self.zVal=value
        # broadcast new value
        self.zVal=MPI.COMM_WORLD.bcast(self.zVal,root=0)
        # update plot
        self.updateDraw()
    
    def updateDraw(self):
        self.plotFigure()
    
    def plotFigure(self):
        # get the slice with v and z values as indicated by the sliders
        theSlice = self.grid.getSlice(v=self.vVal,z=self.zVal)
        
        if (self.rank==0):
            self.fig.canvas.draw()
            
            # remove the old plot
            self.ax.clear()
            self.colorbarax2.clear()
            self.plot = self.ax.pcolormesh(self.x,self.y,theSlice,vmin=self.minimum,vmax=self.maximum)
            self.fig.colorbar(self.plot,cax=self.colorbarax2)
            self.fig.canvas.draw()
    
    def show(self):
        # set-up non-0 ranks as listeners so they can react to interactions with the plot on rank 0
        if (self.rank!=0):
            stop=1
            while (stop!=0):
                stop=MPI.COMM_WORLD.bcast(1,root=0)
                if (stop==1):
                    self.vVal=MPI.COMM_WORLD.bcast(self.vVal,root=0)
                    self.updateDraw()
                elif (stop==2):
                    self.zVal=MPI.COMM_WORLD.bcast(self.zVal,root=0)
                    self.updateDraw()
        plt.show()
    
    def handle_close(self,evt):
        # broadcast 0 to break non-0 ranks listen loop
        MPI.COMM_WORLD.bcast(0,root=0)

class SlicePlotter3d(object):
    def __init__(self,grid: Grid, *args, **kwargs):
        self.grid = grid
        
        # use arguments to get coordinate values (including omitted coordinate)
        if (len(args)==0):
            # if no arguments are provided then assume (r,θ,z)
            self.xVals=Grid.Dimension.R
            self.x=grid.rVals
            self.yVals=Grid.Dimension.THETA
            self.y=np.append(grid.thetaVals,grid.thetaVals[0])
            self.zVals=Grid.Dimension.Z
            self.omit=Grid.Dimension.V
            self.omitVal=grid.vVals[0]
        else:
            assert(len(args)==3)
            # use arguments to get x values
            if (args[0]=='r'):
                self.xVals=Grid.Dimension.R
                self.x=grid.rVals
            elif (args[0]=='q' or args[0]=='theta'):
                self.xVals=Grid.Dimension.THETA
                self.x=np.append(grid.thetaVals,grid.thetaVals[0])
            elif (args[0]=='z'):
                self.xVals=Grid.Dimension.Z
                self.x=grid.zVals
            elif (args[0]=='v'):
                self.xVals=Grid.Dimension.V
                self.x=grid.vVals
            else:
                raise TypeError("%s is not a valid dimension" % self.xVals)
            
            # use arguments to get y values
            if (args[1]=='r'):
                self.yVals=Grid.Dimension.R
                self.y=grid.rVals
            elif (args[1]=='q' or args[1]=='theta'):
                self.yVals=Grid.Dimension.THETA
                self.y=np.append(grid.thetaVals,grid.thetaVals[0])
            elif (args[1]=='z'):
                self.yVals=Grid.Dimension.Z
                self.y=grid.zVals
            elif (args[1]=='v'):
                self.yVals=Grid.Dimension.V
                self.y=grid.vVals
            else:
                raise TypeError("%s is not a valid dimension" % self.yVals)
            
            # use arguments to get z values
            if (args[2]=='r'):
                self.zVals=Grid.Dimension.R
            elif (args[2]=='q' or args[2]=='theta'):
                self.zVals=Grid.Dimension.THETA
            elif (args[2]=='z'):
                self.zVals=Grid.Dimension.Z
            elif (args[2]=='v'):
                self.zVals=Grid.Dimension.V
            else:
                raise TypeError("%s is not a valid dimension" % self.zVals)
            
            # get unused dimensions identifier
            self.omit=(set(Grid.Dimension)-set([self.xVals,self.yVals,self.zVals])).pop()
            
            # fix unused dimension at an arbitrary (but measured) value
            if (self.omit==Grid.Dimension.R):
                self.omitVal=grid.rVals[0]
            elif (self.omit==Grid.Dimension.THETA):
                self.omitVal=grid.thetaVals[0]
            elif (self.omit==Grid.Dimension.Z):
                self.omitVal=grid.zVals[0]
            elif (self.omit==Grid.Dimension.V):
                self.omitVal=grid.vVals[0]
        
        # save x and y grid values
        nx=len(self.x)
        ny=len(self.y)
        self.x = np.repeat(self.x,ny).reshape(nx,ny)
        self.y = np.tile(self.y,nx).reshape(nx,ny)
        
        # if x is sooner in the list of dimensions than y then the slice will need to be trasposed
        self.reverse=self.xVals>self.yVals
        
        # if (x,y) are (r,θ) or (θ,r) then print in polar coordinates
        self.polar=False
        if (self.xVals==Grid.Dimension.R and self.yVals==Grid.Dimension.THETA):
            x=self.x*np.cos(self.y)
            y=self.x*np.sin(self.y)
            self.x=x
            self.y=y
            self.polar=True
        elif (self.yVals==Grid.Dimension.R and self.xVals==Grid.Dimension.THETA):
            y=self.x*np.cos(self.y)
            x=self.x*np.sin(self.y)
            self.x=x
            self.y=y
            self.polar=True
        
        # get MPI values
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        
        # get max and min values of f to avoid colorbar jumps
        self.minimum=grid.getMin()
        self.maximum=grid.getMax()
        
        # on rank 0 set-up the graph
        if (self.rank==0):
            self.fig = plt.figure()
            self.fig.canvas.mpl_connect('close_event', self.handle_close)
            self.ax = self.fig.add_axes([0.1, 0.25, 0.7, 0.7],)
            self.sliderax1 = self.fig.add_axes([0.1, 0.02, 0.6, 0.03],)
            self.colorbarax2 = self.fig.add_axes([0.85, 0.1, 0.03, 0.8],)

            # add slider and remember its value
            if (self.zVals==Grid.Dimension.R):
                self.initVal = grid.rVals[0]
                self.slider = DiscreteSlider(self.sliderax1, "r", valinit=self.initVal,values=grid.rVals)
            elif (self.zVals==Grid.Dimension.THETA):
                self.initVal = grid.thetaVals[0]
                self.slider = DiscreteSlider(self.sliderax1, r'$\theta$', valinit=self.initVal,values=grid.thetaVals)
            elif (self.zVals==Grid.Dimension.Z):
                self.initVal = grid.zVals[0]
                self.slider = DiscreteSlider(self.sliderax1, "z", valinit=self.initVal,values=grid.zVals)
            elif (self.zVals==Grid.Dimension.V):
                self.initVal = grid.vVals[len(grid.vVals)//2]
                self.slider = DiscreteSlider(self.sliderax1, r'$v_\parallel$', valinit=self.initVal,values=grid.vVals)
            self.slider.on_changed(self.updateVal)
            if (not self.polar):
                # add x-axis label
                if (self.xVals==Grid.Dimension.R):
                    self.ax.set_xlabel("r [m]")
                elif (self.xVals==Grid.Dimension.THETA):
                    self.ax.set_xlabel(r'$\theta$ [rad]')
                elif (self.xVals==Grid.Dimension.Z):
                    self.ax.set_xlabel("z [m]")
                elif (self.xVals==Grid.Dimension.V):
                    self.ax.set_xlabel(r'v [$ms^{-1}$]')
                
                # add y-axis label
                if (self.yVals==Grid.Dimension.R):
                    self.ax.set_ylabel("r [m]")
                elif (self.yVals==Grid.Dimension.THETA):
                    self.ax.set_ylabel(r'$\theta$ [rad]')
                elif (self.yVals==Grid.Dimension.Z):
                    self.ax.set_ylabel("z [m]")
                elif (self.yVals==Grid.Dimension.V):
                    self.ax.set_ylabel(r'v [$ms^{-1}$]')
        else:
            # on other ranks save the initial value for the slider
            if (self.zVals==Grid.Dimension.R):
                self.initVal = grid.rVals[0]
            elif (self.zVals==Grid.Dimension.THETA):
                self.initVal = grid.thetaVals[0]
            elif (self.zVals==Grid.Dimension.Z):
                self.initVal = grid.zVals[0]
            elif (self.zVals==Grid.Dimension.V):
                self.initVal = grid.vVals[len(grid.vVals)//2]
            
        self.plotFigure()
    
    def updateVal(self, value):
        # alert non-0 ranks that they must receive the new value and update the drawing
        MPI.COMM_WORLD.bcast(1,root=0)
        self.initVal=value
        # broadcast new value
        self.initVal=MPI.COMM_WORLD.bcast(self.initVal,root=0)
        # update plot
        self.updateDraw()
    
    def updateDraw(self):
        self.plotFigure()
    
    def plotFigure(self):
        # get slice by passing dictionary containing fixed dimensions and their values
        d = {self.zVals : self.initVal, self.omit : self.omitVal}
        theSlice = self.grid.getSliceFromDict(d)
        
        if (self.rank==0):
            self.fig.canvas.draw()
            
            if (hasattr(self,'plot')):
                # remove the old plot
                del self.plot
                del self.ax.get_children()[0]
            self.colorbarax2.clear()
            if (self.reverse):
                self.plot = self.ax.pcolormesh(self.x,self.y,np.transpose(theSlice),vmin=self.minimum,vmax=self.maximum)
            else:
                self.plot = self.ax.pcolormesh(self.x,self.y,theSlice,vmin=self.minimum,vmax=self.maximum)
            self.fig.colorbar(self.plot,cax=self.colorbarax2)
            self.fig.canvas.draw()
    
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
    
    def handle_close(self,evt):
        # broadcast 0 to break non-0 ranks listen loop
        MPI.COMM_WORLD.bcast(0,root=0)

class Plotter2d(object):
    def __init__(self,grid: Grid, *args, **kwargs):
        # use arguments to get x values
        assert(len(args)==2)
        if (args[0]=='r'):
            self.xVals=Grid.Dimension.R
            self.x=grid.rVals
        elif (args[0]=='q' or args[0]=='theta'):
            self.xVals=Grid.Dimension.THETA
            self.x=np.append(grid.thetaVals,grid.thetaVals[0])
        elif (args[0]=='z'):
            self.xVals=Grid.Dimension.Z
            self.x=grid.zVals
        elif (args[0]=='v'):
            self.xVals=Grid.Dimension.V
            self.x=grid.vVals
        else:
            raise TypeError("%s is not a valid dimension" % self.xVals)
        
        # use arguments to get y values
        if (args[1]=='r'):
            self.yVals=Grid.Dimension.R
            self.y=grid.rVals
        elif (args[1]=='q' or args[0]=='theta'):
            self.yVals=Grid.Dimension.THETA
            self.y=np.append(grid.thetaVals,grid.thetaVals[0])
        elif (args[1]=='z'):
            self.yVals=Grid.Dimension.Z
            self.y=grid.zVals
        elif (args[1]=='v'):
            self.yVals=Grid.Dimension.V
            self.y=grid.vVals
        else:
            raise TypeError("%s is not a valid dimension" % self.yVals)
        self.grid = grid
        
        # get unused dimensions identifiers
        omitted=(set(Grid.Dimension)-set([self.xVals,self.yVals]))
        self.omit1=omitted.pop()
        self.omit2=omitted.pop()
        
        # fix first unused dimension at an arbitrary (but measured) value
        if (self.omit1==Grid.Dimension.R):
            self.omitVal1=grid.rVals[0]
        elif (self.omit1==Grid.Dimension.THETA):
            self.omitVal1=grid.thetaVals[0]
        elif (self.omit1==Grid.Dimension.Z):
            self.omitVal1=grid.zVals[0]
        elif (self.omit1==Grid.Dimension.V):
            self.omitVal1=grid.vVals[0]
        
        # fix second unused dimension at an arbitrary (but measured) value
        if (self.omit2==Grid.Dimension.R):
            self.omitVal2=grid.rVals[0]
        elif (self.omit2==Grid.Dimension.THETA):
            self.omitVal2=grid.thetaVals[0]
        elif (self.omit2==Grid.Dimension.Z):
            self.omitVal2=grid.zVals[0]
        elif (self.omit2==Grid.Dimension.V):
            self.omitVal2=grid.vVals[0]
        
        # get MPI vals
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        
        # get max and min values of f to avoid colorbar jumps
        self.minimum=grid.getMin()
        self.maximum=grid.getMax()
        
        # save x and y grid values
        nx=len(self.x)
        ny=len(self.y)
        self.x = np.repeat(self.x,ny).reshape(nx,ny)
        self.y = np.tile(self.y,nx).reshape(nx,ny)
        
        # if x is sooner in the list of dimensions than y then the slice will need to be trasposed
        self.reverse=self.xVals>self.yVals
        
        # if (x,y) are (r,θ) or (θ,r) then print in polar coordinates
        self.polar=False
        if (self.xVals==Grid.Dimension.R and self.yVals==Grid.Dimension.THETA):
            x=self.x*np.cos(self.y)
            y=self.x*np.sin(self.y)
            self.x=x
            self.y=y
            self.polar=True
        elif (self.yVals==Grid.Dimension.R and self.xVals==Grid.Dimension.THETA):
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
                if (self.xVals==Grid.Dimension.R):
                    self.ax.set_xlabel("r [m]")
                elif (self.xVals==Grid.Dimension.THETA):
                    self.ax.set_xlabel(r'$\theta$ [rad]')
                elif (self.xVals==Grid.Dimension.Z):
                    self.ax.set_xlabel("z [m]")
                elif (self.xVals==Grid.Dimension.V):
                    self.ax.set_xlabel(r'v [$ms^{-1}$]')
                # add y-axis label
                if (self.yVals==Grid.Dimension.R):
                    self.ax.set_ylabel("r [m]")
                elif (self.yVals==Grid.Dimension.THETA):
                    self.ax.set_ylabel(r'$\theta$ [rad]')
                elif (self.yVals==Grid.Dimension.Z):
                    self.ax.set_ylabel("z [m]")
                elif (self.yVals==Grid.Dimension.V):
                    self.ax.set_ylabel(r'v [$ms^{-1}$]')
        
        self.plotFigure()
    
    def plotFigure(self):
        # get slice by passing dictionary containing fixed dimensions and their values
        d = {self.omit1 : self.omitVal1, self.omit2 : self.omitVal2}
        theSlice = self.grid.getSliceFromDict(d)
        
        if (self.rank==0):
            self.fig.canvas.draw()
            
            self.colorbarax2.clear()
            if (self.reverse):
                self.plot = self.ax.pcolormesh(self.x,self.y,np.transpose(theSlice),vmin=self.minimum,vmax=self.maximum)
            else:
                self.plot = self.ax.pcolormesh(self.x,self.y,theSlice,vmin=self.minimum,vmax=self.maximum)
            self.fig.colorbar(self.plot,cax=self.colorbarax2)
            self.fig.canvas.draw()
    
    def show(self):
        plt.show()

