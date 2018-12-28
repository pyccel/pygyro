import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Polygon

class BoundedSlider:
    def __init__(self,ax,name,values,*args,**kwargs):
        self.ax=ax
        self.moving = None
        self.name = name
        
        ax.tick_params(bottom=False,top=False,left=False,right=False,labelbottom=False,labelleft=False)
        
        self.poss_vals = values
        
        self.n = len(self.poss_vals)-1
        self.midpoint = self.n//2
        if ('valinit' in kwargs):
            self.idx = kwargs['valinit']
        else:
            self.idx = self.midpoint
        self.min_idx = max(0,self.idx-2)
        self.max_idx = min(self.n,self.idx+2)
        
        self.fix_min = self.min_idx
        self.fix_max = self.max_idx
        
        self.start_poly = np.array([[0,0],[1,0],[1,0.3],[0.7,0.3],[0.7,0.7],[1,0.7],[1,1],[0,1]])*[0.5/self.n,1]
        self.  end_poly = np.array([[0,0],[-1,0],[-1,0.3],[-0.7,0.3],[-0.7,0.7],[-1,0.7],[-1,1],[0,1]])*[0.5/self.n,1]
        
        self.grayed_s = Rectangle((0,0), self.min_idx/self.n, 1,facecolor='gray')
        self.grayed_e = Rectangle((self.max_idx/self.n,0), (self.n-self.max_idx)/self.n, 1,facecolor='gray')
        self.filled = Rectangle((self.min_idx/self.n,0), (self.idx-self.min_idx)/self.n, 1,facecolor='blue')
        self.start_box = Polygon(self.start_poly + [self.min_idx/self.n,0],facecolor='red')
        self.end_box = Polygon(self.end_poly + [self.max_idx/self.n,0],facecolor='red')
        
        # Add patch to axes
        ax.add_patch(self.grayed_s)
        ax.add_patch(self.grayed_e)
        ax.add_patch(self.filled)
        ax.add_patch(self.start_box)
        ax.add_patch(self.end_box)
        
        self.min_text = plt.Text(0,0.5,'{:.2f} '.format(self.min_val),
                            horizontalalignment='right',
                            verticalalignment='center')
        self.max_text = plt.Text(1.0,0.5,' {:.2f}'.format(self.max_val),
                            horizontalalignment='left',
                            verticalalignment='center')
        
        ax.add_image(self.min_text)
        ax.add_image(self.max_text)
        
        ax.set_xlabel('{} : {:.2f} '.format(name,self.val))
        
        cid = ax.figure.canvas.mpl_connect('button_press_event', self.on_press)
        cid = ax.figure.canvas.mpl_connect('button_release_event', self.on_release)
        cid = ax.figure.canvas.mpl_connect('motion_notify_event', self.on_move)
        
        self.changed = lambda val: None
    
    def on_changed(self,lam):
        self.changed = lam
    
    def on_press(self,event):
        if (event.inaxes==self.ax):
            if (self.start_box.contains_point((event.x,event.y))):
                self.moving = 'start'
            elif (self.end_box.contains_point((event.x,event.y))):
                self.moving = 'end'
            else:
                self.moving = 'val'
                x_start_idx = event.xdata
        else:
            self.moving = None
    
    def on_move(self,event):
        if (event.inaxes==self.ax):
            x_idx = np.round(event.xdata * self.n)
            if (self.moving == 'start'):
                if (x_idx!=self.min_idx):
                    self.min_idx = x_idx
                    self.start_box.set_xy(self.start_poly + [self.min_idx/self.n,0])
                    self.ax.figure.canvas.draw()
            elif (self.moving == 'end'):
                if (x_idx!=self.max_idx):
                    self.max_idx = x_idx
                    self.end_box.set_xy(self.end_poly + [self.max_idx/self.n,0])
                    self.ax.figure.canvas.draw()
            elif (self.moving == 'val'):
                if (x_idx!=self.idx and self.inMemory(x_idx)):
                    self.idx = x_idx
                    self.filled.set_width((self.idx-self.fix_min)/self.n)
                    self.ax.figure.canvas.draw()
                    self.changed(self.idx)
    
    def on_release(self,event):
        if (event.inaxes==self.ax):
            x_idx = np.round(event.xdata * self.n)
            if (self.moving == 'start'):
                self.min_idx = x_idx
                self.start_box.set_xy(self.start_poly + [self.min_idx/self.n,0])
                self.ax.figure.canvas.draw()
            if (self.moving == 'end'):
                self.max_idx = x_idx
                self.end_box.set_xy(self.end_poly + [self.max_idx/self.n,0])
                self.ax.figure.canvas.draw()
            elif (self.moving == 'val'):
                if (x_idx!=self.idx and self.inMemory(x_idx)):
                    self.idx = x_idx
                    self.filled.set_width((self.idx-self.fix_min)/self.n)
                    self.ax.figure.canvas.draw()
                self.changed(self.idx)
        self.moving = None
    
    def inMemory(self,idx):
        return (idx<=self.fix_max and idx>=self.fix_min)
    
    def reset_bounds(self):
        self.fix_max = self.max_idx
        self.fix_min = self.min_idx
        if (self.idx<self.fix_min):
            self.idx = self.fix_min
        elif (self.idx>self.fix_max):
            self.idx = self.fix_max
        self.grayed_s.set_width(self.min_idx/self.n)
        self.grayed_e.set_bounds(self.max_idx/self.n,0, (self.n-self.max_idx)/self.n, 1)
        self.filled.set_bounds(self.min_idx/self.n,0, (self.idx-self.min_idx)/self.n, 1)
        return (self.fix_min,self.fix_max)

    def get_next_n_poss_pts(self):
        return self.max_idx-self.min_idx+1

    @property
    def idx(self):
        return self._idx

    @property
    def min_idx(self):
        return self._min_idx

    @property
    def max_idx(self):
        return self._max_idx
    
    @min_idx.setter
    def min_idx(self, x):
        self._min_idx = int(x)
        self.min_val = self.poss_vals[self.min_idx]

    @max_idx.setter
    def max_idx(self, x):
        self._max_idx = int(x)
        self.max_val = self.poss_vals[self.max_idx]

    @idx.setter
    def idx(self, x):
        self._idx = int(x)
        self.val = self.poss_vals[self.idx]
        self.ax.set_xlabel('{} : {:.2f} '.format(self.name,self.val))
        
