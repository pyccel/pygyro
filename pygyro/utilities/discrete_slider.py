from matplotlib.widgets import Slider
import numpy as np

class DiscreteSlider(Slider):
    """A matplotlib slider widget with discrete steps."""
    def __init__(self, *args, **kwargs):
        """Identical to Slider.__init__, except for the "values" kwarg.
        "values" specifies the values that the slider will be discretised to."""
        self.vals = kwargs.pop('values', [])
        kwargs['valmin']=self.vals.min()
        kwargs['valmax']=self.vals.max()
        Slider.__init__(self, *args, **kwargs)

    def set_val(self, val):
        idx = (np.abs(val-self.vals)).argmin()
        discrete_val = self.vals[idx]
        # We can't just call Slider.set_val(self, discrete_val), because this 
        # will prevent the slider from updating properly (it will get stuck at
        # the first step and not "slide"). Instead, we'll keep track of the
        # the continuous value as self.val and pass in the discrete value to
        # everything else.
        xy = self.poly.xy
        xy[2] = discrete_val, 1
        xy[3] = discrete_val, 0
        self.poly.xy = xy
        self.valtext.set_text(self.valfmt % discrete_val)
        self.val = val
        if not self.eventson: 
            return
        for _cid, func in self.observers.items():
            func(discrete_val)
    
    def _update(self, event):
        super(DiscreteSlider, self)._update(event)
        i = int(self.val)
        if i >=self.valmax:
            return
