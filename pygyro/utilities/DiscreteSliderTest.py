from DiscreteSlider import DiscreteSlider
import matplotlib.pyplot as plt
import numpy as np

class ChangingPlot(object):
    def __init__(self):
        self.inc = 0.5

        self.fig, self.ax = plt.subplots()
        self.sliderax = self.fig.add_axes([0.2, 0.02, 0.6, 0.03],
                                          facecolor='yellow')

        vals=np.arange(0,10)
        self.slider = DiscreteSlider(self.sliderax, 'Value', valinit=2,values=vals)
        self.slider.on_changed(self.update)

        x = np.arange(0, 10.5, self.inc)
        self.ax.plot(x, x, 'ro')
        self.dot, = self.ax.plot(self.inc, self.inc, 'bo', markersize=18)

    def update(self, value):
        self.dot.set_data([[value],[value]])
        self.fig.canvas.draw()

    def show(self):
        plt.show()

p = ChangingPlot()
p.show()
