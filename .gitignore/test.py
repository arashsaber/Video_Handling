import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
'''
plt.style.use('ggplot')

fig, ax = plt.subplots()
plt.subplots_adjust(left=0.25, bottom=0.25)
t = np.arange(0.0, 1.0, 0.001)
a0 = 5
f0 = 3
s = a0*np.sin(2*np.pi*f0*t)
l, = plt.plot(t,s, lw=2, color='red')
plt.axis([0, 1, -10, 10])

axcolor = 'lightgoldenrodyellow'
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])#, axisbg=axcolor)
axamp  = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

sfreq = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0)
samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)

def update(val):
    amp = samp.val
    freq = sfreq.val
    l.set_ydata(amp*np.sin(2*np.pi*freq*t))
    fig.canvas.draw_idle()
sfreq.on_changed(update)
samp.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
def reset(event):
    sfreq.reset()
    samp.reset()
button.on_clicked(reset)

rax = plt.axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)
def colorfunc(label):
    l.set_color(label)
    fig.canvas.draw_idle()
radio.on_clicked(colorfunc)

plt.show()




import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

freqs = np.arange(2, 20, 3)

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
t = np.arange(0.0, 1.0, 0.001)
s = np.sin(2*np.pi*freqs[0]*t)
l, = plt.plot(t, s, lw=2)


class Index(object):
    ind = 0

    def next(self, event):
        self.ind += 1
        i = self.ind % len(freqs)
        ydata = np.sin(2*np.pi*freqs[i]*t)
        l.set_ydata(ydata)
        plt.draw()

    def prev(self, event):
        self.ind -= 1
        i = self.ind % len(freqs)
        ydata = np.sin(2*np.pi*freqs[i]*t)
        l.set_ydata(ydata)
        plt.draw()

callback = Index()
axprev = plt.axes([0.6, 0.05, 0.2, 0.075])
axnext = plt.axes([0.8, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(callback.next)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(callback.prev)

plt.show()


"""
============
3D animation
============

A simple example of an animated plot... In 3D!
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


def Gen_RandLine(length, dims=2):
    """
    Create a line using a random walk algorithm

    length is the number of points for the line.
    dims is the number of dimensions the line has.
    """
    lineData = np.empty((dims, length))
    lineData[:, 0] = np.random.rand(dims)
    for index in range(1, length):
        # scaling the random numbers by 0.1 so
        # movement is small compared to position.
        # subtraction by 0.5 is to change the range to [-0.5, 0.5]
        # to allow a line to move backwards.
        step = ((np.random.rand(dims) - 0.5) * 0.1)
        lineData[:, index] = lineData[:, index - 1] + step

    return lineData


def update_lines(num, dataLines, lines):
    for line, data in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    return lines

# Attaching 3D axis to the figure
fig = plt.figure()
ax = p3.Axes3D(fig)

# Fifty lines of random 3-D lines
data = [Gen_RandLine(25, 3) for index in range(50)]

# Creating fifty line objects.
# NOTE: Can't pass empty arrays into 3d version of plot()
lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1])[0] for dat in data]

# Setting the axes properties
ax.set_xlim3d([0.0, 1.0])
ax.set_xlabel('X')

ax.set_ylim3d([0.0, 1.0])
ax.set_ylabel('Y')

ax.set_zlim3d([0.0, 1.0])
ax.set_zlabel('Z')

ax.set_title('3D Test')

# Creating the Animation object
line_ani = animation.FuncAnimation(fig, update_lines, 25, fargs=(data, lines),
                                   interval=50, blit=False)

plt.show()
'''

#   -----------------------------------------------
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets
import matplotlib as mpl

from mpl_toolkits.mplot3d import Axes3D


import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
#plt.style.use('ggplot')
#   -----------------------------------------------
class Player(FuncAnimation):
    def __init__(self, fig, func, init_func=None, fargs=None,
                 save_count=None, button_color='yellow', dis_start=0,
                 dis_stop=100, pos=(0.125, 0.05), **kwargs):

        # setting up the index
        self.start_ind = dis_start
        self.stop_ind = dis_stop
        self.dis_length = self.stop_ind - self.start_ind
        self.ind = self.start_ind

        self.runs = True
        self.forwards = True
        self.fig = fig
        self.button_color = button_color

        self.func = func
        self.setup(pos)
        FuncAnimation.__init__(self, self.fig, self.func, frames=self.play(),
                                           init_func=init_func, fargs=fargs,
                                           save_count=save_count, **kwargs)

    def play(self):
        while self.runs:

            self.ind = self.ind + self.forwards - (not self.forwards)
            self.ind -= self.start_ind
            self.ind %= (self.dis_length)
            self.ind += self.start_ind
            self._update()

            yield self.ind

    def start(self):
        self.runs = True
        self._update()
        self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        #self._update()
        self.event_source.stop()

    def forward(self, event=None):
        self.forwards = True
        self.start()

    def backward(self, event=None):
        self.forwards = False
        self.start()

    def oneforward(self, event=None):
        self.forwards = True
        self.onestep()

    def onebackward(self, event=None):
        self.forwards = False
        self.onestep()

    def onestep(self):
        if self.forward:
            self.ind += 1
        else:
            self.ind -= 1

        self.ind -= self.start_ind
        self.ind %= (self.dis_length)
        self.ind += self.start_ind
        self.func(self.ind)

        self._update()
        self.fig.canvas.draw_idle()

    def _update(self):
        self.slider.set_val(self.ind)

    def setup(self, pos):
        bg_color = 'red'
        playerax = self.fig.add_axes([pos[0], pos[1], 0.22, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        sliderax = self.fig.add_axes((pos[0], pos[1] - 0.045, 0.5, 0.04), axisbg=bg_color)#'lemonchiffon')

        self.button_oneback = matplotlib.widgets.Button(playerax, color=self.button_color,
                                                        hovercolor=bg_color, label='$\u29CF$')#, label=r'$\u29CF$')
        self.button_back = matplotlib.widgets.Button(bax, color=self.button_color,
                                                     hovercolor=bg_color, label='$\u25C0$')#r'$\u25C0$')
        self.button_stop = matplotlib.widgets.Button(sax, color=self.button_color,
                                                     hovercolor=bg_color, label='$\u25A0$')#, label=r'$\u25A0$')
        self.button_forward = matplotlib.widgets.Button(fax, color=self.button_color,
                                                        hovercolor=bg_color, label='$\u25B6$')#, label=r'$\u25B6$')
        self.button_oneforward = matplotlib.widgets.Button(ofax, color=self.button_color,
                                                           hovercolor=bg_color, label='$\u29D0$')#, label=r'$\u29D0$')
        self.button_oneback.on_clicked(self.onebackward)
        self.slider = Slider(sliderax, label='', valfmt='%0.0f',
                             valmin=0, valmax=self.stop_ind - 1,
                             valinit=self.ind, color='black', fc=self.button_color)#, snap='True')
        self.slider.label.set_color(self.button_color)
        #self.slider.valtext.set_color(self.button_color)
        self.slider.valtext.set_position((0.5,0.5))

        self.slider.set_val(self.ind)

        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneforward.on_clicked(self.oneforward)

        def __set_slider(val):
            val = int(val)
            self.ind = val
            self.func(val)
        self.slider.on_changed(__set_slider)


if __name__ == '__main__':
    """
    from matplotlib import cm

    data = np.random.randn(50, 50, 10)
    _x = np.arange(data.shape[0])
    _y = np.arange(data.shape[1])
    _X, _Y = np.meshgrid(_x, _y)

    Z = np.zeros((data.shape[0], data.shape[1]), dtype=float)
    #plt.style.use('ggplot')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    fig.set_facecolor('k')

    def update(ind):
        ind = int(ind)
        ax.clear()
        Z = data[:, :, ind]

        ax.contour(_X, _Y, Z, zdir='z',
                   offset=np.amin(Z) + 0.1*np.amin(Z), cmap='coolwarm')
        ax.plot_surface(_X, _Y, Z, rstride=1, cstride=1,
                        cmap='coolwarm',#'RdGy_r',
                        color='black', shade=False, antialiased=False, alpha=0.6)

    ani = Player(fig, update, dis_start=0, dis_stop=10)
    plt.show()
    """

    desc = "We're looking for a Data Scientist/Machine Learning Engineer with a wide range of " \
        "competencies. Rainforest is changing the way people do QA and data is at the heart " \
        "of how we do that. From crowd management and fraud detection to data visualization " \
        "and automation research, there are myriad opportunities to use your creativity and " \
        "technical skills. We are looking for someone who learns quickly and is a great " \
        "communicator. Since we are a remote, distributed development team, decent writing skills and(over)communication is important to us. You can be based anywhere in the world (including San Francisco, where our HQ is located). We regularly send our data scientists to conferences, both to speak and just learn (e.g. last year we went to NIPS and KDD, this year to Europython and a couple of PyDatas). You can read about some of our work on predicting test run durations and how Kaggle can be useful in the real world. And hear some of our team members speak on the topic of Women, Company Culture, and Remote Teams here."
    words = [len(w.strip('.,)(')) for w in desc.split()]
    median = np.median(words)
    print(median)









