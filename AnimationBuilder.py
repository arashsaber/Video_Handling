#!/usr/bin/python3
"""
The file contains a matplotlib based player object 
Copyright (c) 2018
Licensed under the MIT License (see LICENSE for details)
Written by Arash Tehrani
"""
#   ---------------------------------------------------

import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import mpl_toolkits.axes_grid1
import matplotlib.widgets


from mpl_toolkits.mplot3d import Axes3D


import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
#plt.style.use('ggplot')

#   -----------------------------------------------
class Player(FuncAnimation):
    """
    The class makes a player with play, stop, and next buttons and a frame slider.
    """
    def __init__(self, fig, func, init_func=None, fargs=None,
                 save_count=None, button_color='yellow', bg_color='red', dis_start=0,
                 dis_stop=100, pos=(0.125, 0.05), **kwargs):
        """
        initialization
        :param fig: matplotlib fifure object
        :param func: user-defined function which takes a integer (frame number) as an input
        :param init_func: user-defined initial function used by the FuncAnimation class
        :param fargs: arguments of func, used by FuncAnimation class
        :param save_count: save count arg used by FuncAnimation class
        :param button_color: string, color of the buttons of the player
        :param bg_color: string, hovercolor of the buttons and slider
        :param dis_start: int, start frame number
        :param dis_stop: int, stop frame number
        :param pos: length 2 tuple, position of the buttons
        :param kwargs: kwargs for FuncAnimation class
        """
        # setting up the index
        self.start_ind = dis_start
        self.stop_ind = dis_stop
        self.dis_length = self.stop_ind - self.start_ind
        self.ind = self.start_ind

        self.runs = True
        self.forwards = True
        self.fig = fig
        self.fig.set_facecolor('k')
        self.button_color = button_color
        self.bg_color = bg_color

        self.func = func
        self.setup(pos)
        FuncAnimation.__init__(self, self.fig, self.func, frames=self.play(),
                                           init_func=init_func, fargs=fargs,
                                           save_count=save_count, **kwargs)

    @property
    def ind(self):
        return self._ind
    
    @ind.setter
    def ind(self, val):
        self._ind = val
        self._ind -= self.start_ind
        self._ind %= (self.dis_length)
        self._ind += self.start_ind

    
    def play(self):
        """
        play function
        """
        while self.runs:

            self.ind = self.ind + self.forwards - (not self.forwards)
            self._update()

            yield self.ind
        
        
    def start(self):
        self.runs = True
        self._update()
        self.event_source.start()

    def stop(self, event=None):
        self.runs = False
        self._update()
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
        if self.forwards:
            self.ind += 1
        else:
            self.ind -= 1

        self.func(self.ind)

        self._update()
        self.fig.canvas.draw_idle()

    def _update(self):
        self.slider.set_val(self.ind)

    def __set_slider(self, val):
            val = int(val)
            self.ind = val
            #self.func(self.ind)
            
    def setup(self, pos):
        """
        Setting up the buttons and the slider
        :param pos: length 2 tuple, position of the axes for buttons and tuples
        :return:
        """
        playerax = self.fig.add_axes([pos[0], pos[1], 0.22, 0.04])
        divider = mpl_toolkits.axes_grid1.make_axes_locatable(playerax)
        bax = divider.append_axes("right", size="80%", pad=0.05)
        sax = divider.append_axes("right", size="80%", pad=0.05)
        fax = divider.append_axes("right", size="80%", pad=0.05)
        ofax = divider.append_axes("right", size="100%", pad=0.05)
        sliderax = self.fig.add_axes((pos[0], pos[1] - 0.045, 0.5, 0.04), facecolor=self.bg_color)  # 'lemonchiffon')

        self.button_oneback = matplotlib.widgets.Button(playerax, color=self.button_color,
                                                        hovercolor=self.bg_color, label='$\u29CF$')  # , label=r'$\u29CF$')
        self.button_back = matplotlib.widgets.Button(bax, color=self.button_color,
                                                     hovercolor=self.bg_color, label='$\u25C0$')  # r'$\u25C0$')
        self.button_stop = matplotlib.widgets.Button(sax, color=self.button_color,
                                                     hovercolor=self.bg_color, label='$\u25A0$')  # , label=r'$\u25A0$')
        self.button_forward = matplotlib.widgets.Button(fax, color=self.button_color,
                                                        hovercolor=self.bg_color, label='$\u25B6$')  # , label=r'$\u25B6$')
        self.button_oneforward = matplotlib.widgets.Button(ofax, color=self.button_color,
                                                           hovercolor=self.bg_color, label='$\u29D0$')  # , label=r'$\u29D0$')
        self.button_oneback.on_clicked(self.onebackward)
        self.slider = Slider(sliderax, label='', valfmt='%0.0f',
                             valmin=0, valmax=self.stop_ind - 1,
                             valinit=self.ind, color='black', fc=self.button_color)  # , snap='True')
        self.slider.label.set_color(self.button_color)
        # self.slider.valtext.set_color(self.button_color)
        self.slider.valtext.set_position((0.5, 0.5))

        self.slider.set_val(self.ind)

        self.button_back.on_clicked(self.backward)
        self.button_stop.on_clicked(self.stop)
        self.button_forward.on_clicked(self.forward)
        self.button_oneforward.on_clicked(self.oneforward)
        self.slider.on_changed(self.__set_slider)

#   -----------------------------------------------
if __name__ == '__main__':
    from matplotlib import cm

    data = np.random.randn(100, 200, 10)
    _x = np.arange(data.shape[1])
    _y = np.arange(data.shape[0])
    _X, _Y = np.meshgrid(_x, _y)

    Z = np.zeros((data.shape[0], data.shape[1]), dtype=float)
    # plt.style.use('ggplot')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #fig.set_facecolor('k')

    def update(ind):
        ind = int(ind)
        print(ind)
        ax.clear()
        Z = data[:, :, ind]

        ax.contour(_X, _Y, Z, zdir='z',
                   offset=np.amin(Z) + 0.1 * np.amin(Z), cmap='coolwarm')
        ax.plot_surface(_X, _Y, Z, rstride=1, cstride=1,
                        cmap='coolwarm',  # 'RdGy_r',
                        color='black', shade=False, antialiased=False, alpha=0.6)

    ani = Player(fig, update, dis_start=0, dis_stop=10)
    plt.show()

#   -----------------------------------------------