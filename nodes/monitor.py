import tkinter
from timeflux.core.node import Node
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from abc import ABC
import numpy as np
import pandas as pd
import time
from typing import Tuple, List

class Monitor(Node, ABC):
    """This is an abstract base class that provides monitoring capability for timeflux nodes. Every data 
    frame passed to the monitor is expected to have time points along the initial axis."""

    def __init__(self, name: str, time_frames_in_buffer: int, title: str, y_label: str, y_lim: Tuple[float, float] = None, legend: List[str] = None, width: int=7, height: int=4, is_visible: bool = True) -> object:
        """Constructor for this class.
        
        Inputs:
        - name: The name assigned to this node.
        - time_frames_in_buffer: Integer >=0. The number of time frames of streamed data that should stay in buffer.
        - title: Title of the figure.
        - y_label: The label to be shown on the vertical axis.
        - y_lim: The limits of the y-axis. If None then they are inferred based on the range in the data.
        - legend: The legend to be used. If None then it is ignored. Is not applicate to all monitors.
        - width: The width of the window.
        - height: The height of the widnow.
        - is_visible: Indicates whether the window should be shown on screen.

        Outputs:
        - self: The initialized instance."""

        # Super
        super(Monitor, self).__init__()
        
        # Copy attributes
        self.time_frames_in_buffer = time_frames_in_buffer
        self.name = name
        self.__is_visible__ = is_visible
        self.legend = legend
        self.y_lim = y_lim

        if is_visible:
            # Create a window
            self.__window__ = tkinter.Tk()
            self.__window__.wm_title("")
            self.__window__.protocol("WM_DELETE_WINDOW", self.__close_window__)

            # Create an empty figure
            figure = Figure(figsize=(width, height), dpi=100)
            self.__ax__ = figure.add_subplot()
            self.__ax__.set_title(title)
            self.__ax__.set_xlabel("Time Frames")
            self.__ax__.set_ylabel(y_label)
            
            # Draw figure in canvas
            self.__canvas__ = FigureCanvasTkAgg(figure, master=self.__window__)  # A tk.DrawingArea.
            self.__canvas__.draw()
            self.__canvas__.get_tk_widget().pack(side=tkinter.TOP, fill=tkinter.BOTH, expand=True)

            # Show canvas on window
            self.__window__.update()

    def __close_window__(self):
        """Ensures the window is closed properly."""
        del self.__canvas__
        self.__window__.destroy()

    def __draw_buffer__(self) -> None:
        """This method draws the buffer onto the figure."""
        pass

    def update(self) -> None:
        if self.__is_visible__:
            # Extract input
            if type(self.i.data) == pd.DataFrame:
                new_data = self.i.data.to_numpy()
            elif type(self.i.data) == np.ndarray:
                new_data = self.i.data
            elif self.i.data == None: return
            else:
                self.logger.info(f"{self.name} received unsupported data type {type(self.i.data)}.")
                return
            
            # Propagate data through buffer
            if hasattr(self, "__buffer__"): 
                self.__buffer__ = np.concatenate([self.__buffer__, new_data], axis=0)
                self.__buffer__ = self.__buffer__[-self.time_frames_in_buffer:]
            else: 
                self.__buffer__ = np.zeros([self.time_frames_in_buffer] + list(new_data.shape[1:]), dtype=new_data.dtype)
                self.__buffer__[self.time_frames_in_buffer - new_data.shape[0]:] = new_data 

            # Plot the new data
            self.__draw_buffer__()

            # Update canvas
            if hasattr(self, "__canvas__"): self.__canvas__.draw()
            self.__window__.update()

    def __init_buffer__(self):
        """This function initializes the buffer."""
        self.__buffer__

class Plot(Monitor):
    """Provides a monitor window for line graph streams. It allows to plot multiple lines simultaneously."""
    def __draw_buffer__(self) -> None:
        # Reset old plot
        self.__ax__.set_prop_cycle(None)
        if hasattr(self.__ax__, "lines"): self.__ax__.lines = []
        
        # Plot new data
        self.__ax__.plot(np.arange(-self.__buffer__.shape[0],0), self.__buffer__[:])
        
        # Set axes limits
        if self.y_lim != None: self.__ax__.set_ylim(bottom=self.y_lim[0], top=self.y_lim[1])
        else: self.__ax__.set_ylim(bottom=np.min(self.__buffer__), top=np.max(self.__buffer__))
        self.__ax__.set_xlim(left=-self.__buffer__.shape[0], right=0)

        # Set legend
        if self.legend != None: self.__ax__.legend(self.legend)

class Imshow(Monitor):
    """Provides a monitor window a image streams."""
    def __draw_buffer__(self) -> None:
        self.__ax__.imshow(self.__buffer__.transpose())

class Text(Monitor):
    """Provides a monitor window a text stream."""

    def __draw_buffer__(self) -> None:
        # Reset old text
        if hasattr(self.__ax__, "texts"): self.__ax__.texts = [] 

        for t, time_frame in enumerate(self.__buffer__[1:]):
            if time_frame != self.__buffer__[t]:
                self.__ax__.text(t-self.time_frames_in_buffer,0.5,time_frame)

        self.__ax__.set_xlim(left=-self.time_frames_in_buffer, right=0)
        self.__ax__.set_yticks([])

if __name__ == "__main__":

    mode = 'Plot'
    
    # Demonstrate Plot
    if mode == 'Plot':
        node = Plot(name='plot node', time_frames_in_buffer=100, title="Sinusoids", y_label="Amplitude")
        y = np.sin(np.array([np.pi*np.arange(0,10,0.1), np.pi*0.5*np.arange(0,10,0.1)])).T
        for i in range(8):
            node.i.data = y[i*10:(i+1)*10,:]
            node.update()
            time.sleep(1)

    # Demonstrate Imshow
    if mode == 'Imshow':
        node = Imshow(name='imshow node', time_frames_in_buffer=100, title="Noise", y_label="Features")
        for i in range(8):
            node.i.data = np.random.rand(10,30)
            node.update()
            time.sleep(1)

    # Demonstrate text
    if mode == 'Text':
        node = Text(name='text node', time_frames_in_buffer=100, title='Words', y_label="", height=2)
        words = np.array(["Hello"] * 50 + ["World"] * 20 + ["!"] * 10, dtype='U')
        for i in range(8):
            node.i.data = words[i*10:(i+1)*10]
            node.update()
            time.sleep(1)

    time.sleep(2)
