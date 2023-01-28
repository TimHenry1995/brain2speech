import tkinter
from timeflux.core.node import Node
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from abc import ABC
import numpy as np
import pandas as pd
import time, torch
from typing import Tuple, List

class Monitor(Node, ABC):
    """This is an abstract base class that provides monitoring capability for timeflux nodes. It Every data 
    frame passed to the monitor is expected to have time points along the initial axis.
    These nodes expect a certain number of input streams (see child class description) of type numpy.ndarray or pandas.DataFrame.
    They do not have output streams."""

    def __init__(self, name: str, time_frames_in_buffer: int, title: str, y_label: str, y_lower: float = None, y_upper: float = None, width: int=7, height: int=4, is_visible: bool = True) -> object:
        """Constructor for this class.
        
        Inputs:
        - name: The name assigned to this node.
        - time_frames_in_buffer: Integer >=0. The number of time frames of streamed data that should stay in buffer.
        - title: Title of the figure.
        - y_label: The label to be shown on the vertical axis.
        - y_lower, y_upper: The limits to the y-axis given separately. If not specified then each is inferred from the data stream.
        - width: The width of the window.
        - height: The height of the widnow.
        - is_visible: Indicates whether the window should be shown on screen.

        Outputs:
        - self: The initialized instance."""

        # Super
        super(Monitor, self).__init__()
        
        # Copy attributes
        self.name = name
        self.time_frames_in_buffer = time_frames_in_buffer
        self.__is_visible__ = is_visible
        self.y_lower = y_lower
        self.y_upper = y_upper

        if is_visible:
            # Create a window
            self.__window__ = tkinter.Tk()
            self.__window__.wm_title("")
            self.__window__.protocol("WM_DELETE_WINDOW", self.__close_window__)

            # Create an empty figure
            self.__figure__ = Figure(figsize=(width, height), dpi=100)
            self.__figure__.tight_layout()
            self.__ax__ = self.__figure__.add_subplot()
            self.__ax__.set_title(title)
            self.__ax__.set_xlabel("Time Frames")
            self.__ax__.set_ylabel(y_label)
            
            # Draw figure in canvas
            self.__canvas__ = FigureCanvasTkAgg(self.__figure__, master=self.__window__)  # A tk.DrawingArea.
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
        if self.__is_visible__ and self.i.ready():
            # Extract input
            if type(self.i.data) == pd.DataFrame:
                new_data = self.i.data
            elif type(self.i.data) == np.ndarray:
                new_data = pd.DataFrame(self.i.data)
            elif self.i.data == None: return
            else:
                self.logger.info(f"{self.name} received unsupported data type {type(self.i.data)}.")
                return
            
            # Initialize buffer
            if not hasattr(self, "__buffer__"):  self.__buffer__ = pd.DataFrame(np.zeros([self.time_frames_in_buffer] + list(new_data.shape[1:])))
            
            # Update column names
            self.__buffer__.columns = new_data.columns

            # Propagate new data
            self.__buffer__ = pd.concat([self.__buffer__, new_data], axis=0, ignore_index=True)
            self.__buffer__ = self.__buffer__.iloc[-self.time_frames_in_buffer:,:]

            # Plot the new data
            self.__draw_buffer__()
            self.__figure__.tight_layout()
            # Update canvas
            if hasattr(self, "__canvas__"): self.__canvas__.draw()
            self.__window__.update()
            
        else: return super().update()

    def __init_buffer__(self):
        """This function initializes the buffer."""
        self.__buffer__

class Plot(Monitor):
    """Provides a monitor window for a line graph. It expects one stream of numeric data. 
    If the stream contains multiple columns then this node plots multiple lines simultaneously.
    It creates a legend using the column names."""

    def __draw_buffer__(self) -> None:
        # Reset old plot
        self.__ax__.set_prop_cycle(None)
        if hasattr(self.__ax__, "lines"): self.__ax__.lines = []
        
        # Plot new data
        self.__ax__.plot(np.arange(-self.__buffer__.shape[0],0), self.__buffer__.values)
        
        # Set axes limits
        y_lim = [self.y_lower, self.y_upper]
        if self.y_lower is None: y_lim[0] = 1.2 * np.min(self.__buffer__.values)
        if self.y_upper is None: y_lim[1] = 1.2 * np.max(self.__buffer__.values)
        self.__ax__.set_ylim(bottom=y_lim[0], top=y_lim[1])
        self.__ax__.set_xlim(left=-self.__buffer__.shape[0], right=0)

        # Set legend
        self.__ax__.legend(list(self.__buffer__.columns))

class Imshow(Monitor):
    """Provides a monitor window a single image stream."""

    def __draw_buffer__(self) -> None:
        self.__ax__.imshow(np.flipud(np.array(self.__buffer__.values).transpose()))

class Text(Monitor):
    """Provides a monitor window a single text stream."""

    def __draw_buffer__(self) -> None:
        # Reset old text
        if hasattr(self.__ax__, "texts"): self.__ax__.texts = [] 

        for t, time_frame in enumerate(self.__buffer__.iloc[1:,0].values.tolist()):
            if time_frame != self.__buffer__.iloc[t,0]:
                self.__ax__.text(t-self.time_frames_in_buffer,0.1,time_frame, rotation=90)

        self.__ax__.set_xlim(left=-self.time_frames_in_buffer, right=0)

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
