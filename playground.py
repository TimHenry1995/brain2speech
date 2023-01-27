import numpy as np
import pandas as pd
class dummy():
    def __init__(self):
        self.__previous_loss__ = None

    def __did_loss_change__(self, losses: pd.Series) -> bool:
        """Mutating method that determines whether the loss changed between the last time frame of the previous loss slice and any of the current time frames.
        
        Precondition:
        - self.__previous_loss__ may be None or a float.

        Inputs:
        - losses: the current series of losses. Maybe be empty.
        
        Postcondition:
        - self.__previous_loss__ is be None (if this loss series was empty) or equal to the last entry of losses (if this losses contained an entry)."""
        
        # Extract values
        losses = losses.values

        loss_changed = False
        # If losses contains rows
        if len(losses):
            # Check if previously seen loss is different from the first new loss
            if self.__previous_loss__ != None: loss_changed = self.__previous_loss__ != losses[0]

            # Check if there is a change within the new losses
            zeros = np.zeros(len(losses)-1)
            loss_changed = loss_changed or not np.allclose(zeros, losses[:-1] - losses[1:])

            # Update previous loss
            self.__previous_loss__ = losses[-1]

        # Outputs
        return loss_changed

df = pd.DataFrame({'train':[1,2,3,4,5,6], 'validation':[1,1,1,1,1,1]})
dum = dummy()
loss_changed = dum.__did_loss_change__(df['validation'])
print(loss_changed)

df = pd.DataFrame({'train':[1,2,3,4,5,6], 'validation':[2,2,2,2,2,1]})
loss_changed = dum.__did_loss_change__(df['validation'])
print(loss_changed)

df = pd.DataFrame({'train':[1,2,3,4,5,6], 'validation':[1,1,1,1,1,1]})
loss_changed = dum.__did_loss_change__(df['validation'])
print(loss_changed)