from timeflux.core.node import Node


class Predictor(Node):

    def __init__(self, name: str) -> object:
        """Constructor for this class.
        
        Inputs:
        - name: The name assigned to this node.
        

        Outputs:
        - self: The initialized instance."""
        
        # Super
        super(Predictor, self).__init__()

        # Copy attributes
        self.name = name