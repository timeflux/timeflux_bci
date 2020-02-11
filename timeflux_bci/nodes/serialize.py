from timeflux.core.node import Node
import json


class SerializeColumn(Node):

    def __init__(self, column='data'):
        self._column = column

    def update(self):
        if not self.i.ready():
            return
        self.o = self.i
        self.o.data[self._column] = self.o.data[self._column].apply(lambda d: json.dumps(d) )

class DeserializeColumn(Node):

    def __init__(self, column='data'):
        self._column = column

    def update(self):
        if not self.i.ready():
            return
        self.o = self.i
        self.o.data[self._column] = self.o.data[self._column].apply(lambda d: json.loads(d) if isinstance(d, str) else d)
