from timeflux.core.branch import Branch

from timeflux.core.node import Node
import pandas as pd


class Concat(Node):
    """ Concat list of data ports.
    """

    # todo: should be in core as a 'concat'?

    def __init__(self, axis=1, **kwargs):
        self._axis = axis
        self._kwargs = kwargs

    def update(self):
        ports = list(self.iterate("i*"))
        i_data = [port.data for (name, _, port) in ports if port.data is not None]
        if i_data:
            self.o.data = pd.concat(i_data, axis=self._axis, **self._kwargs)


class AddSuffix(Node):
    ''' Suffix column names with string suffix.
    Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame.

    Args:
        suffix (str): The string to add after each column

    '''

    # todo: should be in core as a 'pandas_helpers'?
    def __init__(self, suffix):
        super().__init__()
        self._suffix = suffix

    def update(self):
        if self.i.ready():
            self.o.meta = self.i.meta
            self.o.data = self.i.data.add_suffix(self._suffix)


class FilterBank(Branch):
    """ Apply multiple IIR Filters to the signal and stack the components horizontally

        Attributes:
        i (Port): Default input, expects DataFrame.
        o (Port): Default output, provides DataFrame.

    Args:
        rate (float): Nominal sampling rate of the input data. If None, rate is get
                  from the meta.
        filters (dict|None): Define the iir filter to apply given its name and its params.

        TODO: decide whether user specifies parameters of each filter or just the
              center frequency and the width.

        TODO: choose whether we want:
            - Behavior 1: An output port will be created with the given names as suffix.
            - Behavior 2: A new set of channels will be created with the given names as suffix.
    """

    def __init__(self, filters, method='IIRFilter', rate=None, **kwargs):
        super().__init__()
        self._filters = filters

        graph = {
            'nodes': [],
            'edges': []
        }
        graph['nodes'].append({
            'id': 'stack',
            'module': 'timeflux_bci.nodes.ssvep',
            'class': 'Concat',
        })

        for filter_name, filter_params in self._filters.items():
            filter_params.update({'rate': rate})
            filter_params.update(kwargs)
            iir = {
                'id': filter_name,
                'module': 'timeflux_dsp.nodes.filters',
                'class': method,
                'params': filter_params
            }
            rename_columns = {
                'id': f'rename_{filter_name}',
                'module': 'timeflux_bci.nodes.ssvep',
                'class': 'AddSuffix',
                'params': {'suffix': f'_{filter_name}'}
            }
            graph['nodes'] += [iir, rename_columns]
            graph['edges'] += [{'source': filter_name, 'target': f'rename_{filter_name}'},
                               {'source': f'rename_{filter_name}', 'target': f'stack:{filter_name}'}]

        self.load(graph)

    def update(self):
        # When we have not received data, there is nothing to do
        if not self.i.ready():
            return
        # set the data in input of each filter
        for filter_name in self._filters.keys():
            self.set_port(filter_name, port_id='i', data=self.i.data, meta=self.i.meta)

        self.run()

        self.o = self.get_port('stack', port_id='o')
        # TODO, choose between Behavior 1 and 2 -> if 1, then map output of iirs to output of node
