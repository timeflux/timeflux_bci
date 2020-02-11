"""Handle meta """

from timeflux.helpers.port import match_events
from timeflux.core.node import Node


class BetweenEvents(Node):
    """ Add metadata extracted from opening/closing events
    Attributes:
        i (Port): Continuous data input, expects DataFrame.
        i_events (Port): Event input, expects DataFrame.
        o (Port): Continuous data output, provides DataFrame and meta
    Args:
        event_start (string):
        event_stop (string):
    todo: this should actually be the job of node `Gate`. (to be migrated in core instead?)
          with data transmission optional.
    """

    def __init__(self, event_start, event_stop, passthrough=False):
        """"""
        self._spreading = False
        self._context = {}
        self.event_start = event_start
        self.event_stop = event_stop
        self._passthrough = passthrough

    def update(self):
        # if meta spreading has not started yet
        if not self._spreading:
            # look for `event_start`
            matches = match_events(self.i_events, self.event_start)
            if matches is not None:
                # todo: what if multiple matches in same chunk?
                self._context = {'context': matches.data.values[0]}
                self._spreading = True
        else:
            # look for `event_stop`
            matches = match_events(self.i_events, self.event_stop)
            if matches is not None:
                # do not spread  meta to the last chunk (todo: make it optional?)
                self._context = {}
                self._spreading = False

        if self._passthrough or self._spreading:
            # If passthrough is True transmit data whatever teh status
            # If status is spreading, transmit data
             self.o = self.i # todo: allow no transmission when outside the start/stop events? (to avoid future warning in line https://github.com/timeflux/timeflux/blob/ec2b368ae304597f94337fbd60cd276e89cc24ce/timeflux/nodes/ml.py#L276)
        else:
            self.o.clear()
            return
        if self._spreading:
            # todo: truncate based on sequences start/stop times, or?
            # eventually, enrich meta output port with context
            self.o.meta.update(self._context)
