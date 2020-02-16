import json

import numpy as np
import pandas as pd
from timeflux.core.node import Node
from timeflux.helpers.port import match_events


class NaiveBayesInference(Node):
    r""" Target prediction using a Naive Bayesian Inference

    Let Ait be the set of symbols illuminated for the ith flash for symbol t in the sequence,
    we define flash scores as follows:

    .. math::

        \begin{equation}
            f(yit∣xt)=
                \begin{cases}
                    p(yi, True),  \text{if xt ∈ Ait}. \\
                    p(yi, False), \text{if xt ∉ Ait}.
                \end{cases}
        \end{equation}

    Args:
        output_label (string)
        reset_label (string|None):
        threshold (float|None):

    """

    def __init__(self,
                 output_label='infer_target',
                 reset_label=None,
                 threshold=.98):

        self._output_label = output_label
        self._reset_label = reset_label
        self._threshold = np.log(threshold)
        self._classes = None

    def update(self):

        if not self.i_events.ready():
            return

        if self._reset_label is not None:
            # todo: allow multiple reset_label
            reset_matches = match_events(self.i_events, self._reset_label)
            if reset_matches is not None:
                # reset prior and classes
                self._classes = None
        predict_matches = match_events(self.i_events, 'predict_proba')
        predict_data = predict_matches.data.apply(json.loads).values
        if predict_matches is not None:
            if self._classes is None:
                # infer number of class from prediction result length
                self._classes = predict_data[0]['classes']
                self._n_classes = len(self._classes)
                # initialize probs
                self._logprobs = np.log(np.zeros(self._n_classes) + 1 / self._n_classes)
            # todo: what if multiple matches in same chunk?
            for data in predict_data:
                prob = data['result']
                # update probs
                self._logprobs += np.log(prob)
                # normalize probs
                self._logprobs = np.log(np.exp(self._logprobs) / np.sum(np.exp(self._logprobs)))

            print(self._logprobs)
            if any(self._logprobs >= self._threshold):
                idxmax = np.argmax(self._logprobs)
                # todo: decide on a serialize policy for events'data
                self.o_events.data = pd.DataFrame(index=[self.i_events.data.index[-1]],
                                                  columns=['label', 'data'],
                                                  data=[[self._output_label,
                                                         json.dumps({'result': self._classes[idxmax]})]])
                # reset prior and classes
                self._classes = None
