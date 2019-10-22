import json

import numpy as np
import pandas as pd
from timeflux.core.exceptions import WorkerInterrupt
from timeflux.core.node import Node


class NaiveBayesInference(Node):
    r""" Target prediction using a Naive Bayesian Inference

    This node should be plugged after a node 'PredictProba'. In th case of p300,
    there are two class: False (non-target) and True (target). The expected data
    is a DataFrame with columns [False, True], where each row corresponds to a flash
    and the values give the flash scores.

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

    A Naive Bayes classifier is used to determine the probability of each symbol
    given the flash scores and the previous decisions.

    Let us remind Bayes' theorem:

    .. math:: P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)}

    If we assume that the individual flashes are conditionally independent given
    the current attended character, the posterior probability is:

    .. math::

        P(xt \mid yt,xt−1,…,x0) &= \frac{P(yt,xt−1,…,x0 \mid xt) \, P(xt)}{P(yt,xt−1,…,x0)} \\
                                &= \frac{1}{Z} \prod_{i} f(yit∣xt)

    We estimate the constant Z at each iteration (ie. each flash, ie. each row),
    as the sum of all posteriors (so that the vector of probability has a sum of 1).

    Once all rows have been considered, the node decides that the target is the symbol
    with greater posterior probability and returns an event with  data ('target', 'precision' )

    Args:
        event_label (string): The column to match for event_trigger.
        event_data (string, None): The column where meta-data is stored.
        event_label_config (string): The marker name on which the node sets its config (ie. nb of repetitions, targets ids...)
        event_label_prediction (string): The marker name of the output events stream when a prediction is made.

    """

    def __init__(self,
                 event_label='label', event_data='data',
                 event_label_config='session_begins',
                 event_label_prediction='model_predicts'):

        self._symbols = None
        self._event_label = event_label
        self._event_data = event_data
        self._event_label_config = event_label_config
        self._event_label_prediction = event_label_prediction
        self._node_ready = False

    def update(self):

        if self.i_events.ready():
            matches = self.i_events.data[self.i_events.data[self._event_label] == self._event_label_config]
            if not matches.empty:
                self._set_config(matches)
                self._reset_pred()

        if not self.i.ready() or not self._node_ready:
            return

        self.o.data = self.i.data
        # deserialize the context
        epochs_context = self.i.meta['epochs_context']
        epochs_context = [json.loads(context) if isinstance(context, str) else context for context in
                          epochs_context]

        self.o.data['onset'] = self.i.meta['epochs_onset']
        self.o.data['group'] = [context['group'] for context in epochs_context]
        self.o.data['intermediate_target'] = np.NaN
        self.o.data['intermediate_precision'] = np.NaN

        # Naive-Bayesian inference
        for k, epoch_data in self.o.data.iterrows():
            flashed_symbols = np.array(self._groups[epoch_data.group]).astype(int)
            self._proba *= epoch_data[False]
            self._proba.iloc[:, flashed_symbols] *= (epoch_data[True] / epoch_data[False])
            _scale = self._proba.sum(axis=1).values[0]
            self._proba /= _scale
            max_proba = self._proba[self._proba.idxmax(axis=1)]
            self._prediction['target'] = max_proba.columns[0]
            self._prediction['precision'] = max_proba.values[0, 0]
            # todo: eventually, append and return the intermediate prediction
            self.o.data.iloc[k, -2] = self._prediction['target']
            self.o.data.iloc[k, -1] = self._prediction['precision']

        self.o_events.data = pd.DataFrame(index=[self.o.data['onset'].values[-1]],
                                          columns=['label', 'data'],
                                          data=[[self._event_label_prediction,
                                                 self._prediction]])
        self._reset_pred()

    def _set_config(self, match):
        """ Initialize the speller configuration. """

        data = match[self._event_data].values[0]
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except ValueError:
                raise WorkerInterrupt(f'Could not deserialize data from event'
                                      f' {self._event_label_config}.')
        try:
            self._symbols = data['symbols']
            self._repetitions = data['repetitions']
            self._groups = data['groups']
        except KeyError as k:
            raise WorkerInterrupt(f'Could not configure {k} from event {self._event_label_config} ')

        self._nb_groups = len(self._groups)
        self._nb_symbols = len(self._symbols)
        # check that each symbol remains to the same number of group
        _unique_occurrence_symbol_per_group = np.unique(
            [np.sum([str(symbols_idx) in group for group in self._groups])
             for symbols_idx in range(self._nb_symbols)])
        if len(_unique_occurrence_symbol_per_group) != 1:
            # todo: raise WorkerInterrupt instead of warning
            self.logger.warning(f'The number of occurrence of each symbol in each group should be fixed. '
                                f'Found {_unique_occurrence_symbol_per_group}. ')

        self._nb_occurrence_symbol_per_round = _unique_occurrence_symbol_per_group[0]
        self.logger.info(
            f'Set configuration for p300 interface with a grid of {self._nb_symbols} symbols flashing '
            f'{self._nb_occurrence_symbol_per_round} times per round and {self._repetitions} rounds per block.  ')
        self._node_ready = True

    def _reset_pred(self):
        """ Reset thee probability of each symbol to be target"""
        self._proba = pd.DataFrame(columns=list(self._symbols),
                                   data=np.zeros([1, self._nb_symbols]))
        self._proba += 1 / self._nb_symbols
        self._prediction = {'target': None, 'precision': None}
