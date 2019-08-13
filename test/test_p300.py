import numpy as np
import pandas as pd
import pytest
from timeflux.core.exceptions import WorkerInterrupt

from timeflux_bci.nodes.p300 import NaiveBayesInference

node = NaiveBayesInference()
node.logger.level = 10


def test_config_ko():
    node.clear()
    # 33, 35, 19 are duplicated
    data_config_ko = {'symbols': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890',
                      'columns': 6,
                      'groups': [['4', '10', '14', '18', '27', '30'],
                                 ['2', '8', '13', '16', '25', '34'],
                                 ['0', '5', '10', '11', '14', '15'],
                                 ['1', '12', '20', '22', '35', '35'],
                                 ['8', '9', '23', '25', '27', '31'],
                                 ['1', '2', '3', '5', '12', '21'],
                                 ['23', '24', '28', '29', '31', '34'],
                                 ['4', '17', '21', '26', '28', '32'],
                                 ['3', '7', '19', '19', '20', '24'],
                                 ['6', '7', '9', '16', '30', '32'],
                                 ['6', '11', '13', '15', '17', '18'],
                                 ['0', '22', '26', '29', '33', '33']],
                      'repetitions': 3}

    event = pd.DataFrame(
        [
            ['session_begins', data_config_ko],
        ], [
            pd.Timestamp('2018-01-01 00:00:00.00'),
        ], columns=['label', 'data'])
    node.i_events.data = event
    with pytest.raises(WorkerInterrupt):
        node.update()


def test_config_ok():
    data_config_ok = {'symbols': 'ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890',
                      'columns': 6,
                      'groups': [['4', '10', '14', '18', '27', '30'],
                                 ['2', '8', '13', '16', '25', '34'],
                                 ['0', '5', '10', '11', '14', '15'],
                                 ['1', '12', '20', '22', '35', '19'],
                                 ['8', '9', '23', '25', '27', '31'],
                                 ['1', '2', '3', '5', '12', '21'],
                                 ['23', '24', '28', '29', '31', '34'],
                                 ['4', '17', '21', '26', '28', '32'],
                                 ['3', '7', '19', '33', '20', '24'],
                                 ['6', '7', '9', '16', '30', '32'],
                                 ['6', '11', '13', '15', '17', '18'],
                                 ['0', '22', '26', '29', '33', '35']],
                      'repetitions': 3}

    event = pd.DataFrame(
        [
            ['session_begins', data_config_ok],
        ], [
            pd.Timestamp('2018-01-01 00:00:00.00'),
        ], columns=['label', 'data'])
    node.i_events.data = event
    node.update()


def test_inference():
    # let us say the target letter was 'A'
    groups_idx = np.arange(11)

    list_proba = {}
    gs_nb_round = np.arange(1, 9)
    gs_p = np.arange(.5, 1, .05)

    for p in gs_p:
        list_proba[p] = []
        meta = {'epochs_context': list(),
                'epochs_onset': list()}
        data = pd.DataFrame(columns=[False, True])
        for k_round in gs_nb_round:
            np.random.shuffle(groups_idx)
            epochs_context_round = [{"group": group, "includes_target": None}
                                    for group in groups_idx]
            epochs_onset_round = [pd.Timestamp(f'2018-01-01 00:00:00.{isi}')
                                  for isi in range(k_round * 11, (k_round + 1) * 11)]

            data_round = pd.DataFrame(np.stack([[p, 1 - p]] * 11), columns=[False, True])
            has_A_in_group = ['0' in node._groups[group_idx] for group_idx in groups_idx]
            data_round.loc[data_round.index[has_A_in_group], :] = [1 - p, p]

            meta['epochs_context'] += epochs_context_round
            meta['epochs_onset'] += epochs_onset_round

            data = data.append(data_round, sort=True)
            node.clear()
            node.i.data = data
            node.i.meta = meta
            node.update()
            list_proba[p].append(node.o_events.data.data.values[0])
            node._reset_pred()

    posterior_data = np.stack([[r['precision'] for r in res] for (p, res) in list_proba.items()], axis=1)
    expected_posterior = np.array([[0.02777778, 0.04518604, 0.07161804, 0.11055757, 0.16586074,
                                    0.24107143, 0.33862434, 0.4596744, 0.60548173, 0.78085155],
                                   [0.02777778, 0.07098568, 0.15613622, 0.28740716, 0.44603888,
                                    0.60548173, 0.74730889, 0.86109395, 0.94114324, 0.98630096],
                                   [0.02777778, 0.10699687, 0.2800535, 0.50059733, 0.69600946,
                                    0.83800238, 0.92640537, 0.97310979, 0.99317867, 0.99927146],
                                   [0.02777778, 0.15397703, 0.42097638, 0.6800334, 0.85082475,
                                    0.94114324, 0.98076809, 0.99516964, 0.99923839, 0.99996163],
                                   [0.02777778, 0.2110769, 0.5548665, 0.80717117, 0.93160868,
                                    0.97975575, 0.99513617, 0.99914487, 0.99991533, 0.99999798],
                                   [0.02777778, 0.27587937, 0.66916032, 0.88872863, 0.96976288,
                                    0.99317867, 0.99878049, 0.99984901, 0.99999059, 0.99999989],
                                   [0.02777778, 0.34505786, 0.76058474, 0.93761109, 0.9868612,
                                    0.99771794, 0.9996949, 0.99997335, 0.99999895, 0.99999999],
                                   [0.02777778, 0.41525228, 0.83043409, 0.96562963, 0.99433519,
                                    0.99923839, 0.99992371, 0.9999953, 0.99999988, 1.]])
    np.testing.assert_array_almost_equal(posterior_data, expected_posterior, 5)


def plot_gs_result(posterior_data, gs_p, gs_nb_round):
    gs_result = pd.DataFrame(posterior_data,
                             columns=[np.round(p, 2) for p in gs_p],
                             index=gs_nb_round)
    gs_result.columns.name = 'p(T)'
    import matplotlib.pyplot as plt
    gs_result.plot(figsize=(10, 7))
    plt.xlabel('Nb of flash per symbol')
    plt.ylabel('Precision (ie. posterior probability)')
    plt.suptitle('Simulated Naive-Bayes for a grid with 36 symbols and 11 flashing groups')
    plt.autoscale()
