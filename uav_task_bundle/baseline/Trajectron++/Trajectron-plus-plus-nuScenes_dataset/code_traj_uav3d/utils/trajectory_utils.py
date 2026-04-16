import numpy as np
from scipy.integrate import cumtrapz


def integrate(f, dx, F0=0.):
    N = f.shape[0]
    return F0 + np.hstack((np.zeros((N, 1)), cumtrapz(f, axis=1, dx=dx)))


def integrate_trajectory(v, x0, dt):
    xd_ = integrate(v[..., 0], dx=dt, F0=x0[0])
    yd_ = integrate(v[..., 1], dx=dt, F0=x0[1])
    zd_ = integrate(v[..., 2], dx=dt, F0=x0[2])
    integrated = np.stack([xd_, yd_, zd_], axis=2)
    return integrated


def prediction_output_to_trajectories(prediction_output_dict,
                                      dt,
                                      max_h,
                                      ph,
                                      map=None,
                                      gmm_agg='mean',
                                      prune_ph_to_future=False):

    prediction_timesteps = prediction_output_dict.keys()

    output_dict = dict()
    histories_dict = dict()
    futures_dict = dict()

    for t in prediction_timesteps:
        histories_dict[t] = dict()
        output_dict[t] = dict()
        futures_dict[t] = dict()
        prediction_nodes = prediction_output_dict[t].keys()
        for node in prediction_nodes:
            predictions_output = prediction_output_dict[t][node]
            position_state = {'position': ['x', 'y', 'z']}
            history = node.get(np.array([t - max_h, t]), position_state)  # History includes current pos
            history = history[~np.isnan(history.sum(axis=1))]

            future = node.get(np.array([t + 1, t + ph]), position_state)
            future = future[~np.isnan(future.sum(axis=1))]

            predictions_output = getattr(predictions_output, gmm_agg)(axis=1)

            if prune_ph_to_future:
                predictions_output = predictions_output[:, :future.shape[0]]
                if predictions_output.shape[1] == 0:
                    continue

            if map is None:
                histories_dict[t][node] = history
                output_dict[t][node] = predictions_output
                futures_dict[t][node] = future
            else:
                histories_dict[t][node] = map.to_map_points(history[..., :2])
                output_dict[t][node] = map.to_map_points(predictions_output[..., :2])
                futures_dict[t][node] = map.to_map_points(future[..., :2])

    return output_dict, histories_dict, futures_dict
