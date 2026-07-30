import copy
from warnings import warn

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import correlate, periodogram, find_peaks, savgol_filter

from .utils import lowpass_butter, pd_interp


def resample_imu(sessiondata, sfreq=400.0):
    """
    Resample all devices and sensors to new sample frequency.

    Resamples all devices and sensors to new sample frequency. Sample intervals are not fixed with ngimu so resampling
    before further analysis is recommended. Translated from xio-Technologies.

    Parameters
    ----------
    sessiondata : dict
        original session data structure to be resampled
    sfreq : float
        new intended sample frequency

    Returns
    -------
    sessiondata : dict
        resampled session data structure

    References
    ----------
    https://github.com/xioTechnologies/NGIMU-MATLAB-Import-Logged-Data-Example

    """
    if sfreq <= 0 or not np.isfinite(sfreq):
        raise ValueError("sfreq must be a finite positive number")

    frames = [data for device, data in sessiondata.items() if device not in ("quaternion", "matrix")]
    if not frames:
        raise ValueError("sessiondata does not contain any resampleable IMU data")

    start_time = max(data["time"].min() for data in frames)
    end_time = min(data["time"].max() for data in frames)
    if not np.isfinite(start_time) or not np.isfinite(end_time) or end_time <= start_time:
        raise ValueError("IMU devices do not have an overlapping time range")

    new_time = np.arange(start_time, end_time, 1 / sfreq)
    if new_time.size == 0:
        raise ValueError("The overlapping IMU time range is shorter than one output sample")

    for device in sessiondata:
        if device == "quaternion":
            sessiondata[device] = pd_interp(sessiondata[device], "time", new_time)
            sessiondata[device] *= 1 / np.linalg.norm(sessiondata[device], axis=0)
        elif device == "matrix":
            warn("Rotation matrix cannot be resampled. This dataframe has been removed")
        else:
            sessiondata[device] = pd_interp(sessiondata[device], "time", new_time)
        if device != "matrix":
            sessiondata[device]["time"] = new_time - start_time
    return sessiondata


def fit_imu_sync(anchors):
    """
    Fit an affine clock mapping from synchronization anchors.

    The returned model maps a source-device time onto the reference-device
    clock using ``reference_time = scale * source_time + offset``. At least two
    anchors are required so clock drift can be estimated as well as a fixed
    offset.

    Parameters
    ----------
    anchors : array-like or list of dict
        Pairs of source and reference times. Dictionary anchors must contain
        ``source_time`` and ``reference_time``.

    Returns
    -------
    model : dict
        Affine model containing scale, offset, residual_rms and the anchors
        retained by robust outlier rejection.
    """
    pairs = []
    for anchor in anchors:
        if isinstance(anchor, dict):
            pairs.append((anchor["source_time"], anchor["reference_time"]))
        else:
            pairs.append((anchor[0], anchor[1]))
    pairs = np.asarray(pairs, dtype=float)

    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) < 2:
        raise ValueError("At least two source/reference synchronization anchors are required")
    if not np.all(np.isfinite(pairs)):
        raise ValueError("Synchronization anchors must be finite")
    if np.ptp(pairs[:, 0]) <= 0:
        raise ValueError("Synchronization anchors must span more than one source time")

    used = pairs
    for _ in range(3):
        scale, offset = _least_squares_clock_fit(used)
        residuals = used[:, 1] - (scale * used[:, 0] + offset)
        median_residual = np.median(residuals)
        deviation = np.abs(residuals - median_residual)
        mad = np.median(deviation)
        threshold = max(0.02, 3 * 1.4826 * mad)
        keep = deviation <= threshold
        if keep.all() or keep.sum() < 2:
            break
        used = used[keep]

    scale, offset = _least_squares_clock_fit(used)
    residuals = used[:, 1] - (scale * used[:, 0] + offset)
    residual_rms = float(np.sqrt(np.mean(residuals ** 2)))
    return {
        "scale": float(scale),
        "offset": float(offset),
        "residual_rms": residual_rms,
        "anchors": [
            {"source_time": float(source), "reference_time": float(reference)}
            for source, reference in used
        ],
    }


def synchronize_imu(sessiondata, reference="frame", devices=None, signal_column="gyroscope_x",
                    window=8.0, max_lag=None, sfreq=None, models=None, inplace=False,
                    return_models=False):
    """
    Synchronize IMU devices with affine offset and clock-drift correction.

    By default, synchronization anchors are estimated in several signal
    windows using cross-correlation. For reproducible or manually reviewed
    synchronization, pass a mapping of device names to models from
    :func:`fit_imu_sync`.

    Parameters
    ----------
    sessiondata : dict
        Mapping of device names to DataFrames containing ``time``.
    reference : str
        Device whose clock defines the synchronized timeline.
    devices : iterable of str, optional
        Devices to synchronize. Defaults to every device except the reference.
    signal_column : str
        Shared signal used for automatic anchor estimation.
    window : float
        Correlation-window duration in seconds.
    max_lag : float, optional
        Maximum local deviation from the whole-recording lag, in seconds.
    sfreq : float, optional
        Output sample frequency. Defaults to the lowest input frequency.
    models : dict, optional
        Explicit affine models keyed by device. Each model needs ``scale`` and
        ``offset``; a sequence of anchors may be supplied instead.
    inplace : bool
        Modify the supplied data when True.
    return_models : bool
        Also return the fitted/applied models when True.

    Returns
    -------
    sessiondata : dict
        Synchronized and resampled session data.
    models : dict, optional
        Returned with sessiondata when ``return_models`` is True.
    """
    if not inplace:
        sessiondata = copy.deepcopy(sessiondata)
    if reference not in sessiondata:
        raise KeyError("Reference IMU {!r} is not present".format(reference))

    if devices is None:
        devices = [
            device for device, data in sessiondata.items()
            if device != reference and "time" in data and signal_column in data
        ]
    else:
        devices = list(devices)
    if reference in devices:
        raise ValueError("The reference IMU cannot also be a source device")

    applied_models = {}
    provided_models = {} if models is None else models
    for device in devices:
        if device not in sessiondata:
            raise KeyError("Source IMU {!r} is not present".format(device))
        model = provided_models.get(device)
        if model is None:
            model = _estimate_imu_sync(
                sessiondata[device],
                sessiondata[reference],
                signal_column=signal_column,
                window=window,
                max_lag=max_lag,
            )
        elif not isinstance(model, dict) or "scale" not in model or "offset" not in model:
            model = fit_imu_sync(model)

        scale = float(model["scale"])
        offset = float(model["offset"])
        if not np.isfinite(scale) or scale <= 0 or not np.isfinite(offset):
            raise ValueError("Synchronization model for {!r} is invalid".format(device))
        sessiondata[device]["time"] = scale * sessiondata[device]["time"] + offset
        applied_models[device] = dict(model, scale=scale, offset=offset)

    if sfreq is None:
        sfreq = min(
            _sample_frequency(data) for data in sessiondata.values()
            if "time" in data
        )
    sessiondata = resample_imu(sessiondata, sfreq=sfreq)
    if return_models:
        return sessiondata, applied_models
    return sessiondata


def imu_synch(sessiondata, right_wheel=True, inplace=False, **kwargs):
    """
    Backwards-compatible wheel/frame synchronization entry point.

    This replaces the historical fixed-lag implementation with affine
    synchronization. Use :func:`synchronize_imu` to align multiple devices in
    one call.
    """
    device = "right" if right_wheel else "left"
    return synchronize_imu(sessiondata, devices=[device], inplace=inplace, **kwargs)


def _estimate_imu_sync(source, reference, signal_column, window, max_lag):
    if window <= 0:
        raise ValueError("window must be positive")
    source_time, source_signal = _sync_arrays(source, signal_column)
    reference_time, reference_signal = _sync_arrays(reference, signal_column)
    source_frequency = _sample_frequency(source)
    reference_frequency = _sample_frequency(reference)
    frequency = min(source_frequency, reference_frequency)
    step = 1 / frequency

    source_elapsed = source_time - source_time[0]
    reference_elapsed = reference_time - reference_time[0]
    source_grid = np.arange(0, source_elapsed[-1], step)
    reference_grid = np.arange(0, reference_elapsed[-1], step)
    if source_grid.size < 8 or reference_grid.size < 8:
        raise ValueError("IMU recordings are too short to estimate synchronization")

    source_values = np.interp(source_grid, source_elapsed, source_signal)
    reference_values = np.interp(reference_grid, reference_elapsed, reference_signal)
    source_values = _standardize_sync_signal(source_values)
    reference_values = _standardize_sync_signal(reference_values)

    whole_correlation = correlate(reference_values, source_values, mode="full", method="fft")
    whole_lags = _full_correlation_lags(len(reference_values), len(source_values))
    global_lag = whole_lags[np.argmax(np.abs(whole_correlation))]

    window_samples = int(round(window * frequency))
    window_samples = min(window_samples, len(source_values) // 2)
    window_samples = max(window_samples, 8)
    if window_samples >= len(source_values):
        raise ValueError("Correlation window must be shorter than the source recording")

    if max_lag is None:
        max_lag = max(1.0, min(source_elapsed[-1], reference_elapsed[-1]) * 0.05)
    if max_lag <= 0:
        raise ValueError("max_lag must be positive")
    lag_tolerance = int(round(max_lag * frequency))

    starts = np.linspace(0, len(source_values) - window_samples, num=7, dtype=int)
    anchors = []
    for start in np.unique(starts):
        source_window = source_values[start:start + window_samples]
        if np.std(source_window) < 1e-8:
            continue
        source_window = _standardize_sync_signal(source_window)
        window_lags = _full_correlation_lags(len(reference_values), len(source_window))
        allowed = (
            (window_lags >= global_lag + start - lag_tolerance)
            & (window_lags <= global_lag + start + lag_tolerance)
        )
        centre = (window_samples - 1) / 2
        allowed &= (window_lags >= 0) & (window_lags + window_samples <= len(reference_grid))
        if not np.any(allowed):
            continue
        candidate_lags = window_lags[allowed]
        scores = []
        for lag in candidate_lags:
            reference_window = reference_values[lag:lag + window_samples]
            reference_window = _standardize_sync_signal(reference_window)
            scores.append(abs(np.mean(reference_window * source_window)))
        best_lag = candidate_lags[np.argmax(scores)]
        source_centre = (start + centre) * step
        reference_centre = (best_lag + centre) * step
        anchors.append({
            "source_time": float(source_time[0] + source_centre),
            "reference_time": float(reference_time[0] + reference_centre),
        })

    if len(anchors) < 2:
        raise ValueError("Could not find enough reliable synchronization anchors")
    model = fit_imu_sync(anchors)
    model["method"] = "windowed-cross-correlation-affine-v1"
    return model


def _least_squares_clock_fit(pairs):
    source = pairs[:, 0]
    reference = pairs[:, 1]
    source_mean = np.mean(source)
    reference_mean = np.mean(reference)
    variance = np.sum((source - source_mean) ** 2)
    if variance <= 0:
        raise ValueError("Synchronization anchors must span more than one source time")
    scale = np.sum((source - source_mean) * (reference - reference_mean)) / variance
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Synchronization anchors imply an invalid clock scale")
    return scale, reference_mean - scale * source_mean


def _sync_arrays(data, signal_column):
    if "time" not in data or signal_column not in data:
        raise KeyError("IMU data must contain 'time' and {!r}".format(signal_column))
    time = np.asarray(data["time"], dtype=float)
    values = np.asarray(data[signal_column], dtype=float)
    valid = np.isfinite(time) & np.isfinite(values)
    time = time[valid]
    values = values[valid]
    if len(time) < 3 or np.any(np.diff(time) <= 0):
        raise ValueError("IMU time values must contain at least three strictly increasing samples")
    return time, values


def _sample_frequency(data):
    if "time" not in data:
        raise KeyError("IMU data must contain a 'time' column")
    time = np.asarray(data["time"], dtype=float)
    differences = np.diff(time[np.isfinite(time)])
    differences = differences[differences > 0]
    if differences.size == 0:
        raise ValueError("IMU time values must contain increasing samples")
    frequency = 1 / np.median(differences)
    if not np.isfinite(frequency) or frequency <= 0:
        raise ValueError("Could not determine a valid IMU sample frequency")
    return float(frequency)


def _standardize_sync_signal(values):
    values = np.asarray(values, dtype=float)
    centred = values - np.mean(values)
    scale = np.std(centred)
    if not np.isfinite(scale) or scale < 1e-12:
        raise ValueError("Synchronization signal does not contain enough variation")
    return centred / scale


def _full_correlation_lags(first_length, second_length):
    return np.arange(-second_length + 1, first_length)


def process_imu(sessiondata, camber=18, wsize=0.32, wbase=0.80, n_sensors=3, sensor_type='ngimu', inplace=False):
    """
    Calculate wheelchair kinematic variables based on NGIMU data

    Parameters
    ----------
    sessiondata : dict
        original sessiondata structure
    camber : float
        camber angle in degrees
    wsize : float
        radius of the wheels
    wbase : float
        width of wheelbase
    n_sensors: float
        number of sensors used: 2: right wheel and frame, 3: right, left wheel and frame
    sensor_type: string
        type of sensor, 'ngimu' or 'ximu3' is for xio-technologies, 'move' is for movesense
    inplace : bool
        performs operation inplace


    Returns
    -------
    sessiondata : dict
        sessiondata structure with processed data

    """
    if not inplace:
        sessiondata = copy.deepcopy(sessiondata)
    frame = sessiondata["frame"]
    right = sessiondata["right"]

    sfreq = 1 / frame["time"].diff().mean()
    frame["rot_vel"] = lowpass_butter(frame["gyroscope_z"], sfreq=sfreq, cutoff=6)
    frame['rot_vel'] = savgol_filter(frame['rot_vel'], window_length=100, polyorder=3)
    right['gyroscope_y'] = lowpass_butter(right['gyroscope_y'], sfreq=sfreq, cutoff=10)

    # Wheelchair camber correction
    deg2rad = np.pi / 180
    right["gyro_cor"] = right["gyroscope_y"] + np.tan(camber * deg2rad) * (
        frame["rot_vel"] * np.cos(camber * deg2rad))
    if n_sensors == 3:
        left = sessiondata["left"]
        left['gyroscope_y'] = lowpass_butter(left['gyroscope_y'], sfreq=sfreq, cutoff=10)
        left["gyro_cor"] = left["gyroscope_y"] - np.tan(camber * deg2rad) * (
            frame["rot_vel"] * np.cos(camber * deg2rad))
        frame["gyro_cor"] = (right["gyro_cor"] + left["gyro_cor"]) / 2
    else:
        frame["gyro_cor"] = right["gyro_cor"]

    # Calculation of rotations, rotational velocity and rotational acceleration
    frame["rot"] = cumulative_trapezoid(abs(frame["rot_vel"]) / sfreq, initial=0.0)
    frame["rot_acc"] = np.gradient(frame["rot_vel"]) * sfreq

    # Calculation of velocity, acceleration and distance
    right["vel"] = right["gyro_cor"] * wsize * deg2rad  # angular velocity to linear velocity
    right["dist"] = cumulative_trapezoid(right["vel"] / sfreq, initial=0.0)  # integral of velocity gives distance

    if n_sensors == 3:
        left["vel"] = left["gyro_cor"] * wsize * deg2rad
        left["dist"] = cumulative_trapezoid(left["vel"] / sfreq, initial=0.0)
        frame["vel_wheel"] = (right["vel"] + left["vel"]) / 2  # mean velocity both sides
        frame["dist_wheel"] = (right["dist"] + left["dist"]) / 2  # mean distance
    else:
        frame["vel_wheel"] = right["vel"]
        frame["dist_wheel"] = right["dist"]

    frame["vel_wheel"] = lowpass_butter(frame["vel_wheel"], sfreq=sfreq, cutoff=10)
    frame["acc_wheel"] = np.gradient(frame["vel_wheel"]) * sfreq  # mean acceleration from velocity
    frame['acc_wheel'] = lowpass_butter(frame['acc_wheel'], sfreq=sfreq, cutoff=10)

    if sensor_type == 'ngimu' or sensor_type == 'ximu3':  # Acceleration for NGIMU/XIMU3 is in g
        frame["accelerometer_x"] = frame["accelerometer_x"] * 9.81
    frame['acc'] = lowpass_butter(frame['accelerometer_x'], sfreq=sfreq, cutoff=10)

    """Perform skid correction from Rienk vd Slikke, please refer and reference to: Van der Slikke, R. M. A., et. al.
    Wheel skid correction is a prerequisite to reliably measure wheelchair sports kinematics based on inertial sensors.
    Procedia Engineering, 112, 207-212."""
    frame["vel_right"] = right["vel"]  # Calculate frame centre distance
    frame["vel_right"] -= np.tan(np.deg2rad(frame["rot_vel"] / sfreq)) * wbase / 2 * sfreq

    if n_sensors == 3:
        frame["vel_left"] = left["vel"]
        frame["vel_left"] += np.tan(np.deg2rad(frame["rot_vel"] / sfreq)) * wbase / 2 * sfreq

        r_ratio0 = np.abs(right["vel"]) / (np.abs(right["vel"]) + np.abs(left["vel"]))  # Ratio left and right
        l_ratio0 = np.abs(left["vel"]) / (np.abs(right["vel"]) + np.abs(left["vel"]))
        r_ratio1 = np.abs(np.gradient(left["vel"])) / (np.abs(np.gradient(right["vel"]))
                                                       + np.abs(np.gradient(left["vel"])))
        l_ratio1 = np.abs(np.gradient(right["vel"])) / (np.abs(np.gradient(right["vel"]))
                                                        + np.abs(np.gradient(left["vel"])))

        comb_ratio = (r_ratio0 * r_ratio1) / ((r_ratio0 * r_ratio1) + (l_ratio0 * l_ratio1))  # Combine speed ratios
        comb_ratio.fillna(value=0., inplace=True)
        comb_ratio = lowpass_butter(comb_ratio, sfreq=sfreq, cutoff=20)  # Filter the signal
        comb_ratio = np.clip(comb_ratio, 0, 1)  # clamp Combine ratio values, not in df
        frame["skid_vel"] = (frame["vel_right"] * comb_ratio) + (frame["vel_left"] * (1 - comb_ratio))
        frame["vel"] = (frame["vel_right"] + frame["vel_left"]) / 2
        frame['dist'] = cumulative_trapezoid(frame["skid_vel"], initial=0.0) / sfreq
    else:
        frame["vel"] = frame["vel_right"]
        frame["dist"] = cumulative_trapezoid(frame["vel"], initial=0.0) / sfreq  # Combined distance

    # distance in the x and y direction
    frame["dist_y"] = cumulative_trapezoid(
        frame['vel'] / sfreq * np.sin(np.deg2rad(cumulative_trapezoid(frame["rot_vel"] / sfreq, initial=0.0))),
        initial=0.0)
    frame["dist_x"] = cumulative_trapezoid(
        frame['vel'] / sfreq * np.cos(np.deg2rad(cumulative_trapezoid(frame["rot_vel"] / sfreq, initial=0.0))),
        initial=0.0)

    return sessiondata


def process_imu_left(sessiondata, camber=18, wsize=0.32, wbase=0.80,
                     sensor_type='ngimu', inplace=False):
    """
    Calculate wheelchair kinematic variables based on NGIMU data

    Parameters
    ----------
    sessiondata : dict
        original sessiondata structure
    camber : float
        camber angle in degrees
    wsize : float
        radius of the wheels
    wbase : float
        width of wheelbase
    sensor_type: string
        type of sensor, 'ngimu' or 'ximu3' is xio-technologies, 'move' is movesense
    inplace : bool
        performs operation inplace


    Returns
    -------
    sessiondata : dict
        sessiondata structure with processed data

    """
    if not inplace:
        sessiondata = copy.deepcopy(sessiondata)
    frame = sessiondata["frame"]
    left = sessiondata["left"]
    sfreq = int(1 / frame["time"].diff().mean())

    # Calculation of rotations, rotational velocity and acceleration
    frame["rot_vel"] = lowpass_butter(frame["gyroscope_z"],
                                      sfreq=sfreq, cutoff=10)
    frame['rot_vel'] = savgol_filter(frame['rot_vel'], window_length=100, polyorder=3)
    frame["rot"] = cumulative_trapezoid(abs(frame["rot_vel"]) / sfreq, initial=0.0)
    frame["rot_acc"] = np.gradient(frame["rot_vel"]) * sfreq

    # Wheelchair camber correction
    deg2rad = np.pi / 180
    left['gyroscope_y'] = lowpass_butter(left['gyroscope_y'], sfreq=sfreq, cutoff=10)
    left["gyro_cor"] = left["gyroscope_y"] - np.tan(camber * deg2rad) * (
        frame["rot_vel"] * np.cos(camber * deg2rad))
    frame["gyro_cor"] = left["gyro_cor"]

    left["vel"] = left["gyro_cor"] * wsize * deg2rad
    left["dist"] = cumulative_trapezoid(left["vel"] / sfreq, initial=0.0)
    frame["vel_wheel"] = left["vel"]
    frame["vel_wheel"] = lowpass_butter(frame["vel_wheel"], sfreq=sfreq, cutoff=10)
    frame["dist_wheel"] = cumulative_trapezoid(frame["vel_wheel"] / sfreq, initial=0.0)

    frame["acc_wheel"] = np.gradient(frame["vel_wheel"]) * sfreq
    frame['acc_wheel'] = lowpass_butter(frame['acc_wheel'],
                                        sfreq=sfreq, cutoff=10)

    if sensor_type == 'ngimu' or sensor_type == 'ximu3':  # Acceleration for NGIMU/XIMU3 is in g
        frame["accelerometer_x"] = frame["accelerometer_x"] * 9.81
    frame['acc'] = lowpass_butter(frame['accelerometer_x'],
                                  sfreq=sfreq, cutoff=10)

    frame["vel_left"] = left["vel"]
    frame["vel_left"] += np.tan(np.deg2rad(frame["rot_vel"] / sfreq)) * wbase / 2 * sfreq
    frame["vel"] = frame["vel_left"]
    frame["dist"] = cumulative_trapezoid(frame["vel"], initial=0.0) / sfreq

    # distance in the x and y direction
    frame["dist_y"] = cumulative_trapezoid(
        frame['vel'] / sfreq * np.sin(np.deg2rad(cumulative_trapezoid(frame["rot_vel"] / sfreq, initial=0.0))),
        initial=0.0)
    frame["dist_x"] = cumulative_trapezoid(
        frame['vel'] / sfreq * np.cos(np.deg2rad(cumulative_trapezoid(frame["rot_vel"] / sfreq, initial=0.0))),
        initial=0.0)

    return sessiondata


def change_imu_orientation(sessiondata, inplace=False):
    """
    Changes IMU orientation from in-wheel to on-wheel

    Parameters
    ----------
    sessiondata : dict
        original sessiondata structure
    inplace : bool
        perform operation inplace

    Returns
    -------
    sessiondata : dict
        sessiondata with reoriented gyroscope data

    """
    if not inplace:
        sessiondata = copy.deepcopy(sessiondata)

    order = {"gyroscope_x": "gyroscope_z", "gyroscope_z": "gyroscope_y", "gyroscope_y": "gyroscope_x"}
    sessiondata["left"]["sensors"].rename(columns=order, inplace=True)
    sessiondata["right"]["sensors"].rename(columns=order, inplace=True)
    sessiondata["right"]["sensors"]["gyroscope_y"] *= -1
    return sessiondata


def push_imu(acceleration, sfreq=400.0):
    """
    Push detection based on velocity signal of IMU on a wheelchair.

    Parameters
    ----------
    acceleration : np.array, pd.Series
        acceleration data structure
    sfreq : float
        sampling frequency

    Returns
    -------
        push_idx, acc_filt, n_pushes, cycle_time, push_freq

    References
    ----------
    van der Slikke, R., Berger, M., Bregman, D., & Veeger, D. (2016). Push characteristics in wheelchair court sport
    sprinting. Procedia engineering, 147, 730-734.

    """
    min_freq = 1.2
    f, pxx = periodogram(acceleration - np.mean(acceleration), sfreq)
    min_freq_f = len(f[f < min_freq])
    max_freq_ind_temp = np.argmax(pxx[min_freq_f: min_freq_f * 5])
    max_freq = f[min_freq_f + max_freq_ind_temp]
    max_freq = min(max_freq, 3.0)
    cutoff_freq = 1.5 * max_freq
    acc_filt = lowpass_butter(acceleration, sfreq=sfreq, cutoff=cutoff_freq)
    std_acc = np.std(acc_filt)
    push_idx, peak_char = find_peaks(
        acc_filt, height=std_acc / 2, distance=round(1 / (max_freq * 1.5) * sfreq), prominence=std_acc / 2
    )
    n_pushes = len(push_idx)
    push_freq = n_pushes / (len(acceleration) / sfreq)
    cycle_time = list()

    for n in range(0, len(push_idx) - 1):
        cycle_time.append((push_idx[n + 1] / sfreq) - (push_idx[n] / sfreq))

    return push_idx, acc_filt, n_pushes, cycle_time, push_freq


def movesense_offset(sessiondata, n_sensors=2, right_wheel=True, gyro_offset=False):
    """
    Remove offset MoveSense sensors

    Parameters
    ----------
    sessiondata : dict
        resampled sessiondata structure
    right_wheel: boolean
        if set to True, right wheel is used, if set to False, left wheel is used
    n_sensors: float
        number of sensors used, 2: right wheel and frame,
        3: right, left wheel and frame
    gyro_offset: boolean
        if set to True, an additional gyroscope offset will be used

    Returns
    -------
    sessiondata : dict
        sessiondata with offset removed

    """
    if right_wheel is True:
        offset_indices = (np.abs(sessiondata['frame']['gyroscope_z']) < 5) & (
            np.abs(sessiondata['right']['gyroscope_y']) < 5)
    else:
        offset_indices = (np.abs(sessiondata['frame']['gyroscope_z']) < 5) & (
            np.abs(sessiondata['left']['gyroscope_y']) < 5)

    if sum(offset_indices) > 10:
        offset_frame_x = np.mean(sessiondata['frame']['gyroscope_x'][offset_indices])
        offset_frame_y = np.mean(sessiondata['frame']['gyroscope_y'][offset_indices])
        offset_frame_z = np.mean(sessiondata['frame']['gyroscope_z'][offset_indices])
        sessiondata['frame']['gyroscope_x'] -= offset_frame_x
        sessiondata['frame']['gyroscope_y'] -= offset_frame_y
        sessiondata['frame']['gyroscope_z'] -= offset_frame_z

        if right_wheel is True:
            offset_right_y = np.mean(sessiondata['right']['gyroscope_y'][offset_indices])
            offset_right_z = np.mean(sessiondata['right']['gyroscope_z'][offset_indices])
            offset_right_x = np.mean(sessiondata['right']['gyroscope_x'][offset_indices])
            sessiondata['right']['gyroscope_y'] -= offset_right_y
            sessiondata['right']['gyroscope_z'] -= offset_right_z
            sessiondata['right']['gyroscope_x'] -= offset_right_x

        if n_sensors == 3 or right_wheel is False:
            offset_left_y = np.mean(sessiondata['left']['gyroscope_y'][offset_indices])
            offset_left_z = np.mean(sessiondata['left']['gyroscope_z'][offset_indices])
            offset_left_x = np.mean(sessiondata['left']['gyroscope_x'][offset_indices])
            sessiondata['left']['gyroscope_y'] -= offset_left_y
            sessiondata['left']['gyroscope_z'] -= offset_left_z
            sessiondata['left']['gyroscope_x'] -= offset_left_x
    else:
        print('No offset corrected')
    if gyro_offset is True:
        sessiondata['frame']['gyroscope_z'] = np.sign(
            sessiondata['frame']['gyroscope_z']) * np.sqrt(sessiondata['frame']['gyroscope_x']**2
                                                           + sessiondata['frame']['gyroscope_y']**2
                                                           + sessiondata['frame']['gyroscope_z']**2)

    return sessiondata
