import numpy as np


def compute_metrics(df, desired, band=1.5):
    e=df.tracking_error.to_numpy(); r=np.abs(df.range_to_target.to_numpy()-desired)
    inside=r<=band; first=float(df.time[inside].iloc[0]) if inside.any() else None
    reacq=[]
    lost=None
    for row in df.itertuples():
        if not row.detection_valid and lost is None: lost=row.time
        if row.detection_valid and lost is not None: reacq.append(row.time-lost); lost=None
    return {
        "mean_estimation_error":float(np.nanmean(e)), "estimation_rmse":float(np.sqrt(np.nanmean(e**2))),
        "estimation_error_p95":float(np.nanpercentile(e,95)), "maximum_estimation_error":float(np.nanmax(e)),
        "total_uav_travel_distance":float(df.step_distance.sum()), "mean_standoff_error":float(np.mean(r)),
        "standoff_rmse":float(np.sqrt(np.mean(r**2))), "time_to_first_valid_standoff":first,
        "percent_time_inside_standoff_band":float(100*np.mean(inside)), "detection_availability":float(100*df.detection_valid.mean()),
        "mean_reacquisition_time":float(np.mean(reacq)) if reacq else None,
    }

