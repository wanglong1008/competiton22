import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from os import path
from PIL import Image
from scipy.spatial.transform import Rotation as R


plt.rcParams['axes.formatter.useoffset'] = False
plt.rcParams['figure.figsize'] = [12.0, 6.0]


def mouse_event(event):
    """On Right-click:
        - print lon/lat if in 'position' plot;
        - print sample index and angle if in 'orientation' plot.
       On Shift + Right-click in orientation plot, show data for the corresponding sample index.
       NOTE: Uses global dd variable.
    """
    if event.button == 3 and event.inaxes is not None:
        if event.inaxes.title.get_text() == 'position':
            print("lon: {:.6f}, lat: {:.6f}".format(event.xdata, event.ydata))
        elif event.inaxes.title.get_text() == 'orientation':
            n = int(event.xdata)
            if n >= 0 and n < dd.shape[0]:
                print("n: {:d}, angle: {:.3f}".format(n, event.ydata))
                if event.key == 'shift':
                    print("   sample[{: 4d}]: lon: {:.6f}, lat: {:.6f}, q_wxyz: ({:.6f},{:.6f},{:.6f},{:.6f})"
                          .format(n, dd.lon.iloc[n], dd.lat.iloc[n],
                                  dd.qw.iloc[n], dd.qx.iloc[n], dd.qy.iloc[n], dd.qz.iloc[n]))
                    print("                 image: {}".format(dd.image.iloc[n]))
                    show_frame(dd.image.iloc[n])


def show_frame(imgp):
    Image.open(path.join(DROOT, imgp)).show(title=str(path.basename(imgp)))


def plot_trj(ds, title=""):
    """Plot potision and orientation data for a given trajectory.
       ds - a DataFrame with trajectory data
    """
    fig = plt.figure()
    fig.suptitle(title)
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(122)
    # Position plot
    # ax1.imshow(fmp, extent=[127.369866, 127.370447, 36.383446, 36.383929])
    ax1.plot(ds.lon, ds.lat, label="t11")
    ax1.scatter(ds.lon.head(1), ds.lat.head(1), label="start")
    ax1.set_title("position")
    ax1.grid(True)
    ax1.legend()
    ax1.set_aspect('equal')
    ax1.set(xlabel='longitude', ylabel='latitude')
    # Orientation plot
    # Get roll,pitch,yaw from quaternion for visualization purposes
    rpy = R.from_quat(ds[["qx", "qy", "qz", "qw"]].to_numpy()).as_euler("XYZ", degrees=True)
    ax2.plot(rpy[:, 0], label="roll")
    ax2.plot(rpy[:, 1], label="pitch")
    ax2.plot(rpy[:, 2], label="yaw")
    ax2.axhline(90, 0, 1, color='lime', alpha=0.3, label='East(yaw=90)')
    ax2.axhline(0, 0, 1, color='lime', alpha=0.3, label='South(yaw=0)')
    ax2.axhline(-90, 0, 1, color='lime', alpha=0.3, label='West(yaw=-90)')
    ax2.set_title("orientation")
    ax2.grid(True)
    ax2.legend()
    ax2.set(xlabel='sample index', ylabel='angle, deg')
    # Connect e custom mouse event handler
    plt.gcf().canvas.mpl_connect('button_press_event', mouse_event)
    plt.tight_layout()
    plt.show()
    # Return rpy, for later inspection.
    return rpy


# Set data paths
DROOT = "./"
DSET = "dataset1.txt"

# Read map
# fmp = np.array(Image.open("fmap.png"))

# Read annotation data
d = pd.read_csv(path.join(DROOT, DSET),
                header=None, sep=" ")

#
# ==============================================================================
#  NOTE: The orientation data was exported with incorrect column order.
#        The following change corrects the problem
# ==============================================================================
# d.columns = ["image", "lon", "lat", "floor", "qw", "qx", "qy", "qz", "pmod"]  # Incorrect!
d.columns = ["image", "lon", "lat", "floor", "qy", "qz", "qw", "qx", "pmod"]    # Corrected!
#


# Get a list of trajectory ids
trj_ids = d.image.apply(lambda x: int(x[32:34]))
# Get the unique trajectory ids
utids = trj_ids.unique()


print("Right-click on position plot shows clicked point's lon/lat.")
print("Right-click on orientation plot shows clicked point's sample index and angle.")
print("Shift + Right-click on orientation plot shows position/orientation and image of the clicked point's sample index.")
for tid in utids:
    flt = (trj_ids == tid)
    dd = d[flt]   # NOTE: dd is used in mouse_event
    print(f"trajectory: t{tid:02d}")
    rpy = plot_trj(dd, title=f"trajectory: t{tid:02d}")
    # initial yaw values for each trajectory
    # print(tid, rpy[0, 2])

