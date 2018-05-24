import numpy as np
from scipy import interpolate
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import os,sys

output_directory = os.path.join('.','slo_frames_simulated')
# Specify the size of the output images
nx = 128
ny = 128

# Specify the number of frames
n_frames = 200

# Simulate idiosyncratic motion
simulate_idiosyncratic = False

resource_directory = os.path.join(output_directory,'resources')
# Create the output and resource directories
try:
    os.mkdir(output_directory)
except Exception as e:
    sys.exit('%s exists; please delete or specify a new directory.'%output_directory)

try:
    os.mkdir(resource_directory)
except Exception as e:
    sys.exit('%s exists; please delete or specify a new directory.'%resource_directory)

# Load the object (retina)
retina = np.load(os.path.join('object','full_mosaic.npy'))

# Load simulated eye traces
gx = np.load(os.path.join('simulated_eye_traces','eye_trace_x.npy'))
gy = np.load(os.path.join('simulated_eye_traces','eye_trace_y.npy'))

# Magnitude of eye movements may be modified if desired:
gx = gx * 1.5
gy = gy * 1.5

if simulate_idiosyncratic:
    # Add idiosyncratic motion of 10% of image width
    # The average shift per acquired line is calculated from the total desired
    # idiosyncratic drift. For example, it will be 0.001 * expected_value of
    # np.random.rand, i.e. 0.001 * 0.5 = 0.0005. Over 200*128 lines, this
    # results in an overall shift of about 12.8 pixels.
    n_saccades = 3.0
    drift_per_saccade = 10.0
    total_drift = n_saccades*drift_per_saccade
    drift_per_line = total_drift/float(ny)/float(n_frames)
    x_idio = np.cumsum(np.random.rand(ny*n_frames)*2.0*drift_per_line)%drift_per_saccade
    x_idio = np.reshape(x_idio,gx.shape)
    gx = gx + x_idio

np.save(os.path.join(resource_directory,'eye_trace_x.npy'),gx)
np.save(os.path.join(resource_directory,'eye_trace_y.npy'),gy)
    
# Select a region of the retina to image. mx0,my0 represents the upper left corner.
mx0 = 100
my0 = 100
motion_free = retina[my0:my0+ny,mx0:mx0+nx]

np.save(os.path.join(resource_directory,'motion_free.npy'),motion_free)

for frame_index in range(n_frames):
    frame = []
    # get the eye motion trace for this frame
    frame_gx = gx[frame_index,:]
    frame_gy = gy[frame_index,:]
    for idx,(x,y) in enumerate(zip(frame_gx,frame_gy)):
        # now we have the fixation position,
        # we have to interpolate the image
        # from object coordinates into
        # these offset coordinates

        xpx = x+mx0
        ypx = y+my0+idx
        
        x1 = np.floor(xpx)
        x2 = x1 + 1
        leftfrac = np.abs(xpx-x2)
        rightfrac = np.abs(xpx-x1)

        y1 = np.floor(ypx)
        y2 = y1 + 1
        topfrac = np.abs(ypx-y2)
        bottomfrac = np.abs(ypx-y1)

        y1,y2,x1,x2=int(round(y1)),int(round(y2)),int(round(x1)),int(round(x2))
        
        topleft = retina[y1,x1:x1+nx]
        topright = retina[y1,x2:x2+nx]
        bottomleft = retina[y2,x1:x1+nx]
        bottomright = retina[y2,x2:x2+nx]

        line = leftfrac*topfrac*topleft + leftfrac*bottomfrac*bottomleft + rightfrac*topfrac*topright + rightfrac*bottomfrac*bottomright
        frame.append(line)
        
    frame = np.array(frame)
    np.save(os.path.join(output_directory,'%03d.npy'%frame_index),frame)
    plt.cla()
    plt.imshow(frame,cmap='gray')
    plt.pause(.001)
