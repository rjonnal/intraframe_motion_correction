import numpy as np
from matplotlib import pyplot as plt
import glob,os,sys
from scipy.interpolate import griddata

# This program and the associated data illustrate how to do strip-registration
# on AO-SLO frames and use the resulting registration statistics to correct
# intra-frame motion in the reference frame. References to equations and figures
# are to Azimipour et al., PLoS One, 2018.

source_directory = 'slo_frames_real_small'
output_directory = os.path.join(source_directory,'stabilization')
resource_directory = os.path.join(source_directory,'resources')

reference_filename = os.path.join(source_directory,'065.npy')
strip_width = 13
show_plots = False
redo = False

filename_list = glob.glob(os.path.join(source_directory,'*.npy'))
filename_list.sort()
n_frames = len(filename_list)

# Make a directory for storing results such as the stabilized reference
# frame.
try:
    os.mkdir(output_directory)
except Exception as e:
    print e

# Make a directory for caching results to avoid re-running slow registration
# process.
cache_dir = os.path.join('.','tmp')
try:
    os.mkdir(cache_dir)
except OSError as ose:
    # catch and ignore in case directory already exists
    pass


# Make a list of frame filenames, and load the desired reference frame.
ref = np.load(reference_filename)
ref_frame_index = int(os.path.split(reference_filename)[1].replace('.npy',''))
n_rows,n_cols = ref.shape

# We compute lags for each row of each frame, so n_strips is equal to the
# number of lines.
n_strips = n_rows

# To compute cross-correlations we need the FFT2 of the reference frame;
# pre-compute this here.
f_ref = np.fft.fft2(ref)

# Now, we'll loop through the frames. On each frame we loop through the
# strips and compute cross-correlation. Let's make some N_FRAMES x N_STRIPS
# matrices in which to store the results (x and y lags, and correlations).
# The values computed in these loops are described by equations 1 and 2.
# We will try to follow the nomenclature in the paper closely, with the
# lag matrices called s_x and s_y, and so forth:
s_x_filename = os.path.join(cache_dir,'%s_s_x_%03d.npy'%(source_directory,ref_frame_index))
s_y_filename = os.path.join(cache_dir,'%s_s_y_%03d.npy'%(source_directory,ref_frame_index))
corr_filename = os.path.join(cache_dir,'%s_corrs_%03d.npy'%(source_directory,ref_frame_index))


def cross_correlate(im1,im2):
    im1c = (im1-im1.mean())/im1.std()
    im2c = (im2-im2.mean())/im2.std()
    
    xc = np.abs(np.fft.fftshift(np.fft.ifft2(np.fft.fft2(im1c)*np.conj(np.fft.fft2(im2c)))))
    xc = xc/np.prod(xc.shape)
    return xc

# If these have been previously cached, just load the old versions. If not
# run through the long registration loops.
try:
    s_x = np.load(s_x_filename)
    s_y = np.load(s_y_filename)
    corrs = np.load(corr_filename)
except Exception as e:
    print e
    s_x = np.zeros((n_frames,n_strips))
    s_y = np.zeros((n_frames,n_strips))
    corrs = np.zeros((n_frames,n_strips))

    for frame_index,filename in enumerate(filename_list):
        target_frame = np.load(filename)

        for row_index in range(n_rows):
            print frame_index,row_index
            # Make a rectangular window of the same shape
            # as the frame, zeros everywhere except for
            # strip_width rows, centered about row_index
            window = np.ones(target_frame.shape)
            distance = np.abs(np.arange(n_rows)-row_index)
            mask = np.zeros(distance.shape)
            mask[np.where(distance<=strip_width//2)] = 1
            # A bit of transposing weirdness in order to
            # multiply by broadcasting:
            window = (window.T*mask).T
            tar = target_frame*window
            f_tar = np.conj(np.fft.fft2(tar))
            xcorr = np.abs(np.fft.ifft2(f_tar*f_ref))
            # Remove cross-correlation bias due to
            # horizontal strips. This is an alternative to
            # normalizing the cross-correlation (mean subtracting
            # and dividing by standard deviation). Normalization
            # is difficult because we would have to normalize the
            # reference by the statistics of the matched reference
            # region, but we don't yet know the matched region. 
            mean_xcorr = np.mean(xcorr,axis=1)
            #std_xcorr = np.std(xcorr,axis=1)
            xcorr = (xcorr.T-mean_xcorr).T

            y_lag,x_lag = np.unravel_index(np.argmax(xcorr),xcorr.shape)
            corr = xcorr[y_lag,x_lag]

            # Correct lags for circular cross-correlation:
            if y_lag>n_rows//2:
                y_lag = y_lag - n_rows

            if x_lag>n_cols//2:
                x_lag = x_lag - n_cols

            if show_plots:
                plt.clf()
                plt.imshow(np.fft.fftshift(xcorr))
                plt.colorbar()
                plt.pause(1)

            # Put the x_lag, y_lag, and correlation
            # into their respective locations.
            s_x[frame_index,row_index] = x_lag
            s_y[frame_index,row_index] = y_lag
            corrs[frame_index,row_index] = corr

    np.save(s_x_filename,s_x)
    np.save(s_y_filename,s_y)
    np.save(corr_filename,corrs)

# Now, as described in equations 3 and 4, we compute the lag differences between
# successive strips
s_hat_x = np.diff(np.hstack((s_x[:,0:1],s_x)),axis=1)
s_hat_y = np.diff(np.hstack((s_y[:,0:1],s_y)),axis=1)

# Now, on the assumption that the relative shift between successive rows of an SLO
# frame must be smaller than some nominal value, we can exclude outliers. Oridnarily
# we would compute this threshold as a function of saccadic eye movements, but for
# simplicity here we'll limit inter-row shifts to 2 pixels. We'll replace outliers
# with NaN to make statistical calculations simple.
s_hat_x[np.where(np.abs(s_hat_x)>2)] = np.nan
s_hat_y[np.where(np.abs(s_hat_y)>2)] = np.nan

# The next step, described in equations 5 and 6, is calculation of the lag biases
# by averaging the inter-row shifts across frames.
delta_x_r = np.nanmean(s_hat_x,axis=0)
delta_y_r = np.nanmean(s_hat_y,axis=0)
#delta_x_r = np.nansum(s_hat_x,axis=0)
#delta_y_r = np.nansum(s_hat_y,axis=0)

# Next, as described in equations 7 and 8, we integrate these lag biases (and
# invert) in order to obtain meaningful traces of eye motion in the reference frame
x_hat_t = np.cumsum(delta_x_r)
y_hat_t = np.cumsum(delta_y_r)

# Finally, as described in equations 9 and 10, we interpolate the reference frame
# from unstabilized coordinates into a coordinate set stabilized by the eye motion
# traces x_hat_t and y_hat_t
unstabilized_x = np.arange(n_rows)
unstabilized_y = np.arange(n_rows)
uXX,uYY = np.meshgrid(unstabilized_x,unstabilized_y)

sXX,sYY = uXX.copy(),uYY.copy()
for row in range(n_rows):
    sXX[row,:] = sXX[row,:]+x_hat_t[row]
    sYY[row,:] = sYY[row,:]+y_hat_t[row]

points = np.vstack((uXX.ravel(),uYY.ravel())).T
values = ref.ravel()

corrected_reference = griddata(points,values,(sXX.ravel(),sYY.ravel()),method='cubic')
corrected_reference = np.reshape(corrected_reference,ref.shape)
corrected_reference[np.where(np.isnan(corrected_reference))] = np.nanmean(corrected_reference)
np.save(os.path.join(output_directory,os.path.split(reference_filename)[1]),corrected_reference)


# If a motion-free image is available (i.e. for simulated trials), load the motion-free version
# and visualize the improvement through cross-correlation of motion-free with unstabilized and
# stabilized versions of the reference.
try:
    motion_free = np.load(os.path.join(resource_directory,'motion_free.npy'))
    unstabilized_xc = cross_correlate(ref,motion_free)
    stabilized_xc = cross_correlate(corrected_reference,motion_free)
    
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(unstabilized_xc,cmap='gray')
    plt.title('unstabilized xcorr')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(stabilized_xc,cmap='gray')
    plt.title('stabilized xcorr')
    plt.colorbar()

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(ref,cmap='gray')
    plt.title('unstabilized reference')
    plt.subplot(1,3,2)
    plt.imshow(motion_free,cmap='gray')
    plt.title('object')
    plt.subplot(1,3,3)
    plt.imshow(corrected_reference,cmap='gray')
    plt.title('stabilized reference')

except Exception as e:
    print e

    
# If true eye-motion traces are available (i.e. for simulated trials), load the eye traces
# and plot for comparison with reconstructed motion traces.
try:
    x_trace = np.load(os.path.join(resource_directory,'eye_trace_x.npy'))[ref_frame_index,:]
    y_trace = np.load(os.path.join(resource_directory,'eye_trace_y.npy'))[ref_frame_index,:]
    plt.figure()
    plt.plot(x_trace,label='x')
    plt.plot(y_trace,label='y')
    plt.title('simulated eye movements')
    plt.legend()
except Exception as e:
    print e

# Plot the reconstructed eye movement trace.
plt.figure()
plt.plot(-x_hat_t,label='x')
plt.plot(-y_hat_t,label='y')
plt.title('reconstructed eye movement')
plt.legend()

plt.figure()
plt.subplot(1,2,1)
plt.imshow(ref,cmap='gray')
plt.title('unstabilized reference')
plt.subplot(1,2,2)
plt.imshow(corrected_reference,cmap='gray')
plt.title('stabilized reference')
plt.show()
