import numpy as np 
import pandas as pd
from numpy import arange, array, ones
from numpy.linalg import lstsq
import os
import pickle
from utils import parameters as parm

from matplotlib.pylab import gca, figure, plot, subplot, title, xlabel, ylabel, xlim,show
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12,4) #(20, 8)


# define dirs
data_dir = parm.data_dir
base_dir = parm.base_dir
cache_dir = parm.cache_dir
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)


# draw
def draw_plot(start, data, plot_title):
    plot(range(start, start+len(data)),data,alpha=0.8,color='red',marker='*')
    title(plot_title)
    xlabel("Samples")
    ylabel("Signal")
    xlim((start, start+len(data)-1))

# draw line
def draw_segments(segments):
    ax = gca()
    for segment in segments:
        line = Line2D((segment[0],segment[2]),(segment[1],segment[3]))
        ax.add_line(line)

# draw main function
def draw_part(segments_list, data, start_ids=0, end_ids=5, num_seg=50):
    # 从前向后画,从start_ids,到end_ids组趋势，每组num_seg个趋势片段
    for ids in sorted([i*num_seg for i in range(start_ids, end_ids)]):
        target = segments_list[ids:(ids+num_seg)]
        figure()
        draw_plot(target[0][0], data[target[0][0] : target[-1][2]],"PLR with simple interpolation")
        draw_segments(target)
        plt.show()
        plt.close()


# compute plr segments
def plr_seg(data, max_error=0.0005, TD=1): #6753个
    """
    generate plr segments
    out: [[x0,y0,x1,y1]...]
    """

    # using lstsq
    def _leastsquareslinefit(sequence,seq_range):
        """Return the parameters and error for a least squares line fit of one segment of a sequence"""
        x = arange(seq_range[0],seq_range[1]+1)
        y = array(sequence[seq_range[0]:seq_range[1]+1])
        A = ones((len(x),2),float)
        A[:,0] = x
        (p,residuals,rank,s) = lstsq(A,y)
        try:
            error = residuals[0]
        except IndexError:
            error = 0.0
        return (p,error)

    # compute_error functions
    def _sumsquared_error(sequence, segment):
        """Return the sum of squared errors for a least squares line fit of one segment of a sequence"""
        x0,y0,x1,y1 = segment
        p, error = _leastsquareslinefit(sequence,(x0,x1))
        return error
        
    # create_segment functions
    def _regression(sequence, seq_range):
        """Return (x0,y0,x1,y1) of a line fit to a segment of a sequence using linear regression"""
        p, error = _leastsquareslinefit(sequence,seq_range)
        y0 = p[0]*seq_range[0] + p[1]
        y1 = p[0]*seq_range[1] + p[1]
        return (seq_range[0],y0,seq_range[1],y1)

    def _interpolate(sequence, seq_range):
        """Return (x0,y0,x1,y1) of a line fit to a segment using a simple interpolation"""
        return (seq_range[0], sequence[seq_range[0]], seq_range[1], sequence[seq_range[1]])

    # compute PLR using BU
    def _bottomupsegment(sequence, create_segment, compute_error, max_error):
        """
        Return a list of line segments that approximate the sequence.
        The list is computed using the bottom-up technique.
        Parameters
        ----------
        sequence : sequence to segment
        create_segment : a function of two arguments (sequence, sequence range) that returns a line segment that approximates the sequence data in the specified range
        compute_error: a function of two argments (sequence, segment) that returns the error from fitting the specified line segment to the sequence data
        max_error: the maximum allowable line segment fitting error
        """
        segments = [create_segment(sequence,seq_range) for seq_range in zip(range(len(sequence))[:-1],range(len(sequence))[1:])]
        mergesegments = [create_segment(sequence,(seg1[0],seg2[2])) for seg1,seg2 in zip(segments[:-1],segments[1:])]
        mergecosts = [compute_error(sequence, segment) for segment in mergesegments]

        while min(mergecosts) < max_error:
            idx = mergecosts.index(min(mergecosts))
            segments[idx] = mergesegments[idx]
            del segments[idx+1]

            if idx > 0: 
                mergesegments[idx-1] = create_segment(sequence,(segments[idx-1][0],segments[idx][2])) # 因为删了一个原始的点所以更新前边的mergesegemnts
                mergecosts[idx-1] = compute_error(sequence,mergesegments[idx-1])

            if idx+1 < len(mergecosts):
                mergesegments[idx+1] = create_segment(sequence,(segments[idx][0],segments[idx+1][2])) # 因为删了一个原始的点所以更新后边的mergesegemnts
                mergecosts[idx+1] = compute_error(sequence,mergesegments[idx])

            del mergesegments[idx] # 因为mergesegments是跨一个点merge的所以应该删除这个没有跨点的merge点
            del mergecosts[idx] # 删除对应的cost点，继续找下一个最小cost的merge点

        return segments

    
    def _topdownsegment(sequence, create_segment, compute_error, max_error, seq_range=None):
        """
        Return a list of line segments that approximate the sequence.

        The list is computed using the bottom-up technique.

        Parameters
        ----------
        sequence : sequence to segment
        create_segment : a function of two arguments (sequence, sequence range) that returns a line segment that approximates the sequence data in the specified range
        compute_error: a function of two argments (sequence, segment) that returns the error from fitting the specified line segment to the sequence data
        max_error: the maximum allowable line segment fitting error

        """
        if not seq_range:
            seq_range = (0,len(sequence)-1)

        bestlefterror,bestleftsegment = float('inf'), None
        bestrighterror,bestrightsegment = float('inf'), None
        bestidx = None

        for idx in range(seq_range[0]+1,seq_range[1]):
            segment_left = create_segment(sequence,(seq_range[0],idx))
            error_left = compute_error(sequence,segment_left)
            segment_right = create_segment(sequence,(idx,seq_range[1]))
            error_right = compute_error(sequence, segment_right)
            if error_left + error_right < bestlefterror + bestrighterror:
                bestlefterror, bestrighterror = error_left, error_right
                bestleftsegment, bestrightsegment = segment_left, segment_right
                bestidx = idx

        if bestlefterror <= max_error:
            leftsegs = [bestleftsegment]
        else:
            leftsegs = _topdownsegment(sequence, create_segment, compute_error, max_error, (seq_range[0],bestidx))

        if bestrighterror <= max_error:
            rightsegs = [bestrightsegment]
        else:
            rightsegs = _topdownsegment(sequence, create_segment, compute_error, max_error, (bestidx,seq_range[1]))

        return leftsegs + rightsegs


    if TD == 0:
        segments = _bottomupsegment(data, _interpolate, _sumsquared_error, max_error)
    else:
        segments = _topdownsegment(data, _interpolate, _sumsquared_error, max_error)
        
    return segments


# load data from npy file
def load_data_npy(data_path):
    data = np.load(data_path).tolist()
    return data


def main(machineID, max_error, TD=1):
    file_name = 'cpu_rate_%s.npy' % machineID
    data_path = os.path.join(data_dir, file_name)
    
    print(data_path)
    data = load_data_npy(data_path) # list

    segments = plr_seg(data, max_error=max_error, TD=TD) # list
    machineID_cache_dir = os.path.join(cache_dir, machineID) # 为每一个machineID单独创建cache dir
    if(not os.path.exists(machineID_cache_dir)):
        os.mkdir(machineID_cache_dir)
    
    cached_seg_file_name_prefix = machineID + 'TD_ ' + str(TD) +'_' + str(max_error)
    res = {'segments': segments, 'data':data}
    with open(os.path.join(machineID_cache_dir, cached_seg_file_name_prefix + '_segments_list.pkl'), 'wb') as f:
        pickle.dump(res ,f)

    print('total length:', len(data)) # 41760
    print('total segments:',len(segments))
    draw_part(segments, data, start_ids=0, end_ids=3, num_seg=50)