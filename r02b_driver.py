import logging
import operator
import pandas as pd
import numpy as np
import numpy.typing as npt
from scipy.cluster.hierarchy import fclusterdata
from westpa.core.we_driver import WEDriver
from westpa.core.binning.bins import Bin
from westpa.core.segment import Segment

log = logging.getLogger(__name__)

class ClusteringDriver(WEDriver):
    '''
    This class implements Huber and Kim weighted ensemble using distance-based clustering on a 2D progress coordinate
    to define split/merge eligibility rather than user-defined bins. Includes functions to send splitting and merging
    to base WESTPA functions, as well as a function containing the split/merge logic. There is also a function to 
    optionally print out a dataframe of walkers that were split each iteration to the west.log file.
    '''
    def _split_by_data(self, bin: Bin, to_split: Segment, split_into: int) -> None:
        '''
        Takes in a Segment object containing a walker to split, and an int determining
        how many child Segments to make from the parent Segment. 
        '''
        #remove the walker to split from the existing bin
        bin.remove(to_split)
        #get a list of child Segments by sending the parent Segment to _split_walker
        new_segments_list = self._split_walker(to_split, split_into, bin)
        #add the child Segments to the bin
        bin.update(new_segments_list)

    def _merge_by_data(self, bin: Bin, to_merge: npt.NDArray) -> None:
        '''
        Takes in a Numpy array containing Segment objects to merge.
        '''
        #removes walkers that are in both to_merge and the existing bin
        bin.difference_update(to_merge)
        #use _merge_walkers function to create conglomerate walker
        new_segment, parent = self._merge_walkers(to_merge, None, bin)
        #add new conglomerate walker to the bin
        bin.add(new_segment)

    def _split_and_merge(self, cluster_idx: npt.NDArray, ideal_threshold: float, weights: npt.NDArray, 
                         segments: npt.NDArray, to_split_idx: list, to_merge_idx: list, bin: Bin) -> tuple[list,list]:
        '''
        Implements Huber and Kim splitting and merging for a given cluster array of Segments.
        Parameters:
        Numpy array representing one cluster of Segments as cluster_idx
        Ideal weight of this cluster
        Numpy array of Segment weights for this cluster
        Numpy array of weights of all Segments for this bin
        Numpy array of all Segments in this bin
        List of indices for which walkers will be split this iteration
        Output:
        Updated list of split indices
        '''
        #calculate the ideal weight for this cluster
        cluster_weights = weights[cluster_idx]
        ideal_weight = np.mean(cluster_weights) * ideal_threshold
        #iterate over each Segment index in the cluster
        for idx in cluster_idx:
            #split if the weight of this segment is greater than or equal to the ideal weight
            if weights[idx] >= 2 * ideal_weight:
                #split into a number that will ensure child walkers have weight less than or equal to twice the ideal weight
                split_into = max(2, int(weights[idx] / (2 * ideal_weight)))
                to_split_idx.append(idx)
                self._split_by_data(bin, segments[idx], split_into)
                #remove the split walker from the walkers in the cluster eligible for merging
                cluster_idx = np.delete(cluster_idx, np.argwhere(cluster_idx == idx))

        #sort the leftover walkers in the cluster by weight from lightest to heaviest
        sorted_idx = [x for (y,x) in sorted(zip(cluster_weights, cluster_idx), key=lambda pair: pair[0])]
        merge_list, merge_weight = [], 0

        #iterate over the walkers in the sorted list
        for idx in sorted_idx:
            #if the weight of the walker is greater than half the ideal weight:
            if weights[idx] > ideal_weight / 2:
                #break out of the loop if the merge list is empty, since the starting weight
                #is already ineligible for merging
                if merge_weight == 0:
                    break
                #if the current merge pile + the weight of the current walker
                #is greater than 1.5 times the ideal weight, merge the pile without
                #including the current walker and break
                elif merge_weight + weights[idx] > 1.5 * ideal_weight:
                    if len(merge_list) > 1:
                        self._merge_by_data(bin, segments[merge_list])
                        print("Merge Done")
                        to_merge_idx.extend(merge_list)
                    break
                #if the current merge pile + the weight of the current walker would be 
                #less than or equal to 1.5 times the ideal weight, add the current walker
                #to the merge pile and merge. Then break
                else:
                    merge_list.append(idx)
                    merge_weight += weights[idx]
                    if len(merge_list) > 1:
                        self._merge_by_data(bin, segments[merge_list])
                        print("Merge Done")
                        to_merge_idx.extend(merge_list)
                    break
            #if the weight of the current walker is less than or equal to half the ideal weight,
            #but the weight of the current merge pile is greater than the ideal weight, merge the 
            #current pile without including the current walker
            elif merge_weight > ideal_weight:
                if len(merge_list) > 1:
                    self._merge_by_data(bin, segments[merge_list])
                    print("Merge Done")
                    to_merge_idx.extend(merge_list)
                    merge_list, merge_weight = [], 0
            #if the walker has passed all these checks, then it is eligible for merging and it is
            #added to the merge pile
            merge_list.append(idx)
            merge_weight += weights[idx]

        return to_split_idx, to_merge_idx

    def _log_clusters(self, pcoords: npt.NDArray, weights: npt.NDArray, bins: Bin, distance_magnitude: float, to_split_idx: list, to_merge_idx: list) -> None:
        '''
        Function to optionally print out the current clusters and the splits that have occurred in this iteration
        '''
        #current cluster dataframe
        mydf = pd.DataFrame({
            'pcoords d1': pcoords[:, 0],
            'pcoords d2': pcoords[:, 1],
            'bins': bins,
            'distance magnitude': distance_magnitude
        })
        print("\n", mydf.sort_values(by=['distance magnitude']))

        #dataframe of splits this iteration
        if to_split_idx:
            mydf_split = pd.DataFrame({
                'split index': to_split_idx,
                'pcoords d1': pcoords[:, 0][to_split_idx],
                'pcoords d2': pcoords[:, 1][to_split_idx],
                'weights': weights[to_split_idx],
                'bins': bins[to_split_idx],
                'distance magnitude': distance_magnitude[to_split_idx]
            })
            print("\n", mydf_split)
        
        #dataframe of merges this iteration
        if to_merge_idx:
            mydf_merge = pd.DataFrame({
                'merge index': to_merge_idx,
                'pcoords d1': pcoords[:, 0][to_merge_idx],
                'pcoords d2': pcoords[:, 1][to_merge_idx],
                'weights': weights[to_merge_idx],
                'bins': bins[to_merge_idx],
                'distance magnitude': distance_magnitude[to_merge_idx]
            })
            print("\n", mydf_merge)

    def _calculate_distances(self, pcoords: npt.NDArray, target_d1=0, target_d2=0) -> float:
        '''
        Calculate the distance of each walker from the given target state
        '''
        distances_d1 = np.abs(pcoords[:, 0] - target_d1) ** 2
        distances_d2 = np.abs(pcoords[:, 1] - target_d2) ** 2
        return np.sqrt(distances_d1 + distances_d2)

    def _run_we(self):
        '''
        Run resampling. The workflow is:
        1. Recycle walkers
        2. Cluster the current distribution of walkers
        3. Split and merge within clusters
        '''
        self._recycle_walkers()

        #sanity check
        self._check_pre()

        for bin in self.next_iter_binning:
            if not bin:
                continue
            else:
                #pull information about the current distribution of walkers
                segments = np.array(sorted(bin, key=operator.attrgetter('weight')), dtype=np.object_)
                pcoords_d1 = np.array(list(map(operator.attrgetter('pcoord'), segments)))[:,:,0]
                pcoords_d2 = np.array(list(map(operator.attrgetter('pcoord'), segments)))[:,:,1]
                weights = np.array(list(map(operator.attrgetter('weight'), segments)))
                
                #collect pcoord information into one array
                pcoords = np.column_stack((pcoords_d1[:, 0], pcoords_d2[:, 0]))
                
                #pull the pcoords for all frames to check for initialization
                nsegs, nframes = pcoords_d1.shape
                current_iter_segments = np.array(
                    sorted(self.current_iter_segments, key=operator.attrgetter('weight')), dtype=np.object_
                )
                curr_pcoords_d1 = np.array(list(map(operator.attrgetter('pcoord'), current_iter_segments)))[:, :, 0].reshape(nsegs, nframes)

                #check if not initializing
                if np.any(curr_pcoords_d1[:, 0] != curr_pcoords_d1[:, -1]):
                    #check for recycled walkers (where parentid is negative)
                    recycled_walkers = [seg for seg in segments if seg.parent_id < 0]
                    #if there are recycled walkers, merge them together
                    if recycled_walkers:
                        self._merge_by_data(bin, np.array(recycled_walkers))

                        #now, since the bin has been updated, the segment data needs to be re-pulled
                        segments = np.array(sorted(bin, key=operator.attrgetter('weight')), dtype=np.object_)
                        pcoords_d1 = np.array(list(map(operator.attrgetter('pcoord'), segments)))[:,:,0]
                        pcoords_d2 = np.array(list(map(operator.attrgetter('pcoord'), segments)))[:,:,1]
                        pcoords = np.column_stack((pcoords_d1[:, 0], pcoords_d2[:, 0]))
                        weights = np.array(list(map(operator.attrgetter('weight'), segments)))
                    
                    #calculate the distance of each walker from the given target state in each dimension
                    distance_magnitude = self._calculate_distances(pcoords)
                    
                    #cluster the current distribution of walkers using the max - min of the current walker distances
                    #e.g., the furthest walker from the target state - the closest
                    #this value is then multiplied by a constant
                    #the cluster_threshold determines how finely the clusters will be spaced. This will be system dependent
                    cluster_threshold = np.ptp(distance_magnitude) * 0.14
                    bins = fclusterdata(pcoords, t=cluster_threshold, criterion='distance', method='ward')

                    #calculate the average progress of each cluster, as determined by the average distance from the target state
                    #of each walker in the cluster
                    cluster_avgprogress = [
                        np.mean(distance_magnitude[np.where(bins == ibin)])
                        for ibin in np.unique(bins)
                    ]
                    #sort clusters by progress from the least progress to the most progress
                    sorted_clusters = np.argsort(cluster_avgprogress)[::-1] + 1
                    #create lists to keep track of which walkers are split and merged
                    to_split_idx, to_merge_idx = [], []
                    #start a bin count to keep track of how many clusters have been iterated over
                    #this will be important since the splitting/merging thresholds are lowered for the top 2% of clusters
                    bincount = 1
                    #iterate over clusters for splitting and merging
                    for ibin in sorted_clusters:
                        #if the cluster is in the bottom 98% of clusters in terms of progress, set the ideal weight as the mean
                        #of the weights in the cluster
                        if bincount < (len(sorted_clusters) * 0.98):
                            #pull the indices of the walkers in this cluster
                            cluster_idx = np.where(bins == ibin)[0]
                            ideal_threshold = 1.0
                            to_split_idx, to_merge_idx = self._split_and_merge(cluster_idx, ideal_threshold, weights, 
                                                                               segments, to_split_idx, to_merge_idx, bin)
                        #if the cluster is in the top 2% of clusters in terms of progress towards the target state,
                        #set the ideal weight threshold at 1/2 its normal value to encourage splits and lower the probability of merges
                        else:
                            cluster_idx = np.where(bins == ibin)[0]
                            ideal_threshold = 0.5
                            to_split_idx, to_merge_idx = self._split_and_merge(cluster_idx, ideal_threshold, weights, 
                                                                               segments, to_split_idx, to_merge_idx, bin)
                        bincount +=1
                    self._log_clusters(pcoords, weights, bins, distance_magnitude, to_split_idx, to_merge_idx)
                                
        # another sanity check
        self._check_post()

        self.new_weights = self.new_weights or []

        log.debug('used initial states: {!r}'.format(self.used_initial_states))
        log.debug('available initial states: {!r}'.format(self.avail_initial_states))