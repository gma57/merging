# Improved Merging through Clustering

This driver implements Huber and Kim splitting and merging in clusters rather than bins. Clusters are defined through distance-based heirarchical clustering on the current distribution of walkers. The cutoff distance can be freely modified and may be system dependent. A smaller cutoff distance is analogous to finer bin spacing, which may be necessary for more challenging systems.

To use this driver, place it in the WESTPA simulation folder and specify ClusteringDriver as the system driver in the west.cfg. The WESTPA simulation can then be run normally (by calling init.sh and then run.sh, or w_init and w_run).

Note that in WESTPA 2.0 the target state definition and the binning scheme are coupled, so defining the target state in terms of rectilinear bins will result in three non-target state "macro" bins. This resampler will run separately in each of these "macro" bins. I worked around this by defining a custom BinMapper that defined just two bins, target state and not-target state, but this problem may be resolved in the next version of WESTPA.
