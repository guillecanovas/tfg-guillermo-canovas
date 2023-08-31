'''
Created on August 20, 2020

@author: Simona Bernardi
Analysis of the CER Electricity dataset according to the Badrinath thesis:
- Loads the Y_B, P_B and testSet  for a given meterID (from the PCA analysis of matrix B)
- Execute DB-SCAN for a meterID
- Plos core, fringe and noisy points
- Detector: compute n. of false alarms considering the testSet
'''

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from robustbase.robustbase import Sn

nObs = 336  # number of readings


class dataAnalyzer:

    def __init__(self, fYB, fPB, fTS, meterID):

        # Columns of fYB: [;0;1;week;meterID]
        path = os.path.join(os.path.dirname(__file__))
        fYB = path + fYB
        Y_B = pd.read_csv(fYB, delimiter=";")

        print("Y_B dim: ", Y_B.shape)

        # Select the data related to the meterID
        meterID = int(meterID)
        self.Y_BmeterID = Y_B[Y_B.meterID == meterID]

        print("Y_BmeterID dim: ", self.Y_BmeterID.shape)

        # Columns of fPB: nObs=336  (each half-an-hour)
        fPB = path + fPB
        self.P_B = pd.read_csv(fPB, delimiter=";")
        # Transform to numpy array and remove the column 0 (the index)
        self.P_B = self.P_B.to_numpy()
        self.P_B = np.delete(self.P_B, 0, 1)
        print("P_B dim: ", self.P_B.shape)

        # Columns of fTS: [;ID;Week;DT;Usage]
        fTS = path + fTS
        testSet = pd.read_csv(fTS, delimiter=";")

        print("testSet dim: ", testSet.shape)

        # Select the data related to the meterID
        self.TS = testSet[testSet.ID == meterID]

        print("TS dim: ", self.TS.shape)

    def plotPoints(self, db, meterID):
        # Pre: there is only one cluster (0)
        # Core point mask
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True

        # Number of clusters in labels, ignoring noise if present.
        n_noise_ = list(db.labels_).count(-1)
        print("Estimated number of noise points:", n_noise_)

        # Points belonging to the cluster 0
        class_member_mask = (db.labels_ == 0)

        # Core points of the cluster 0
        coreP = self.Y_BmeterID[core_samples_mask]

        # Plot eps neighborhoods of core  points
        npoints = 2  # dimension of a point in the plot
        plt.plot(coreP[:, 0], coreP[:, 1], 'o', color='yellow', alpha=0.4,
                 markersize=5.0 * npoints * radiusSn)

        # Plot core points of the cluster 0
        plt.plot(coreP[:, 0], coreP[:, 1], 'o', color='green', markersize=npoints)

        # Fringe points of the cluster 0
        fringeP = self.Y_BmeterID[class_member_mask & ~core_samples_mask]
        # Plot fringe points of the cluster 0
        plt.plot(fringeP[:, 0], fringeP[:, 1], 'v', color='blue', markersize=npoints)

        # Noise points (outside the cluster 0)
        noise_member_mask = (db.labels_ == -1)
        noiseP = self.Y_BmeterID[noise_member_mask]
        # Plot noise points = anomalous weeks
        plt.plot(noiseP[:, 0], noiseP[:, 1], 'x', color='red', markersize=npoints)

        plt.title("Core, fringe and noise points of meterID " + str(meterID))
        plt.xlabel("Principal component 1")
        plt.ylabel("Principal component 2")
        plt.show()

    def getMatrixB(self, mat):
        # Matrix B: rearranged from df dimension B[nObs=336,nMeterID X nweeks]
        nWeeks = int(mat.shape[0] / nObs)
        print("N. of weeks: ", nWeeks)

        # Week 0 for the first week
        B = mat[0:nObs, :]

        for i in range(nWeeks - 1):
            B = np.block([B, mat[nObs * (i + 1):nObs * (i + 2), :]])

        return B

    def computeOutliers(self, db, radius):

        # Calculate matrices A and B for the testSet
        A = mg.TS.pivot(index='DT', columns='ID', values='Usage').to_numpy()
        print("A original shape: {}".format(str(A.shape)))
        B = mg.getMatrixB(A)
        print("B original shape: {}".format(str(B.shape)))

        # Get the reduced matrix for the testSet
        Y_Btest = np.matmul(mg.P_B, B)

        # Getting the transpose
        Y_Btest = np.transpose(Y_Btest)
        print("Y_Btest shape: ", Y_Btest.shape)

        # Get the core points of the cluster 0
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        coreP = self.Y_BmeterID[core_samples_mask]

        # Detect the number of points outside the cluster
        count = 0
        for p in Y_Btest:
            min_dist = np.min(euclidean_distances(coreP, [p]))
            if min_dist <= radius:
                count += 1
        return count, len(Y_Btest)


if __name__ == '__main__':

    '''
    args: 
    sys.argv[1]: Y_B file
    sys.argv[2]: P_B file
    sys.argv[3]: testSet file
    sys.argv[4]: meterID

    '''

    if (len(sys.argv) == 5):
        ###########################################################
        # Load (Y_B, P_B, testSet) and select Y_B[meterID]
        ###########################################################
        mg = dataAnalyzer(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

        # Convert the selected columns to numpy array
        mg.Y_BmeterID = mg.Y_BmeterID[['0', '1']].to_numpy()
        # YB[:,1] = -YB[:,1] # The component 2 of PCA is specular w.r.t. the result of Badrinath

        # ===========================================================
        # Set DBSCAN parameters: eps and minPts
        # ===========================================================
        # eps: Radius for the cluster

        # Compute the pairwise Euclidean distance
        paird = pairwise_distances(mg.Y_BmeterID, metric='euclidean')
        # The radius is set calculating the Sn measure of Rousseeuw and Croux
        # Flatten the distance matrix paird (ravel)
        radiusSn = Sn(paird.ravel())
        print("Radius: ", radiusSn)

        # minPts: The (min) number of points in a neighborhood for a point to be considered as a core point.
        # The number of points is the majority of the entire set of points
        minPts = int(mg.Y_BmeterID.shape[0] / 2) + 1
        print(minPts)
        # ===========================================================
        # Build DBSCAN model
        # ===========================================================
        db = DBSCAN(eps=radiusSn, min_samples=minPts).fit(mg.Y_BmeterID)

        # ===========================================================
        # Plot core, fringe and noisy points
        # ===========================================================

        mg.plotPoints(db, sys.argv[4])

        # ===========================================================
        # Compute outliers using the tests dataset
        # ===========================================================

        n_false, n_obs = mg.computeOutliers(db, radiusSn)

        print("ACCURACY RESULTS -----------------------")
        print("n. false alarms: ", n_false)
        print("n. observations: ", n_obs)
        print("% of false alarms: ", n_false / n_obs)