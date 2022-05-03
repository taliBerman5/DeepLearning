import scipy.io as sio

GMM = sio.loadmat('GMMData.mat')
Peaks = sio.loadmat('PeaksData.mat')
SwissRoll = sio.loadmat('SwissRollData.mat')

Ct_SwissRoll = SwissRoll["Ct"]
Cv_SwissRoll = SwissRoll["Cv"]
Yt_SwissRoll = SwissRoll["Yt"]
Yv_SwissRoll = SwissRoll["Yv"]

Ct_GMM = GMM["Ct"]
Cv_GMM = GMM["Cv"]
Yt_GMM = GMM["Yt"]
Yv_GMM = GMM["Yv"]

Ct_Peaks = Peaks["Ct"]
Cv_Peaks = Peaks["Cv"]
Yt_Peaks = Peaks["Yt"]
Yv_Peaks = Peaks["Yv"]
