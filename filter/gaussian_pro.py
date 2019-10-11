import time
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

rng = np.random.RandomState(0)

data = pd.read_csv('/Users/sotainoue1/uavTracker/filter/filter_test.csv',sep=",")
time = data.iloc[1:800:,0]
xpo = data.iloc[1:800,3]

plt.plot(time,xpo) #check original plot

xpo_1 = np.atleast_2d(xpo).T   #convert data type to 2d array 
time_1 = np.atleast_2d(time).T

scaler_xpo = StandardScaler().fit(xpo_1) #Scaling


kernel = ConstantKernel() + 1* RBF() + WhiteKernel() #modeling Gaussian process
gpr = GPR(kernel=kernel, alpha=0) #when use whiteKernel, alpha must be 0

gpr.fit(time_1, scaler_xpo.transform(xpo_1))
print(gpr.kernel_)

# prediction mean value and standard deviation of plot_X 
plot_X = np.atleast_2d(np.linspace(0, 850, 100)).T
pred_mu, pred_sigma = gpr.predict(plot_X, return_std=True)
pred_mu = scaler_xpo.inverse_transform(pred_mu)
pred_sigma = pred_sigma.reshape(-1, 1) * scaler_xpo.scale_
fig = plt.figure(figsize=(8, 6))
#plt.plot(plot_X, f(plot_X), 'k')
plt.plot(time_1,xpo_1, 'r.', markersize=3)
plt.plot(plot_X, pred_mu)


plt.savefig("fig_1011.png", dpi = 320,facecolor = "white", tight_layout = True)

plt.fill_between(plot_X.squeeze(), (pred_mu - 1.9600 * pred_sigma).squeeze(), (pred_mu + 1.9600 * pred_sigma).squeeze())
plt.xlabel('$xpo$', fontsize=16)
plt.ylabel('$time$', fontsize=16)
plt.ylim(-6.2, -5.8)
plt.tick_params(labelsize=16)
plt.plot(time_1,xpo_1, 'r.', markersize=2)
plt.show()

