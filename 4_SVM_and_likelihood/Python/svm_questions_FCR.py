#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Mariana Clare mc4117@ic.ac.uk
"""


import numpy as np
import matplotlib.pyplot as plt

# use seaborn plotting defaults
import seaborn as sns; sns.set()

from sklearn.datasets.samples_generator import make_blobs

# consider two classes of points which are well separated
X, y = make_blobs(n_samples=50, centers=2,
                  random_state=0, cluster_std=0.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

plt.show()

np.shape(X)

# Task 1: Attempt to use linear regression to separate this data using linear regression.
# Note there are several possibilities which separate the data? 
# What happens to the classification of point [0.6, 2.1] (or similar)?


xfit = np.linspace(-1, 3.5)
print(xfit)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

for m, b in [(0.8, 0.23), (1.2, 1.5), (-0.5, 3.4)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    
plt.plot([0.6], [2.1], 'x', color = 'red', markeredgewidth = 2, markersize = 10)

plt.xlim(-1, 3.5)
plt.show()



# With SVM rather than simply drawing a zero-width line between the 
# classes, we draw a margin of some width around each line, up to the nearest point. 
# For example for these lines (n.b. d is uncertainty around your line):


xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]: 
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
                     color='#AAAAAA', alpha=0.4)

plt.xlim(-1, 3.5)
plt.show()

# Task 2: Draw the margin around the lines you chose in Task 1. Use plot. fill between
#to create confidence intervals (margins refers to area of uncertainty in these lines) by adding
#and subtracting uncertainty term d from predicted values; 

xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
                     color='#AAAAAA', alpha=0.4)
    
plt.xlim(-1, 3.5)
plt.show()



# For SVM the line that maximises the margin is the optimal model

# Task 3: Use the sklearn package to build a support vector classifier using a linear kernel
# (hint: you will need from sklearn.svm import SVC). Plot the decision fuction on the data
from sklearn.svm import SVC

def plot_svc_decision_function(model, ax=None):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    X, Y = np.meshgrid(x,y)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    

    ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

model = SVC(kernel = "linear", C = 1E10, gamma = 0.1)
m1 = model.fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(m1)
plt.show()

# Task 4: Change the number of points in the dataset using X = X[:N] and y = y[:N]
# and build the classifier again using a linear kernel
# Plot the decision function. Do you see any differences?
X = X[:20,]
y = y[:20,]
m2 = model.fit(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(m2)
plt.show()


## So far we have considered linear boundaries but this is not always the case

## Consider the new dataset
    
from sklearn.datasets.samples_generator import make_circles
X2, y2 = make_circles(100, factor=.1, noise=.1)

np.shape(X2)
plt.scatter(X2[:,0], X2[:,1], c=y2, cmap = 'autumn')

#Task 5: Build a classifier using a linear kernel and plot the decision making function
m3 = SVC(kernel = "linear", gamma = 0.1).fit(X2, y2)
plt.scatter(X2[:,0], X2[:,1], c=y2, s=50)
plot_svc_decision_function(m3)
plt.show()

# These results should look wrong so we will try something else

# Consider projecting our data into a 3D plane
r = np.exp(-(X2 ** 2).sum(1))

from mpl_toolkits import mplot3d

ax = plt.subplot(projection='3d')
ax.scatter3D(X2[:, 0], X2[:, 1], r, c=y2, s=50, cmap='autumn')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('r')

plt.show()

# Looking at the data it is now clear to see that we could draw a linear plane through
# it in the 3D space and classify the data. We can then project back to the 2D
# space. This is what the 'rbf' kernel does. rbf is gaussian. 

#Task 6: Try building a classifier using the 'rbf' kernel
m4 = SVC(kernel = "rbf", gamma = 0.2).fit(X2, y2)
plt.scatter(X2[:,0], X2[:,1], c=y2, s=50)
plot_svc_decision_function(m4)
plt.show()
#The dotted line is the confidence intervals/margin but remember in 3D!

# Task 7: Go back to your original dataset (ie. make blobs) and try using different kernels 
# to build the classifier and plot the results
# Compare the differences between the models
X, y = make_blobs(n_samples=50, centers=2,
                  random_state=0, cluster_std=0.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

m5 = SVC(kernel = "poly", gamma = 0.5, C = 1).fit(X,y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(m5)
plt.show()

m6 = SVC(kernel = "rbf", gamma = 0.5, C = 1).fit(X,y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(m6)
plt.show()

## So far we have looked at clearly delineated data. Consider the following dataset
## where the margins are less clear

X3, y3 = make_circles(n_samples=100, factor=0.2, noise = 0.35)
plt.scatter(X3[:, 0], X3[:, 1], c=y3, s=50, cmap='autumn')
plt.show()

## SVM has a tuning parameter C which softerns the margins. For very large C, 
## the margin is hard, and points cannot lie in it. For smaller $C$, the margin 
# is softer, and can grow to encompass some points.

# Task 8: Try experimenting with different values of C and see what different
# results you get
m7 = SVC(kernel = "rbf", gamma = 0.5, C = 100000).fit(X3, y3)
plt.scatter(X3[:, 0], X3[:, 1], c=y3, s=50, cmap='autumn')
plot_svc_decision_function(m7)
plt.show() #Creates random islands - overfitting

m8 = SVC(kernel = "rbf", gamma = 0.5, C = 0.1).fit(X3, y3)
plt.scatter(X3[:, 0], X3[:, 1], c=y3, s=50, cmap='autumn')
plot_svc_decision_function(m7)
plt.show() #small C - underfitting

    
# Task 9: Use GridSearchCV (hint: from sklearn.model_selection import GridSearchCV)
# to find the optimum parameters for C. 
from sklearn.model_selection import GridSearchCV
?GridSearchCV

param_grid = {'C': [0.1, 1, 5, 10, 100], 'gamma': [0.01, 0.1, 0.3, 0.5]} #so pick any value from one of these ones, 
# The {} is a dictionary - a way to look up values for a term
#Always optimise multiple parameters at once

model = SVC(kernel = 'rbf', C = 1, gamma = 0.1).fit(X3, y3)

grid = GridSearchCV(model, param_grid)

grid.fit(X3, y3)
print(grid.best_params_)

model = grid.best_estimator_

plt.scatter(X3[:,0], X3[:,1], c = y3, s = 50, cmap = 'autumn')
plot_svc_decision_function(model)
plt.show()