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
                  random_state=0, cluster_std=0.50)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

plt.show()

# Task 1: Attempt to use linear regression to separate this data using linear regression.
# Note there are several possibilities which separate the data? 
# What happens to the classification of point [0.6, 2.1] (or similar)?




# With SVM rather than simply drawing a zero-width line between the 
# classes, we draw a margin of some width around each line, up to the nearest point. 
# For example for these lines:

""" 
xfit = np.linspace(-1, 3.5)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')

for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none',
                     color='#AAAAAA', alpha=0.4)

plt.xlim(-1, 3.5)
plt.show()
"""
# Task 2: Draw the margin around the lines you chose in Task 1.




# For SVM the line that maximises the margin is the optimal model

# Task 3: Use the sklearn package to build a support vector classifier using a linear kernel
# (hint: you will need from sklearn.svm import SVC). Plot the decision fuction on the data



# Task 4: Change the number of points in the dataset using X = X[:N] and y = y[:N]
# and build the classifier again using a linear kernel
# Plot the decision function. Do you see any differences?



## So far we have considered linear boundaries but this is not always the case

## Consider the new dataset
    
from sklearn.datasets.samples_generator import make_circles
X2, y2 = make_circles(100, factor=.1, noise=.1)

#Task 5: Build a classifier using a linear kernel and plot the decision making function


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
# space. This is what the 'rbf' kernel does.

#Task 6: Try building a classifier using the 'rbf' kernel


# Task 7: Go back to your original dataset (ie. make blobs) and try using different kernels 
# to build the classifier and plot the results
# Compare the differences between the models



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


    
# Task 9: Use GridSearchCV (hint: from sklearn.model_selection import GridSearchCV)
# to find the optimum parameters for C. 

