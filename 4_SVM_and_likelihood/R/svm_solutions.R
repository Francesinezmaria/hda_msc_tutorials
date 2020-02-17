library(ggplot2)

# Consider two classes of points which are well separated

# Reading in dataset (you will have to change the working directory path)
setwd("~/Documents/PhD/teaching/Machine_learning_2020/hda_msc_tutorials_private/4_SVM_and_likelihood/R/")
df <- read.csv(file = 'make_blobs_1.csv')
df_ext <- read.csv(file = 'make_blobs_2.csv')
df2 <- read.csv(file = 'make_circles_1.csv')
df3 <- read.csv(file = 'make_circles_2.csv')

plot(df$X_x, df$X_y, col = df$labels+3)

# Task 1: Attempt to use linear regression to separate this data using linear regression.
# Note there are several possibilities which separate the data? 
# What happens to the classification of point [0.6, 2.1] (or similar)?

xfit <- seq(-1, 3.5, length.out=50)
plot(df$X_x, df$X_y, col = df$labels+3)

for (i in list(c(1, 0.65), c(0.5, 1.6), c(-0.2, 2.9))){
  lines(xfit, i[1]*xfit + i[2], type = "l")
}


points(0.6, 2.1, col = "red")

# With SVM rather than simply drawing a zero-width line between the 
# classes, we draw a margin of some width around each line, up to the nearest point. 
# For example for these lines:

# xfit <- seq(-1, 3.5, length.out=50)
# plot(df$X_x, df$X_y, col = df$labels+3)


# for (i in list(c(1, 0.65, 0.33), c(0.5, 1.6, 0.55), c(-0.2, 2.9, 0.2))){
#   yfit <- i[1]*xfit + i[2]
#   lines(xfit, yfit, type = "l")
#   lines(xfit, yfit+i[3], type="l", lty = 3, col = 'orange')
#   lines(xfit, yfit-i[3], type="l", lty = 3, col = 'orange')
# }


# for (i in list(c(1, 0.65, 0.33), c(0.5, 1.6, 0.55), c(-0.2, 2.9, 0.2))){
#   yfit <- i[1]*xfit + i[2]
#   lines(xfit, yfit, type = "l")
#   lines(xfit, yfit+i[3], type="l", lty = 3, col = 'orange')
#   lines(xfit, yfit-i[3], type="l", lty = 3, col = 'orange')
# }

# for (i in list(c(1, 0.65, 0.33), c(0.5, 1.6, 0.55), c(-0.2, 2.9, 0.2))){
#   yfit <- i[1]*xfit + i[2]
#   lines(xfit, yfit, type = "l")
#   lines(xfit, yfit+i[3], type="l", lty = 3, col = 'orange')
#   lines(xfit, yfit-i[3], type="l", lty = 3, col = 'orange')
# }

# Task 2: Draw the margin around the lines you chose in Task 1.




# For SVM the line that maximises the margin is the optimal model

# Task 3: Use the e1071 package to build a support vector classifier using a linear kernel
# Plot the decision fuction on the data

library("e1071")

# Fit an SVM classifier
svm.model <- svm(labels~X_x+X_y, data=df, kernel="linear", cost=1E10, type="C-classification", scale=FALSE)

plot.svm.margin <- function(model, data, main="") {
  if (model$kernel==0) {
    # Linear kernel

    # Plot the data
    plot(data$X_x, data$X_y, col=data$labels+3, main=main)
    
    # Add the decision boundary and margins
    x1min <- min(data$X_x)
    x1max <- max(data$X_x)
    coef1 <- sum(model$coefs*data$X_x[model$index]);
    coef2 <- sum(model$coefs*data$X_y[model$index]);

    lines(c(x1min,x1max),  (model$rho-coef1*c(x1min, x1max))/coef2)
    lines(c(x1min,x1max),  (model$rho+1-coef1*c(x1min, x1max))/coef2, lty=2)
    lines(c(x1min,x1max),  (model$rho-1-coef1*c(x1min, x1max))/coef2, lty=2)
  }
  else if(model$kernel==2|1) {
    # RBF or polynomial kernel
    
    # Create grid
    xx <- seq(min(data$X_x), max(data$X_x), length.out=100)
    yy <- seq(min(data$X_y), max(data$X_y), length.out=100)
    xgrid <- expand.grid(X_x=xx, X_y=yy) #generating grid points
    ygrid <- predict(model, newdata=xgrid)
    
    # Evaluate model on grid
    func <- predict(model, xgrid, decision.values=TRUE)
    func <- attributes(func)$decision 
    
    # Plot
    plot(xgrid, col=as.numeric(ygrid)+3, cex=0.3)
    points(data$X_x, data$X_y, col=data$labels+3)
    contour(xx, yy, matrix(func, length(xx), length(yy)), level=0, add=TRUE, lwd=3)
  }
  else
    stop("Linear (0), Polynomial (1) or RBF (2) kernel only")
}

plot.svm.margin(svm.model, df)

# Task 4: Change the number of points in the dataset using X = X[1:N] and df$labels = df$labels[1:N]
# and build the classifier again using a linear kernel
# Plot the decision function. Do you see any differences?

svm.model.n.60 <- svm(labels~X_x+X_y, data=df_ext[1:60,], kernel="linear", cost=1E10, type="C-classification", scale=FALSE)
svm.model.n.120 <- svm(labels~X_x+X_y, data=df_ext[1:120,], kernel="linear", cost=1E10, type="C-classification", scale=FALSE)

# Plot them side-by-side
par(mfrow=c(1,2))
plot.svm.margin(svm.model.n.60, df_ext[1:60,], main="n=60")
plot.svm.margin(svm.model.n.120, df_ext[1:120,], main="n=120")


## So far we have considered linear boundaries but this is not always the case

## Consider the new dataset df2

dev.off() # Turn off plot grid
plot(df2$X_x, df2$X_y, col=as.numeric(df2$labels)+3)

#Task 5: Build a classifier using a linear kernel and plot the decision making function

svm.model <- svm(labels~X_x+X_y, data=df2, kernel="linear", cost=1E10, type="C-classification", scale=FALSE)
plot.svm.margin(svm.model, df2)


# These results should look wrong so we will try something else

# Consider projecting our data into a 3D plane

df2$r = exp(-(df2$X_x**2 + df2$X_y** 2))

library("scatterplot3d") # load

scatterplot3d(x=df2$X_x, y=df2$X_y, z=df2$r, color=df2$labels+3)

# Looking at the data it is now clear to see that we could draw a linear plane through
# it in the 3D space and classify the data. We can then project back to the 2D
# space. This is what the 'rbf' kernel does.

#Task 6: Try building a classifier using the 'rbf' kernel

svm.model <- svm(labels~X_x+X_y, data=df2, kernel="radial", cost=1E10, type="C-classification", scale=FALSE)
plot.svm.margin(svm.model, df2)
  

# Task 7: Go back to your original dataset (ie. make blobs_1) and try using different kernels 
# to build the classifier and plot the results
# Compare the differences between the models

svm.model_linear <- svm(labels~X_x+X_y, data=df, kernel="linear", cost=1E10, type="C-classification", scale=FALSE)
plot.svm.margin(svm.model_linear, df, main = "linear")

svm.model.poly <- svm(labels~X_x+X_y, data=df, kernel="polynomial", cost=1E10, type="C-classification", scale=FALSE)
plot.svm.margin(svm.model.poly, df)

svm.model.radial <- svm(labels~X_x+X_y, data=df, kernel="radial", cost=1E10, type="C-classification", scale=FALSE)
plot.svm.margin(svm.model.radial, df)

## So far we have looked at clearly delineated data. Consider the following dataset
## where the margins are less clear

plot(df3$X_x, df3$X_y, col = df3$labels+3)

## SVM has a tuning parameter C which softerns the margins. For very large C, 
## the margin is hard, and points cannot lie in it. For smaller $C$, the margin 
# is softer, and can grow to encompass some points.

# Task 8: Try experimenting with different values of C and see what different
# results you get

svm.model.radial_C_10 <- svm(labels~X_x+X_y, data=df3, kernel="radial", cost=10, type="C-classification", scale=FALSE)
svm.model.radial_C_01 <- svm(labels~X_x+X_y, data=df3, kernel="radial", cost=0.1, type="C-classification", scale=FALSE)

par(mfrow=c(1,2))
plot.svm.margin(svm.model.radial_C_10, df3)
plot.svm.margin(svm.model.radial_C_01, df3)

# Task 9: Use e1071::tune.svm from to find the optimum parameters for C. 

# 5-fold cross-validation on costs of 10^6 to 10^-3 on a log10 scale
df3$labels <- as.factor(df3$labels)
svm.crossval <- tune.svm(labels~., data=df3, cost=10^(6:-3), tunecontrol=tune.control(cross=5), kernel="radial", type="C-classification", scale=FALSE)

# Look at the error as C varies
summary(svm.crossval)
plot(svm.crossval)

# Get the best model
best.svm <- svm.crossval$best.model

