# Practical Machine Learning Course Project
Arthur F Heezen III  
February 28, 2016  

### Executive Summary
The challenge was to predict the type of human activity given a large number of features derived from accelerometers.  A random forest model predicts the type of activity with very high accuracy.  Using ten-fold cross validation the accuracy was estimated to be 0.9952 with a standard deviation of 0.0013.  On the 20 item test dataset the prediction model classified 100% of the cases correctly.

### Background and Data Summary
Data was collected by researchers studying Human Activity Recognition with a focus on the quality of exercise [1].  Six subjects performed exercises in five differnt ways, labeled in the variable `classe` as A, B, C, D, and E.

After exploring the supplied training and testing data sets, it was clear that a significant number of the 160 variables supplied had no predictive value because they were invariant in the test data set.  Many of these variables also had a very large number of missing values and blanks.  After removing these variables and log-type information (like timestamps) that should not be used for prediction, 53 predictors were left for modeling.

### Model Description and Analysis

A random forest model was trained in R using the caret package.  Parallel execution made runtime acceptable. 

Cross validation (10 folds) enabled an estimate of the out-of-sample accuracy as 0.9952 with a standard deviation of 0.0013.  Accuracy of 100% was observed on the 20 item test set.  Based on this, no further improvements to the prediction model were sought.

A summary of the model fit from R follows:

```r
fit
```

```
## Random Forest 
## 
## 19622 samples
##    53 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 17660, 17658, 17660, 17661, 17660, 17660, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##    2    0.9945471  0.9931021  0.001781869  0.002254250
##   29    0.9952606  0.9940047  0.001338099  0.001692595
##   57    0.9902152  0.9876223  0.002640528  0.003340490
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 29.
```

The top 20 variables in the random forest model are: 

```r
library(caret)
varImp(fit)
```

```
## rf variable importance
## 
##   only 20 most important variables shown (out of 57)
## 
##                      Overall
## roll_belt             100.00
## pitch_forearm          60.54
## yaw_belt               55.82
## pitch_belt             45.54
## magnet_dumbbell_z      44.53
## magnet_dumbbell_y      43.71
## roll_forearm           43.63
## accel_dumbbell_y       23.16
## roll_dumbbell          18.86
## magnet_dumbbell_x      18.21
## accel_forearm_x        17.94
## magnet_belt_z          17.18
## accel_dumbbell_z       14.61
## total_accel_dumbbell   14.52
## magnet_forearm_z       14.12
## magnet_belt_y          13.93
## accel_belt_z           13.29
## yaw_arm                12.51
## gyros_belt_z           12.32
## magnet_belt_x          10.98
```

A confusion matrix demonstrates the excellent fit of the model in a different way.

```r
library(caret)
confusionMatrix.train(fit)
```

```
## Cross-Validated (10 fold) Confusion Matrix 
## 
## (entries are percentages of table totals)
##  
##           Reference
## Prediction    A    B    C    D    E
##          A 28.4  0.1  0.0  0.0  0.0
##          B  0.0 19.2  0.1  0.0  0.0
##          C  0.0  0.0 17.3  0.1  0.0
##          D  0.0  0.0  0.0 16.2  0.0
##          E  0.0  0.0  0.0  0.0 18.3
```

### References
[1] Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. [Qualitative Activity Recognition of Weight Lifting Exercises](http://groupware.les.inf.puc-rio.br/work.jsf?p1=11201). Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13) . Stuttgart, Germany: ACM SIGCHI, 2013.
