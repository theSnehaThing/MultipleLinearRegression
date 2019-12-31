#MULTIPLE LINEAR REGRESSION
dataset = read.csv('50_Startups.csv')
dataset$State = factor(dataset$State, levels =c('New York', 'California', 'Florida'),
                         labels = c(1, 2 ,3))
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
train_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

#Fitting and predicting(Simple linear regression with all variables)
regressor = lm(formula = Profit ~ .,
               data = train_set)
y_pred = predict(regressor, newdata = test_set)

#Fitting and predicting(Multiple linear regression with 1 variable)
regressor_multi = lm(formula = Profit ~ R.D.Spend,
               data = train_set)
y_pred_multi = predict(regressor, newdata = test_set)

#Backtracking
regressor_backtrack = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State,
                     data = train_set)
summary(regressor_backtrack)

backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(train_set, SL)



#Visualising