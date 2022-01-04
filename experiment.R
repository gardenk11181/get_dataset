conda_path = "C:\\Users\\Jeongwon\\anaconda3\\envs\\ml"
Sys.setenv(RETICULATE_PYTHON = conda_path)
library(reticulate)
library(randomForest)
library(dplyr)
use_virtualenv(conda_path)
pickle5 = import('pickle5')
source_python('experiment.py')

train = 1:30162

data = preprocess(1)

z_train = data[[1]][[1]]
z_test = data[[1]][[2]]
s_train = data.frame(s = as.factor(data[[2]][[1]]$to_numpy()))
s_test = data.frame(s = as.factor(data[[2]][[2]]$to_numpy()))
y_train = data.frame(y = as.factor(data[[3]][[1]]$to_numpy()))
y_test = data.frame(y = as.factor(data[[3]][[2]]$to_numpy()))

x = data.frame(rbind(py_to_r(z_train[[1]]), py_to_r(z_test[[1]])))
z_vae = rbind(z_train[[2]], z_test[[2]])
z_vfae = rbind(z_train[[3]], z_test[[3]])

s = rbind(s_train, s_test)
y = rbind(y_train, y_test)

df = data.frame(x,s)
rf <- randomForest(s ~ ., data = df, subset = train, mtry = 6, importance = TRUE, ntree = 10)
rf_s_hat = predict(rf, newdata = df[-train,])
sum(rf_s_hat == s[-train,])/length(rf_s_hat)

log = glm(s ~ ., family = binomial, data = df, subset = train)
log_s_hat = predict(log, newdata = df[-train,], type = 'response')
log_s_predict = rep(0,15060)
log_s_hat[log_s_hat>0.5] = 1
sum(log_s_hat == s[-train,])/length(log_s_hat)
