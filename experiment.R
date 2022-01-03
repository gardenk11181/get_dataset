install.packages('reticulate')
library(reticulate)
library(randomForest)
library(dplyr)
use_condaenv("r-reticulate")
pickle5 = import('pickle5')
conda_install('r-reticulate', 'matplotlib')
source_python('experiment.py')

# x
x_train = read_pickle('./adult/adult_train.pkl')
x_test = read_pickle('./adult/adult_test.pkl')
x = rbind(x_train, x_test)
x = data.frame(apply(x, 2, function(x) as.factor(as.integer(x))))
x = x %>% mutate_if(is.character,as.factor)
y = x[, 104]
s = x[, 103]
x = x[, -c(103,104)]

train = 1:30162


data = experiment(1)
z_train = data.frame(data[[1]][[1]][[3]])
z_test = data.frame(data[[1]][[2]][[3]])
z = rbind(z_train, z_test)

df = data.frame(x,s)
rf <- randomForest(s ~ ., data = df, subset = train, importance = TRUE, ntree = 10)
s_hat = predict(rf, newdata = df[-train,])
sum(s_hat == s[-train])/length(y_hat)
