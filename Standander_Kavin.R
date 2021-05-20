#clear the r environment
rm(list=ls())

library(rpart)
library(ggplot2)
library(ROSE)
library(caret)
library(randomForest)
library(ggplot2)
library(glmnet)

#setting the working directory
setwd("D:/Kavin/Edwisor/Dataset/Santander")

#verifying the directory
getwd()

#loading the train and test dataset
Santander_train = read.csv("train.csv", header = T , na.strings = c(" ", "", "NA"))
Santander_test = read.csv("test.csv", header = T , na.strings = c(" ", "", "NA"))

##Copy the original data set to new set
df_train=Santander_train
df_test=Santander_test

#Data Exploration
summary(df_train)
summary(df_test)



#########################################Missing value analysis#########################################
#Finding the missing values in train dataset
missing_val_Train = data.frame(missing_val_Train=apply(df_train,2,function(x){sum(is.na(x))}))
missing_val_Train = sum(missing_val_Train)
missing_val_Train

#Finding the missing values in test dataset
missing_val_Test = data.frame(missing_val_Test=apply(df_test,2,function(x){sum(is.na(x))}))
missing_val_Test = sum(missing_val_Test)
missing_val_Test


#########################################Outlier Analysis#########################################
##Outlier Analysis

numeric_index = sapply(df_train,is.numeric) #selecting only numeric variables
numeric_data = df_train[,numeric_index]
#numeric_data
cnames = colnames(numeric_data[2:201])
#Santander_train[,numeric_index[3:202]]
#View(numeric_data[2:201])
#View(cnames)

##Box plot to see the  outliers for var_0

ggplot(aes_string(y = (cnames[1])), data = subset(df_train))+
  stat_boxplot(geom = "errorbar", width = 0.5) +
  geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
               outlier.size=1, notch=FALSE)


#Removing Outliers data for Training dataset
for(i in cnames){
  #print(i)
  val = df_train[,i][df_train[,i] %in% boxplot.stats(df_train[,i])$out]
  #print(length(val))
  df_train = df_train[which(!df_train[,i] %in% val),]
}  

#Removing Outliers data for Testing dataset
for(i in cnames){
  #print(i)
  val = df_test[,i][df_test[,i] %in% boxplot.stats(df_test[,i])$out]
  #print(length(val))
  df_test = df_test[which(!df_test[,i] %in% val),]
}

#imputation of the outliers we are using the capping function for Training dataset
x = as.data.frame(df_train[cnames])
caps = data.frame(apply(df_train[cnames],2, function(x){
  quantiles <- quantile(x, c(0.25, 0.75))
  x[x < quantiles[1]] <- quantiles[1]
  x[x > quantiles [2]] <- quantiles[2]
}))


#imputation of the outliers we are using the capping function for Test dataset
x = as.data.frame(df_test[cnames])
caps = data.frame(apply(df_test[cnames],2, function(x){
  quantiles <- quantile(x, c(0.25, 0.75))
  x[x < quantiles[1]] <- quantiles[1]
  x[x > quantiles [2]] <- quantiles[2]
}))

dim(df_train)  

#########################################Standardisation of Dataset#####################################
# #Standardisation For Train dataset
for(i in cnames){
  #print(i)
  df_train[,i] = (df_train[,i] - mean(df_train[,i]))/
    sd(df_train[,i])
}

##Standardisation For Test dataset
for(i in cnames){
  #print(i)
  df_test[,i] = (df_test[,i] - mean(df_test[,i]))/
    sd(df_test[,i])
}

#########################################Feature Selection#########################################
#Correlations in Train dataset
Train_dataset_correlations=cor(df_train[,c(3:202)])
#View(Train_dataset_correlations)

#Correlations in test dataset
Test_Dataset_correlations=cor(df_test[,c(2:201)])
#View(Test_Dataset_correlations)

#########################################Handling Imbalanced Dataset##################################
#Target class count

require(gridExtra)
table(df_train$target)

#Bar of count of target classes

ggplot(df_train,aes(target))+geom_bar(stat='count',fill='green')

#Performing Under Sampling
data_balanced_under=ovun.sample(target ~ ., data = df_train, method = "under", N = 34206, seed = 1)$data

table(data_balanced_under$target)

#Bar of count of target classes after Under Sampling

ggplot(data_balanced_under,aes(target))+geom_bar(stat='count',fill='green')

#Divide data into train and test using stratified sampling method
set.seed(1234)

train.index = createDataPartition(data_balanced_under$target, p = .80, list = FALSE)
train = data_balanced_under[ train.index,2:202]
test  = data_balanced_under[-train.index,2:202]

#View(train)
#########################################Logistic regression Model##################################
##Logistic regression Model
logit_model = glm(target ~ ., data = train, family = "binomial")
summary(logit_model)
logit_Predictions = predict(logit_model, newdata = test, type = "response")


#convert prob
logit_Predictions = ifelse(logit_Predictions > 0.5, 1, 0)

target=test$target
target=as.factor(target)


logit_Predictions=as.factor(logit_Predictions)
print("Performance of Logistic Regression :")
confusionMatrix(data=logit_Predictions,reference=target)
#False Negative rate
#FNR = FN/FN+TP 


################################Cross validation prediction in Logistic regression#################################
#Training dataset
X_train=as.matrix(train[,-c(1,2)])
y_train=as.matrix(train$target)
#Test dataset
X_test=as.matrix(test[,-c(1,2)])
y_test=as.matrix(test$target)

#View(test[,-c(1,2)])



#Cross validation prediction inLogistic regression
set.seed(8909)
Crossvalidation_logitRegr=cv.glmnet(X_train,y_train,family = "binomial", type.measure = "class")
#Crossvalidation_logitRegr


#Model performance on validation dataset
set.seed(5363)
Crossvalidation_logitRegr_Prediction=predict(Crossvalidation_logitRegr,X_test,s = "lambda.min", type = "class")
#Crossvalidation_logitRegr_Prediction

#Confusion matrix
set.seed(689)
#actual target variable
target=test$target
#convert to factor
target=as.factor(target)
#predicted target variable
#convert to factor
Crossvalidation_logitRegr_Prediction=as.factor(Crossvalidation_logitRegr_Prediction)
print("Performance of Cross Validation Logistic Regression:")
confusionMatrix(data=Crossvalidation_logitRegr_Prediction,reference=target)


#########################################Random Forest Model##################################
##Random Forest Model

#View(test[,-c(1,2)])

train$target = as.factor(train$target)
RF_model = randomForest(target ~ ., train, importance = TRUE, ntree = 500)


RF_Predictions = predict(RF_model, test[,-1])#
ConfMatrix_RF = table(test$target, RF_Predictions)
print("Performance of Random Forest Model:  ")
confusionMatrix(ConfMatrix_RF)


#########################################Test Dataset Predictions##################################
##Test Data set Predictions

#View(df_test[,-c(1)])
#Picking Independent Variables from Test Dataset
test_data=(df_test[,-c(1)])


##Applying Logicistic Regression on Test data set
logit_model_testData=predict(logit_model,test_data)
logit_model_testData=ifelse(logit_model_testData>0.5,1,0)
print('Count of Target variable predicted from  Test Dataset')
table(logit_model_testData)

##Applying Random Forest Model on Test data set
RF_model_testData=predict(RF_model,test_data)
table(RF_model_testData)


#######################################Final submission on Test Dataset##################################
#Final submission

Final_Data=data.frame(ID_code=df_test$ID_code,
                      logit_model_Prediction=logit_model_testData
                      ,Random_Forest_Prediction=RF_model_testData
                      
)
View(Final_Data)
write.csv(Final_Data, "Prediction of Test Dataset.csv", row.names = F)