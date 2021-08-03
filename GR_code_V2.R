
library(plotmo)
library(VIM)
library(tidyverse)
library(corrplot)
library(missForest)
library(moments)
library(glmnet)


# Import the data from a url
theUrl<-"http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
adult.data<- read.table(file = theUrl, header = FALSE, sep = ",", 
                        strip.white = TRUE, stringsAsFactors = TRUE,
                        col.names=c("age","workclass","fnlwgt","education","educationnum","maritalstatus","occupation","relationship","race","sex","capitalgain","capitalloss",                     "hoursperweek","nativecountry","income")
)

# observation 32561 variables 15
dim (adult.data)

# Data preprocessing (collapse the factor levels & re-coding)

levels(adult.data$workclass)<- c("misLevel","FedGov","LocGov","NeverWorked","Private","SelfEmpNotInc","SelfEmpInc","StateGov","NoPay")

levels(adult.data$education)<- list(presch=c("Preschool"), primary=c("1st-4th","5th-6th"),upperprim=c("7th-8th"), highsch=c("9th","Assoc-acdm","Assoc-voc","10th"),secndrysch=c("11th","12th"), graduate=c("Bachelors","Some-college"),master=c("Masters"), phd=c("Doctorate"))

levels(adult.data$maritalstatus)<- list(divorce=c("Divorced","Separated"),married=c("Married-AF-    spouse","Married-civ-spouse","Married-spouse-absent"),notmarried=c("Never-married"),widowed=c("Widowed"))

levels(adult.data$occupation)<- list(misLevel=c("?"), clerical=c("Adm-clerical"), lowskillabr=c("Craft-repair","Handlers-cleaners","Machine-op-inspct","Other-service","Priv-house-    serv","Prof-specialty","Protective-serv"),highskillabr=c("Sales","Tech-support","Transport-moving","Armed-Forces"),agricultr=c("Farming-fishing"))

levels(adult.data$relationship)<- list(husband=c("Husband"), wife=c("Wife"), outofamily=c("Not-in-family"),unmarried=c("Unmarried"), relative=c("Other-relative"), ownchild=c("Own-child"))

levels(adult.data$nativecountry)<- list(misLevel=c("?","South"),SEAsia=c("Vietnam","Laos","Cambodia","Thailand"),Asia=c("China","India","HongKong","Iran","Philippines","Taiwan"),NorthAmerica=c("Canada","Cuba","Dominican-Republic","Guatemala","Haiti","Honduras","Jamaica","Mexico","Nicaragua","Puerto-Rico","El-Salvador","United-States"), SouthAmerica=c("Ecuador","Peru","Columbia","Trinadad&Tobago"),Europe=c("France","Germany","Greece","Holand-Netherlands","Italy","Hungary","Ireland","Poland","Portugal","Scotland","England","Yugoslavia"),PacificIslands=c("Japan","France"),Oceania=c("Outlying-US(Guam-USVI-etc)"))


# Missing data visualization

aggr_plot <- aggr(adult.data, col=c('navyblue','red'), numbers=TRUE, sortVars=TRUE, 
                  labels=names(adult.data), cex.axis=.7, gap=3, 
                  ylab=c("Histogram of missing data","Pattern"))

# Missing data treatment
# library(missForest)
imputdata<- missForest(adult.data) 
# check imputed values
imputdata$ximp
# assign imputed values to a data frame
adult.cmplt<- imputdata$ximp


# Some obvious relationships ,, Box plot AGe and Income
boxplot (age ~ income, data = adult.cmplt, 
         main = "Age distribution for different income levels",
         xlab = "Income Levels", ylab = "Age", col = "salmon")

# Boxplot for hours per week in office and income
boxplot (hoursperweek ~ income, data = adult.cmplt, 
         main = "More work hours, more income",
         xlab = "Income Levels", ylab = "Hours per week", col = "salmon")

# Some not-so-obvious relationships
# Q-plot for occupation and income, 
#Question: Does higher skill-set (sales, technical-support, transport movers, armed forces) is a guarantor to high income?

qplot(income, data = adult.cmplt, fill = occupation) + facet_grid (. ~ occupation)

# Does higher education help earn more money? plot for education and income
qplot(income, data = adult.cmplt, fill = education) + facet_grid (. ~ education)

# plot for race, relationship and income
qplot(income, data = adult.cmplt, fill = relationship) + facet_grid (. ~ race)


# plot for workclass and maritalstatus
qplot(income, data = adult.cmplt, fill = workclass) + facet_grid(. ~maritalstatus)

# plot for Relationship and Occupation
qplot(income, data = adult.cmplt, fill = occupation) + facet_grid(. ~relationship)

# plot for married, education and income
qplot(income, data = adult.cmplt, fill = maritalstatus) + facet_grid(. ~education)

# Detecting skewed variables, highly skewed if its absolute value is greater than 1. A. 
# moderately skewed if its absolute value is greater than 0.5.
skewedVars<- NA
# library(moments) # for skewness()
for(i in names(adult.cmplt)){
  if(is.numeric(adult.cmplt[,i])){
    if(i != "income"){
      # Enters this block if variable is non-categorical
      skewVal <- skewness(adult.cmplt[,i])
      print(paste(i, skewVal, sep = ": "))
      if(abs(skewVal) > 0.5){
        skewedVars <- c(skewedVars, i)
      }
    }
  }
}

# Skewed variable treatment 
adult.cmplt<- adult.cmplt[c(3,11:12,1,5,13,2,4,6:10,14:15)]
str(adult.cmplt)
# Post skewed treatment
adult.cmplt.norm<- adult.cmplt
adult.cmplt.norm[,1:3]<- log(adult.cmplt[1:3],2) # where 2 is log base 2
adult.cmplt.norm$capitalgain<- NULL
adult.cmplt.norm$capitalloss<-NULL

# Correlation detection 

correlat<- cor(adult.cmplt.norm[c(1:4)])
corrplot(correlat, method = "pie")
highlyCor <- colnames(adult.cmplt.num)[findCorrelation(correlat, cutoff = 0.7, verbose = TRUE)]
# All correlations <= 0.7 
highlyCor # No high Correlations found
character(0)

#its evident that none of the predictors are highly correlated to each other. We now proceed to building the prediction model.
# Predictive data analytics
# In this section, we will discuss various approaches applied to model building, predictive power and their trade-offs.

# Creating the train and test dataset
ratio = sample(1:nrow(adult.cmplt), size = 0.25*nrow(adult.cmplt))
test.data = adult.cmplt[ratio,] #Test dataset 25% of total
train.data = adult.cmplt[-ratio,] #Train dataset 75% of total
dim(train.data)   #  [1] 24421    15
dim(test.data)    # [1] 8140   15

# We fit a logistic regression model.
glm.fit<- glm(income~., family=binomial(link='logit'),data = train.data)
# This Warning: glm.fit: fitted probabilities numerically 0 or 1 occurred 
#means that the data is possibly linearly separable. Let's look at the summary for the model 
summary(glm.fit)
Call:
  glm(formula = income ~ ., family = binomial(link = "logit"), 
      data = train.data)

# Deviance Residuals: 
#   Min       1Q   Median       3Q      Max  
# -5.1559  -0.4717  -0.1783  -0.0318   3.5043  
# Coefficients: (1 not defined because of singularities)
# Estimate Std. Error z value Pr(>|z|)    
# (Intercept)                 -2.717e+01  1.642e+02  -0.165 0.868555
# Its evident that the significant predictors are age, workclassSelfEmpInc,fnlwgt,educationnum and maritalstatusmarried

# We now create another logistic model that includes only the significant predictors, AIC: 18540
glm.fit1<- glm(income ~ age + workclass + educationnum + fnlwgt + maritalstatus, family=binomial(link='logit'),data = train.data)
# Now we can run the anova() function on the improved model to analyze the table of deviance. comparing model 1 and model 2
# Model 2 is statistically significant as the p value is less than 0.05
#age + workclass + educationnum + fnlwgt + maritalstatus) are relevant for the model. Anova to analyze the defiance
anova(glm.fit, glm.fit1, test="Chisq") 

# Backward selection on model1, gl.fit Residual Deviance: 15080 	AIC: 15170, we take the lowest AIC
step(glm.fit, trace = F, scope = list(lower=formula(glm.fit), upper=formula(glm.fit)),
     direction = 'backward')
# Logistic regression model using selected variables by team, AIC= 18500
glm.fit3<- glm(income ~ age + workclass + educationnum + race + nativecountry + maritalstatus, family=binomial(link='logit'),data = train.data)
AIC(glm.fit3)

#LM using updated variables
glm.fit4<- glm(income~ educationnum + race + maritalstatus + age,family=binomial(link='logit'),data = train.data)
AIC(glm.fit4) #18638.7

glm.fit5<- glm(income~ educationnum + race,family=binomial(link='logit'),data = train.data)
AIC(glm.fit5) #23794.94

glm.fit6<- glm(income~ educationnum + age + race + maritalstatus +relationship ,family=binomial(link='logit'),data = train.data)
AIC(glm.fit6) #18358.02



# Accuracy also reduced when using variables selected by team
set.seed(1234)
glm.pred<- predict(glm.fit3, test.data, type = "response")
hist(glm.pred, breaks=20)
hist(glm.pred[test.data$income], col="red", breaks=20, add=TRUE)
table(actual= test.data$income, predicted= glm.pred>0.5)
predicted



(5704+1006)/8140   # 8140 test.data , 6974/8140 , accuracy = 70..i.e 70% Accuracy. model includes variables chosen by team.

# We now test the logistic model on all predictors and make predictions on unseen data
# But when using all variables Accuracy increases. Note: Logistic regression is good when using unseen data
set.seed(1234)
glm.pred<- predict(glm.fit, test.data, type = "response")
hist(glm.pred, breaks=20)
hist(glm.pred[test.data$income], col="red", breaks=20, add=TRUE)
table(actual= test.data$income, predicted= glm.pred>0.5)
predicted


(5680+1294)/8140   # 8140 test.data , 6974/8140 , accuracy = 85.7..i.e 86% Accuracy. model includes all predictors in it











###  My code   LASSO selection for a logistic reg model.

## website ::: https://zitaoshen.rbind.io/project/machine_learning/how-to-plot-roc-curve-for-multiple-classes/


## using the above test and train sets

library(plotmo)

x_train <-  model.matrix(income~.-1,train.data)
lasso.glm.fit = cv.glmnet(x=x_train,y = as.factor(train.data$income), intercept=FALSE ,family =   "binomial", alpha=1, nfolds=7)
best_lambda <- lasso.glm.fit$lambda[which.min(lasso.glm.fit$cvm)]

best_lambda


plot(lasso.glm.fit)















