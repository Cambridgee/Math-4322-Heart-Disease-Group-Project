library(tree)
library(randomForest)

# Import Heart.csv
Heart <- read.csv("Heart.csv")

# Converts non-numerical variables
Heart$sex <- as.factor(Heart$sex)
Heart$exang <- as.factor(Heart$exang)
Heart$fbs <- as.factor(Heart$fbs)
Heart$target <- as.factor(Heart$target)

# Number of rows in the Heart Data
n <- nrow(Heart)
# Number of predictors in the Heart Data
p <- ncol(Heart) - 1
B <- 100

# initial model before changes
set.seed(1)

# Splits the data into 80% training and 20% testing
sample <- sample.int(n = nrow(Heart), size = round(0.8 * nrow(Heart)), replace = FALSE)
train <- Heart[sample, ]
test <- Heart[-sample, ]

# Logistic model
log.mod <- glm(target ~ ., family = "binomial", data = train)
summary(log.mod)
step.mod <- step(log.mod)
summary(step.mod)

# Random Forest model
rf.model <- randomForest(target ~ ., data = train,
                         ntree = B, mtry = sqrt(p), importance = TRUE)

# Output
print("Logistic Regression Model:")
print(summary(log.mod))
print("Random Forest Model:")
print(rf.model)
par(mar = c(5, 4, 4, 2) + 0.1)
varImpPlot(rf.model)

# Set for error on the Logistic Model
set <- 1:10
log.test.err <- numeric(length(set))
rf.test.error <- numeric(length(set))
log.sensitivity <- numeric(length(set))
rf.sensitivity <- numeric(length(set))
log.specificity <- numeric(length(set))
rf.specificity <- numeric(length(set))

for (i in 1:10) {
  # Sets the seed for 10 different subdivisions of data
  set.seed(i)
  # Splits the data into 80% training and 20% testing
  sample <- sample.int(n = nrow(Heart), size = round(0.8 * nrow(Heart)), replace = FALSE)
  train <- Heart[sample, ]
  test <- Heart[-sample, ]
  
  # Logistic model
  log.mod <- glm(target ~ restecg + chol + exang + trestbps + thal + ca + sex + oldpeak + cp,
                 family = "binomial", data = train)
  
  # Predicts heart disease on the testing data
  log.predict <- predict(log.mod, type = "response", newdata = test)
  log.predict.target <- ifelse(log.predict > 0.5, "1", "0")
  
  # Confusion table
  log.conf <- table(log.predict.target, test$target)
  
  # Calculates the test error rate, sensitivity, and specificity
  log.test.err[i] <- (log.conf[1, 2] + log.conf[2, 1]) / sum(log.conf)
  log.specificity[i] <- log.conf[1,1] / (log.conf[1, 1] + log.conf[2, 1])
  log.sensitivity[i] <- log.conf[2,2] / (log.conf[1, 2] + log.conf[2, 2])
  
  target.test <- test$target
  X.test <- subset(test, select = -target)
  
  # Random Forest model
  rf.model <- randomForest(target ~ ., data = train,
                           ntree = B, mtry = sqrt(p), importance = TRUE)
  
  # Predicts heart disease on the testing data
  rf.predict <- predict(rf.model, newdata = test)
  
  # Confusion table
  rf.conf <- table(rf.predict, test$target)
  
  # Calculates the test error rate, sensitivity, and specificity
  rf.test.error[i] <- (rf.conf[1, 2] + rf.conf[2, 1]) / sum(rf.conf)
  rf.specificity[i] <- rf.conf[1,1] / (rf.conf[1, 1] + rf.conf[2, 1])
  rf.sensitivity[i] <- rf.conf[2,2] / (rf.conf[1, 2] + rf.conf[2, 2])
  
  # Outputs the first subdivision results
  if (i == 1){
    print("Logistic Regression Model:")
    print(summary(log.mod))
    print("Logistic Model Confusion Table:")
    print(log.conf)
    print("Random Forest Model:")
    print(rf.model)
    par(mar = c(5, 4, 4, 2) + 0.1)
    varImpPlot(rf.model)
  }
}

print("Test Error Model Summaries:")

print("Logistic Model")
print("Test Error:")
mean(log.test.err)
print("Sensitivity:")
mean(log.sensitivity)
print("Specificity:")
mean(log.specificity)

print("Random Forest Model")
print("Test Error:")
mean(rf.test.error)
print("Sensitivity:")
mean(rf.sensitivity)
print("Specificity:")
mean(rf.specificity)
