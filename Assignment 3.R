# Assignment 3 – Package Setup
# Machine Learning in Finance 2025
# University of Deusto

install.packages("CHAID", repos = "https://R-Forge.R-project.org")

# Required packages
library(caret)
library(pROC)
library(ggplot2)
library(rpart)
library(CHAID)
library(C50)
library(randomForest)
library(grid)

team_data <- readRDS("t_01.rds")

# Verify that the dataset loaded correctly
str(team_data)   # Check structure
head(team_data)  # Preview first rows

summary(team_data)  # Summary statistics

# Check the distribution of the target variable (y)
table(team_data$y)  # Count of each class
prop.table(table(team_data$y))  # Proportions

# Now we are ready to explore the data and build your model

cols <- sapply(team_data, is.numeric)
numerical_cols <- names(cols[cols == TRUE])
nonnumerical_cols<-names(cols[cols == FALSE]) #we obtain numeric and non numeric columns to realize more efficient analysis
numerical_cols


team_data$y2 <- team_data$y
team_data$y <- ifelse(team_data$y=="yes",1,0) #we factor it


#We group the variables
team_data$balance_group <- cut(team_data$balance, breaks = c(-Inf,0,1000,5000,10000, Inf),labels = c("Negative", "0-1000", "1000-5000", "5000-10000","10000+"))
team_data$age_group <- cut(team_data$age,
                           breaks = c(0,18 ,25, 35, 50, 65, Inf),
                           labels = c("0","18-25", "25-35", "35-50", "50-65", "65+"))
team_data$pdays_group <- cut(team_data$pdays, breaks = c(-Inf, -1,180,360, Inf), 
                             labels = c("No Contact", "last half year", "last year","more than a year"))
team_data$duration_group <- cut(team_data$duration,
                                breaks = c(0, 60, 120, 300,600,  Inf),
                                labels = c("very_short", "short", "medium", "long","very long"))
team_data$campaign_group <- cut(team_data$campaign,
                                breaks = c(0, 1, 3,10, Inf),
                                labels = c("1_contact", "2-3_contacts", "4-10_contacts", "10+_contacts"))
team_data$previous_group <- cut(team_data$previous,
                                breaks = c(-Inf, 0, 1, Inf),
                                labels = c("none", "one", "1+"))

# convert categorical variables to ordered factors
str(team_data) # already as factor


str(team_data$pdays)

# check missing values
colSums(is.na(team_data)) #no missing values


#Split the model between training and testing
smp_size <- round(0.8 * nrow(team_data))
set.seed(578)
train_ind <- sample(seq_len(nrow(team_data)), size = smp_size)
train <- team_data[train_ind, ]
test <- team_data[-train_ind, ]
summary(train)
summary(test)
str(train)

# Start training the models, starting from a model with nearly all variables
model1 <- glm(y ~ age+job+marital+education+default+balance+housing+loan+contact+duration+campaign+pdays_group+previous+poutcome, data = train, family = binomial)
summary(model1)
model2 <- glm(y ~ job+marital+housing+loan+contact+duration+campaign+pdays_group+poutcome, data = train, family = binomial)
summary(model2)
model3 <- glm(y ~ age+job+marital+education+default+housing+loan+contact+duration_group+campaign_group+pdays_group+previous_group+poutcome+balance_group, data = train, family = binomial)
summary(model3)



# Start making comparisons of variance between models as well as AIC values
AIC(model1,model2,model3)
anova(model1,model2,model3)

# Creation of the confusion matrix
library(caret)
train <- na.omit(train)
train$prob <- predict(model3, newdata = train, type = "response")
train$predicted_class <-ifelse(train$prob>=0.4083,1,0)
train$predicted_class <- ifelse(train$predicted_class ==1,"yes","no")
train$predicted_class <- factor(train$predicted_class, levels = c("yes", "no"))
train$y2<- factor(train$y2, levels = c("yes", "no"))

conf_matrix1 <- confusionMatrix(train$predicted_class,train$y2,positive="yes")
conf_matrix1 #nice accuracy and sensitivity, exchanged for specifity as we have lowered the threshold

# Here we plot the ROC curve
library(pROC)
roc_curve <- roc(train$y2,train$prob,levels = c("no","yes"),direction = "<")

library(ggplot2)
roc_train <- data.frame(
  threshold = roc_curve$thresholds,
  sensitivity = roc_curve$sensitivities,
  specificity = roc_curve$specificities
)

# Calculate the Youden Index
roc_train$youden_index <- roc_train$sensitivity + roc_train$specificity - 1

# Find the row with the maximum Youden Index
optimal_row <- roc_train[which.max(roc_train$youden_index), ]

# Extract the optimal threshold, specificity, and sensitivity
# Find the optimal point to create a more precise model
optimal_threshold_youden <- optimal_row$threshold
optimal_threshold_youden
optimal_specificity <- optimal_row$specificity
optimal_sensitivity <- optimal_row$sensitivity

# We plot de ROC curve
ggplot(roc_train, aes(x = specificity, y = sensitivity)) +
  geom_line(color = "blue", size = 1.2) +  # ROC Curve in Blue
  geom_segment(aes(x = 1, y = 0, xend = 0, yend = 1), linetype = "dashed", color = "red", size = 1) +  # Proper diagonal line
  geom_point(aes(x = optimal_specificity, y = optimal_sensitivity), color = "black", size = 4) +  # Optimal point
  geom_text(aes(x = optimal_specificity, y = optimal_sensitivity), 
            label = paste0("Threshold: ", round(optimal_threshold_youden, 3), 
                           "\nSens: ", round(optimal_sensitivity, 3), 
                           "\nSpec: ", round(optimal_specificity, 3)),
            hjust = -0.1, vjust = -0.5, size = 5, color = "black") +  # Label for the point
  scale_x_reverse(name = "Specificity", limits = c(1, 0)) +  # Force Specificity from 1 to 0
  scale_y_continuous(name = "Sensitivity", limits = c(0, 1)) +
  ggtitle("ROC Curve for Logistic Regression Model") +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.2, label = paste("AUC:", round(auc(roc_curve), 3)), color = "blue", size = 5)

# We splited the data before between testing and training
# Now we test the model on the testing part
colSums(is.na(test))
test$prob <- predict(model3, newdata = test, type = "response")
test$predicted_class <-ifelse(test$prob>=optimal_threshold_youden,1,0)
test$predicted_class <- ifelse(test$predicted_class ==1,"yes","no")
test$predicted_class <- factor(test$predicted_class, levels = c("yes", "no"))
test$y2<- factor(test$y2, levels = c("yes", "no"))

roc_test_auc <- auc(roc(test$y2,test$prob,levels = c("no","yes"),direction = "<"))

# Create the confusion matrix for this last model
conf_matrix2 <- confusionMatrix(test$predicted_class,test$y2,positive="yes")
conf_matrix2

roc_test_auc

### 3. CHAID Model – Initial Tree
churn_data <- team_data
str(train)
str(churn_data)
churn_data <-churn_data[-c(1,6,10:13,21:22)]
str(churn_data)

# Define CHAID control parameters
# Define control parameters for the CHAID algorithm
ctrl <- chaid_control(
  minsplit = 10,     # Minimum number of observations needed in a node for a split to be attempted
  minprob  = 0.05,   # Minimum proportion of observations in a child node (helps avoid very small groups)
  stump    = FALSE   # If TRUE, forces the tree to be a single-level stump (no deep splits); we allow full tree here
)

train$y2 <- as.factor(train$y2)
str(train)
chaid_y2 <- chaid(y2 ~ age_group+job+marital+education+default+housing+loan+contact+duration_group+campaign_group+pdays_group+previous_group+balance_group+poutcome, data = train, control = ctrl)
#chaid_y4 <- chaid(y2 ~age_group+job+education, data = churn_data, control = ctrl)
#chaid_y3 <- chaid(y2 ~ job+marital+housing+loan+duration_group+campaign_group+pdays_group+poutcome ,data=train,control = ctrl)
print(chaid_y2)
plot(chaid_y2)

y2_P_chaid_train <- predict(chaid_y2, train)


#We create the confusion matrix for CHAID
confusionMatrix(y2_P_chaid_train, train$y2, positive = "yes")

test$y2 <- as.factor(test$y2)

y2_P_chaid_test <- predict(chaid_y2, test)
chaid_probs <- ifelse(y2_P_chaid_test == "yes", 1, 0)

confusionMatrix(y2_P_chaid_test, test$y2, positive = "yes")
chaid_probs <- ifelse(y2_P_chaid_test == "yes", 1, 0)

#Now we plot the ROC curve for the CHAID
roc_test_chaid <- data.frame(
  threshold = roc_curve$thresholds,
  sensitivity = roc_curve$sensitivities,
  specificity = roc_curve$specificities
)
roc_curve_chaid <- roc(response = test$y, predictor = chaid_probs,levels = c("0","1"),direction = "<")
ggplot(roc_test_chaid, aes(x = specificity, y = sensitivity)) +
  geom_line(color = "blue", size = 1.2) +  # ROC Curve in Blue
  geom_segment(aes(x = 1, y = 0, xend = 0, yend = 1), linetype = "dashed", color = "red", size = 1)+  # Proper diagonal line 
  scale_x_reverse(name = "Specificity", limits = c(1, 0)) +  # Force Specificity from 1 to 0
  scale_y_continuous(name = "Sensitivity", limits = c(0, 1)) +
  ggtitle("ROC Curve for CHAID decision tree") +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.2, label = paste("AUC:", round(auc(roc_curve_chaid), 3)), color = "blue", size = 5)
auc_chaid <- auc(roc_curve_chaid)

cat("AUC for CHAID:", auc_chaid, "\n")


#CART model-Initial tree
CART_tree <- rpart(
  y2 ~ age_group+job+marital+education+default+housing+loan+contact+duration_group+campaign_group+pdays_group+previous_group+balance_group+poutcome, 
  data = train,
  control = rpart.control(
    cp = 0.001,     # Complexity parameter: lower values allow deeper trees with more splits
    # A very small cp means the model will keep splitting even for tiny gains in purity
    minsplit = 10     # Minimum number of observations required in a node to attempt a split
    # A value of 2 allows splitting even the smallest groups — increases risk of overfitting
  )
)
plot(CART_tree)
CART_prediction_train <- predict(CART_tree, newdata = train, type = "class")
confusionMatrix(CART_prediction_train, train$y2, positive = "yes")

#We create the confusion matrix for CART
CART_prediction_test <- predict(CART_tree, newdata = test, type = "class")
confusionMatrix(CART_prediction_test, test$y2, positive = "yes")

CART_probs <- ifelse(CART_prediction_test == "yes", 1, 0)

#We plot the ROC curve for CART
roc_test_CART <- data.frame(
  threshold = roc_curve$thresholds,
  sensitivity = roc_curve$sensitivities,
  specificity = roc_curve$specificities
)
roc_curve_CART <- roc(response = test$y, predictor = CART_probs,levels = c("0","1"),direction = "<")
ggplot(roc_test_CART, aes(x = specificity, y = sensitivity)) +
  geom_line(color = "blue", size = 1.2) +  # ROC Curve in Blue
  geom_segment(aes(x = 1, y = 0, xend = 0, yend = 1), linetype = "dashed", color = "red", size = 1)+  # Proper diagonal line 
  scale_x_reverse(name = "Specificity", limits = c(1, 0)) +  # Force Specificity from 1 to 0
  scale_y_continuous(name = "Sensitivity", limits = c(0, 1)) +
  ggtitle("ROC Curve for CART decision tree") +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.2, label = paste("AUC:", round(auc(roc_curve_CART), 3)), color = "blue", size = 5)

auc_CART <- auc(roc_curve_CART)


cat("AUC for CART:", auc_CART, "\n")


#C5.0 model-Initial Tree
treeModel1 <- C5.0(
  y2 ~ age_group+job+marital+education+default+housing+loan+contact+duration_group+campaign_group+pdays_group+previous_group+balance_group+poutcome, 
  data = train
)
summary(treeModel1)
plot(treeModel1,
     compress = TRUE,
     uniform = TRUE,
     gp = gpar(fontsize = 6))

summary(treeModel1)

#We create the confusion matrix for C5.0
c50_prediction_train <- predict(treeModel1, newdata = train, type = "class")
confusionMatrix(c50_prediction_train, train$y2, positive = "yes")

C50_prediction_test <- predict(treeModel1, newdata = test, type = "class")
confusionMatrix(C50_prediction_test, test$y2, positive = "yes")


C50_probs <- ifelse(C50_prediction_test == "yes", 1, 0)

#We plot the ROC curve for C5.0
roc_test_C50 <- data.frame(
  threshold = roc_curve$thresholds,
  sensitivity = roc_curve$sensitivities,
  specificity = roc_curve$specificities
)

roc_curve_C50 <- roc(response = test$y, predictor = C50_probs,levels = c("0","1"),direction = "<")
ggplot(roc_test_C50, aes(x = specificity, y = sensitivity)) +
  geom_line(color = "blue", size = 1.2) +  # ROC Curve in Blue
  geom_segment(aes(x = 1, y = 0, xend = 0, yend = 1), linetype = "dashed", color = "red", size = 1)+  # Proper diagonal line 
  scale_x_reverse(name = "Specificity", limits = c(1, 0)) +  # Force Specificity from 1 to 0
  scale_y_continuous(name = "Sensitivity", limits = c(0, 1)) +
  ggtitle("ROC Curve for C50 decision tree") +
  theme_minimal() +
  annotate("text", x = 0.6, y = 0.2, label = paste("AUC:", round(auc(roc_curve_C50), 3)), color = "blue", size = 5)

auc_C50 <- auc(roc_curve_C50)


cat("AUC for C50:", auc_C50, "\n")


original_data <- team_data[-c(1,6,10:13)]
str(original_data)



# Fit Random Forest with 500 trees and different mtry values
# mtry = number of predictors considered at each split
data <- original_data[-9]
str(data)
data <- na.omit(data)

#We start training the random forest model with 500 trees
rf2 <- randomForest(y2 ~ ., data = data, ntree = 500, mtry = 2)
floor(sqrt(ncol(data) - 1))

#We evaluate the OOB error with different mtry values
oob.values <- numeric(8)
for (i in 1:8) {
  temp.model <- randomForest(y2 ~ ., data = data, ntree = 500, mtry = i)
  oob.values[i] <- temp.model$err.rate[nrow(temp.model$err.rate), 1]
}
print(oob.values)
min(oob.values)
which(oob.values == min(oob.values))
str(original_data)
mtry <- tuneRF(data[-9], data$y2, ntreeTry = 500,
               stepFactor = 0.8, improve = 0.05, trace = TRUE, plot = TRUE)
best.m <- mtry[mtry[, 2] == min(mtry[, 2]), 1]
print(mtry)
print(best.m)


#Visualization of the OOB error
error <- data.frame(rf2$err.rate)
error$n_trees <- 1:500
ggplot(error, aes(x = n_trees)) +
  geom_line(aes(y = OOB, color = "OOB Error")) +
  geom_line(aes(y = yes, color = "False Stay Rate")) +
  geom_line(aes(y = no, color = "False Churn Rate")) +
  scale_color_manual(values = c("green", "red", "blue")) +
  labs(x = "Number of Trees", y = "Error Rate") +
  theme_bw()

#We divide between train and test
set.seed(578)
train_idx <- sample(1:nrow(data), size = 0.8 * nrow(data))
test_idx <- setdiff(1:nrow(data), train_idx)

train_data <- data[train_idx, ]
test_data <- data[test_idx, ]
rf_model <- randomForest(y2 ~ ., data = train_data, ntree = 500, mtry = 3, importance = TRUE)
print(rf_model)
test_data$churn_pred <- predict(rf_model, test_data, type = "response")
confusionMatrix(data = test_data$churn_pred, reference = test_data$y2, positive = "yes")


rf_model2 <- randomForest(y2 ~ ., data = data, ntree = 500, mtry = 2, importance = TRUE)
print(rf_model2)
data$churn_pred <- predict(rf_model2, data, type = "response")
confusionMatrix(data = data$churn_pred, reference = data$y2, positive = "yes")

rf_model3 <- randomForest(y2 ~ .-churn_pred, data = data, ntree = 500, mtry = 3, importance = TRUE)
data$churn_pred <- predict(rf_model3, data, type = "response")
confusionMatrix(data = data$churn_pred, reference = data$y2, positive = "yes")


test_data$churn_pred <- predict(rf_model, test_data, type = "response")
confusionMatrix(data = test_data$churn_pred, reference = test_data$y2, positive = "yes")
importance(rf_model)


# Plot using built-in function
varImpPlot(rf_model3)

# Custom ggplot bar chart
var_imp <- importance(rf_model2)
var_imp_df <- data.frame(Variable = rownames(var_imp), Mean_Decrease_Gini = var_imp[, "MeanDecreaseGini"])

ggplot(var_imp_df, aes(x = reorder(Variable, Mean_Decrease_Gini), y = Mean_Decrease_Gini)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Variable Importance", x = "Variable", y = "Mean Decrease in Gini") +
  theme_minimal()

# -----------------------------------------------------------
# 6. ROC CURVE AND AUC (EVALUATING CLASSIFICATION PERFORMANCE)
# -----------------------------------------------------------

# Step 1: Predict class probabilities for the test set
# We extract the probability of the "Churn" class
prob_pred <- predict(rf_model2, test_data, type = "prob")[, "yes"]

# Step 2: Compute the ROC curve using the pROC package
# This shows the tradeoff between sensitivity (TPR) and 1 - specificity (FPR)
roc_curve <- roc(test_data$y2, prob_pred)
auc_rf <- round(roc_curve$auc, 4)
# Step 3: Print the Area Under the Curve (AUC)
# AUC tells us how well the model separates churners from non-churners
print(paste("AUC:", round(roc_curve$auc, 4)))

# Step 4: Find the optimal classification threshold using Youden's J
# This statistic maximizes the sum of sensitivity and specificity
youden <- roc_curve$sensitivities + roc_curve$specificities - 1
optimal_threshold <- roc_curve$thresholds[which.max(youden)]
print(paste("Optimal threshold (Youden's J):", round(optimal_threshold, 3)))

# Step 5: Plot the ROC curve with the optimal threshold highlighted
# This helps visually interpret model performance
threshold_table <- data.frame(
  Threshold = roc_curve$thresholds,
  Sensitivity = roc_curve$sensitivities,
  Specificity = roc_curve$specificities,
  Youden_J = youden
)

# Add column for 1 - Specificity (False Positive Rate)
threshold_table$spec_minus <- 1 - threshold_table$Specificity

# Identify the point with the maximum Youden's J
max_youden_row <- threshold_table[which.max(threshold_table$Youden_J), ]

# Plot using ggplot2
ggplot(threshold_table, aes(x = spec_minus, y = Sensitivity)) +
  geom_step(color = "#0072B2") +  # ROC curve line
  geom_point(data = max_youden_row, aes(x = spec_minus, y = Sensitivity),
             color = "#D55E00", size = 3) +  # Optimal point
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#009E73") +  # Random baseline
  labs(title = "ROC Curve",
       x = "1 - Specificity (False Positive Rate)",
       y = "Sensitivity (True Positive Rate)") +
  theme_minimal()

#AUCS
print(auc_chaid)
print(auc_CART)
print(auc_C50)
print(auc_rf)
print(roc_test_auc)

confusionMatrix_Chaid
confusionMatrix_C50
confusionMatrix_CART
confusionMatrix_RF

