# UTS Data Mining
# Case 01: Breast Cancer
# Vira Jessica, NIM: 01112210045

# I. DATA PREPROCESSING
# Libraries associated with data preprocessing:
library(dplyr)
library(ggplot2)
library(reshape2)
library(MASS)

# Read CSV
df <- read.csv("wisc_bc_data.csv")

# Information about the data frame.
nrow(df) # Number of row in data frame.
ncol(df) # Number of column in data frame.
summary(df) # Summary of data frame.

# Check for duplicates and missing values.
df <- distinct(df) # Check for duplicates in the data.
missing_values <- sum(is.na(df)) # Check for missing values in the data.
missing_values # missing_values is 0, there are no missing values in the data.

# Check for misclassifications.
unique_class <- unique(df$diagnosis)
unique_class # unique_class are B" and "M", which means that there are only 2 classes in the target column and that there are no misclassifications.

# Drop ID and diagnosis for the features data frame.
features <- df[, !(names(df) %in% c("diagnosis", "id"))]

# Check for multicollinearity between features.
correlation_matrix <- cor(features)
melted_corr <- melt(correlation_matrix)

# Plot correlation heat-map.
ggplot(melted_corr, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "yellow", high = "blue", mid = "white", midpoint = 0, limit = c(-1,1), space = "Lab", name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 8, hjust = 1)) +
  coord_fixed() # Same explanation as above.

# Print which variables have correlation that's above or below 0.7.
high_corr <- which(correlation_matrix > 0.7 | correlation_matrix < -0.7, arr.ind = TRUE)
for (i in 1:nrow(high_corr)) {
  row_index <- high_corr[i, 1]
  col_index <- high_corr[i, 2]
  cat(colnames(correlation_matrix)[row_index], "and", colnames(correlation_matrix)[col_index], ":", correlation_matrix[row_index, col_index], "\n")
}

# Dropping columns which has correlation that's almost close to 1.
features <- as.data.frame(features)
features[, c("radius_mean", "perimeter_mean", "radius_se", "perimeter_se", "radius_worst", "perimeter_worst")] <- NULL

# Transform and normalize our features which are numerical data.
boxplot(features, las = 2, cex.axis = 0.8, cex.main = 1.2, main = "Boxplot of Features")
#constant_value <- 1e-20
#features <- features + constant_value
#log_features <- log(features) # Remove # in these two lines, change scale(features) to scale(log_features) below for model with log transformed data.
scaled_features <- scale(features)

#boxplot(scaled_features, las = 2, cex.axis = 0.8, cex.main = 1.2, main = "Boxplot of Scaled Features through Log Transformation")
boxplot(scaled_features, las = 2, cex.axis = 0.8, cex.main = 1.2, main = "Boxplot of Scaled Features")

absolute_z <- abs(scaled_features) # Find the absolute value of the scaled features.
threshold <- 3 # Set threshold to 3 (Z-score is 3) for identifying outliers
outlier_rows <- apply(absolute_z, 1, function(x) any(x > threshold)) # Return true is a row contains an outlier value.

# Label encode the target column (diagnosis).
class <- df$diagnosis
class_encoded <- ifelse(class == "M", 1, 0)
class_encoded # Label encode the target variable. "B" is 0 and "M" is 1.

# Final processed data & remove outliers.
df_processed <- cbind(scaled_features, class_encoded)
df_processed <- df_processed[!outlier_rows, ] # Remove outliers now that the features and class columns are combined.

df_processed <- as.data.frame(df_processed)

# Re-check by boxplot and correlation heat-map.
features_processed <- df_processed[, -ncol(df_processed)]
boxplot(features_processed, las = 2, cex.axis = 0.8, cex.main = 1.2, main = "Boxplot after Outlier Handling (Threshold of Z-Score = 3)")

correlation_matrix1 <- cor(features_processed)
melted_corr1 <- melt(correlation_matrix1)
ggplot(melted_corr1, aes(Var1, Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(low = "yellow", high = "blue", mid = "white", midpoint = 0, limit = c(-1,1), space = "Lab", name="Correlation") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 8, hjust = 1)) +
  coord_fixed() # Same explanation as above.

# Check histogram for every features now after data preprocessing.
hist(features_processed$texture_mean)
hist(features_processed$area_mean)
hist(features_processed$smoothness_mean)
hist(features_processed$compactness_mean)
hist(features_processed$concavity_mean)
hist(features_processed$points_mean)
hist(features_processed$symmetry_mean)
hist(features_processed$dimension_mean)

hist(features_processed$texture_se)
hist(features_processed$area_se)
hist(features_processed$smoothness_se)
hist(features_processed$compactness_se)
hist(features_processed$concavity_se)
hist(features_processed$points_se)
hist(features_processed$symmetry_se)
hist(features_processed$dimension_se)

hist(features_processed$texture_worst)
hist(features_processed$area_worst)
hist(features_processed$smoothness_worst)
hist(features_processed$compactness_worst)
hist(features_processed$concavity_worst)
hist(features_processed$points_worst)
hist(features_processed$symmetry_worst)
hist(features_processed$dimension_worst)
# Some features are heavily skewed and should be transformed for skewness correction. 
# However, model performed better without log transformation. Further analysis will be described in the paper.

# II. MODEL: LOGISTIC REGRESSION
# Libraries associated with the model:
library(caret)

# Training and testing data split.
set.seed(1)

index <- createDataPartition(df_processed$class_encoded, p = 0.8, list = FALSE) # Set train data to be 80% and test data to be 20%.
df_train <- df_processed[index, ] # Assign all train data to df_train.
df_test <- df_processed[-index, ] # Assign all test data to df_test.

# Set X and Y for train and test data.
x_train <- df_train[, !(names(df_train) %in% c("class_encoded"))] # Drop target column, leaving only the features in x.
y_train <- df_train$class_encoded # Set y as the target column.

x_test <- df_test[, !(names(df_test) %in% c("class_encoded"))]
y_test <- df_test$class_encoded # Same explanation as above.

# Logistic regression model.
model <- glm(y_train ~ ., data = x_train, family = binomial(link = "logit"))
# Note that logit is a sigmoid function for binary classification.

summary(model) # Summary of our model.

# III. VALIDATION AND TEST
# Predict validation data.
predictions <- predict(model, newdata = x_train, type = "response") # Predict train data for validation.
final_predictions <- ifelse(predictions >= 0.5, 1, 0) # Threshold for logistic regression, return 1 if prediction is greater than or equal to 0.5, return 0 otherwise.

# Confusion matrix for validation.
confusion_matrix <- confusionMatrix(factor(final_predictions), factor(y_train)) 
confusion_matrix # 0 values for false positive and false negative. Model is able to accurately predict the class.

# F1 score for validation.
f1_score <- 2 * confusion_matrix$byClass["Sensitivity"] * confusion_matrix$byClass["Pos Pred Value"] / 
  (confusion_matrix$byClass["Sensitivity"] + confusion_matrix$byClass["Pos Pred Value"])
f1_score # F1 Score is 1 since predicted data are all correct when compared to actual data.

# Testing model on test data.
predictions <- predict(model, newdata = x_test, type = "response") # Testing model on test data (data that didn't go to the model to be trained).
final_predictions <- ifelse(predictions >= 0.5, 1, 0)

# Confusion matrix for testing.
confusion_matrix <- confusionMatrix(factor(final_predictions), factor(y_test))
confusion_matrix # 1 false positive and 3 false negative, the rest are all accurate predictions.

# F1 score for testing.
f1_score <- 2 * confusion_matrix$byClass["Sensitivity"] * confusion_matrix$byClass["Pos Pred Value"] / 
  (confusion_matrix$byClass["Sensitivity"] + confusion_matrix$byClass["Pos Pred Value"])
f1_score # F1 score of the final model after tested on test data is 0.9672131. This value is extremely high and indicates that the model performs excellently on this data.

# Calculate accuracy score.
accuracy <- mean(final_predictions == y_test) * 100
accuracy

# Calculate precision.
precision <- sum(final_predictions & y_test) / sum(final_predictions)
precision

# Analyzing feature importance.
coefficients <- coef(model) # Load the coefficients.
barplot(abs(coefficients[-1]), names.arg = names(coefficients[-1]), 
        main = "Feature Importance", ylab = "Absolute Coefficient Value",
        cex.names = 0.8, las = 2)
# Plot the absolute value of the coefficients to determine which features affect the model significantly.
# Based on the bar plot, area_mean and area_worst affect the model the most.
