# Churn_Prediction_CLV
A churn prediction model using Customer Lifetime Value for Target Variable Creation

The main code snippets file churn_prediction contains the two main classess which will prepare the data for modelling and create a model, here just a Random Forest Model is used ,but different other models like GradientBoosting ,LR were tested. The Model will be created based on hyper parameters tuned using GridSearch CV with 3 fold cross validation with the pipelining technique.

The example.py file will show how to use the model just for a simple prediction which oputputs the final dataset with the churn probabilities for each team_id.
