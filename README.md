# Churn_Prediction_CLV
A churn prediction model using Customer Lifetime Value for Target Variable Creation

The main code snippets file churn_prediction contains the two main classess which will prepare the data for modelling and create a model, here just a Random Forest Model is used ,but different other models like GradientBoosting ,LR were tested. The Model will be created based on hyper parameters tuned using GridSearch CV with 3 fold cross validation with the pipelining technique.

The example.py file will show how to use the model just for a simple prediction which oputputs the final dataset with the churn probabilities for each team_id.


Run help(dataPreparation) and help(churnModel) to better understand the fucntions inside the two classes build.Defintions and explanation of each functions are given within each fuction inside both the classes.
An explantion is also given for each line of code as comment for better understanding what each line of code represents in that function.

The first class "dataPreparation" will prepare the data for modelling and the second class ""churnModel" actually does the modelling and output a model with the best parameters found from hyper parameter tuning - in this case a RandomForest Model.
