#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#%%

###After running the main code classes Run this

prediction_database = dataPreparation(licenses,activity).prediction_dataset()

model = churnModel(prediction_database).random_forest()


feature_columns = [x for x in prediction_database.columns.tolist() if x not in ['team_id','license_start_recency','license_end_recency_y','license_end_recency_x', 'Target1','Target2']]
X_train, X_test, y_train, y_test = train_test_split(prediction_database[feature_columns], 
                                                    prediction_database.Target1,
                                                    test_size=0.25 , random_state=0)


y_pred = model.predict(X_test)


cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(15,10)) 
sns.heatmap(cm, annot=True, fmt='d', cmap='plasma',ax=ax)

#To check accuracy and other stats.
print(classification_report(y_pred, y_test))

#Feature Importances
importances = model.feature_importances_
#forest_importances = pd.Series(importances, index=feature_names)
feat_importances = pd.DataFrame(importances, index=feature_columns)
feat_importances1=feat_importances.sort_values([0], ascending=[False])
#feat_importances1 = feat_importances1.iloc[:8,:]
feat_importances1[0] = feat_importances[0]*100

feat_importances1.plot(kind='barh')


#Final Probability
#Probability prediction
probability = model.predict_proba(prediction_database[feature_columns]) 

prediction_database['churn_probability'] = probability[:,1]




