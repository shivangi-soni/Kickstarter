
Kickstarter is an American public benefit corporation that maintains a global crowdfunding platform. The way the platform works is that the project creators choose a minimum funding goal for their project and a deadline by which the funds should be collected from backers (people who pledge money). If the goal is not met by the deadline, the project is deemed as failed. The goal of this report is to develop classification, regression and clustering models to predict the amount of money pledged for a particular project, if a project will be successful or not, and gain insights regarding what makes a project successful, respectively.

**Data Pre-Processing and Feature Analysis**

Before building the classification and regression models, the data was pre-processed to get rid of missing/ null values, features that give identical information, and any features that can only be realized after the prediction starts. Since prediction tasks are assumed to be done at the time each project is launched, features such as staff pick, spotlight, backers count, pledged amount, launch to state change days, and weekday, month, day, hour, and year of state change were removed. Variables such as currency, launched hour, which were highly collinear to country and deadline hour, respectively were removed. Other features such as name, project ID were deemed irrelevant to making predictions; hence, they were removed as well. Any anomalies in the training dataset were removed using Isolation Forest algorithm as before proceeding with the feature selection.
After this data pre-processing and preliminary feature analysis, two feature selection techniques, Random Forest and Lasso were used to determine other insignificant features. It was observed that the important features selected by Lasso resulted in lower mean squared error (MSE) for regression model when compared to features selected by Random Forest. However, the opposite proved to be the case for classification, where the important features selected by Random Forest gave a higher accuracy as opposed to features selected by Lasso. It was observed for regression model that excluding features, such as deadline being on Friday and project’s creation day between 21st and 31st day of the month gave best results. For classification model, excluding features such as, robot category, flight category, deadline year of 2009, country Australia etc. gave a higher accuracy. 

**Final Regression and Classification Model Selection**
Different algorithms such as Random Forest, Gradient Boosting Tree, KNN, ANN, Logistic regression were tested to determine the best classification model to predict if the project will be successful or not. These same algorithms except logistic regression with some modifications (using regressor instead of classifier in Python) along with linear regression, Lasso, Ridge Regression were tested to determine the best regression model to predict the amount pledged in USD. The best MSE for regression model was obtained using Randon Forest while Gradient Boosting tree gave the highest accuracy score for the classification model. For both algorithms, the hyperparameters were tuned to build accurate models. 

**Conclusion and Business Interpretation**
Along with classification and regression models, clustering algorithm, K-Mean was used to gain business insights. Below are some of the conclusions that were derived from the models:
Regression Model: The best model gave mean squared error of 17.3 Billion, which indicates that on an average the error between predicted and actual value of pledged amount can be around +/- 131,148 USD. This error value is an important factor to consider as the higher the error, the predicted pledged amount will be less accurate.

Classification Model: The accuracy score of the best model was obtained to be 75.2%, which implies that the probability of the model misclassifying the state of the projects is around 25%. A higher precision and accuracy are necessary for businesses like Kickstarter since the higher the precision and accuracy, the more Kickstarter can trust the model to make informative decisions regarding the amount they would like to invest in advertising projects they deem would succeed. The precision score of the best model was obtained to be 67%, which means that if the algorithm identifies a project as a successful project, the probability that the algorithm is correct is 67%. 

Clustering Model: There were seven clusters generated using K-means to answer questions regarding what makes a project successful. Features that were used to obtain clusters include goal amount, length of the blurb and project name, duration between launch and deadline day, if project was staff pick or not, count of backers and state of the project. Based on the elbow method results, the optimal number of clusters was determined to be 7, with an average silhouette score of 0.389 and cluster scores ranging from 0.25 to 0.51. A snake plot was used to visualize different clusters. Cluster 5 has the projects with highest goal amount and a relatively higher duration between launch to deadline days while cluster 6 has projects with the lowest goal amount but the lengthiest projects’ name, backers count, and more successful projects. Cluster 3 has all the projects that have the lengthiest blurb with relative low number of backers and less chance of getting picked by staff. Cluster 4 has projects with the highest duration between launch and deadline date and lowest number of backers while also having the projects that were unsuccessful and not picked by staff. Cluster 1 includes all projects that were mostly picked by staff and had relatively lower goal amount and less duration between launch to deadline days while Cluster 2 includes projects with shortest length of the projects’ name and relatively lower goal amount but not picked by staff. 
