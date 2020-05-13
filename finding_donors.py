#!/usr/bin/env python
# coding: utf-8

# ## Supervised Learning
# ## Project: Finding Donors for *CharityML*

# In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Please specify WHICH VERSION OF PYTHON you are using when submitting this notebook. Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ## Getting Started
# 
# In this project, you will employ several supervised algorithms of your choice to accurately model individuals' income using data collected from the 1994 U.S. Census. You will then choose the best candidate algorithm from preliminary results and further optimize this algorithm to best model the data. Your goal with this implementation is to construct a model that accurately predicts whether an individual makes more than $50,000. This sort of task can arise in a non-profit setting, where organizations survive on donations.  Understanding an individual's income can help a non-profit better understand how large of a donation to request, or whether or not they should reach out to begin with.  While it can be difficult to determine an individual's general income bracket directly from public sources, we can (as we will see) infer this value from other publically available features. 
# 
# The dataset for this project originates from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Census+Income). The datset was donated by Ron Kohavi and Barry Becker, after being published in the article _"Scaling Up the Accuracy of Naive-Bayes Classifiers: A Decision-Tree Hybrid"_. You can find the article by Ron Kohavi [online](https://www.aaai.org/Papers/KDD/1996/KDD96-033.pdf). The data we investigate here consists of small changes to the original dataset, such as removing the `'fnlwgt'` feature and records with missing or ill-formatted entries.

# ----
# ## Exploring the Data
# Run the code cell below to load necessary Python libraries and load the census data. Note that the last column from this dataset, `'income'`, will be our target label (whether an individual makes more than, or at most, $50,000 annually). All other columns are features about each individual in the census database.

# In[35]:


# Import libraries necessary for this project
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # Allows the use of display() for DataFrames
# Import supplementary visualization code visuals.py
import visuals as vs
# Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
# Pretty display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')
#Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import accuracy_score,fbeta_score
# Import functionality for cloning a model
from sklearn.base import clone
# Import the three supervised learning models from sklearn
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
# Import sklearn.preprocessing.StandardScaler
from sklearn.preprocessing import MinMaxScaler
# cross_validation depreceated in my version
from sklearn.model_selection import train_test_split

# ### Implementation - Creating a Training and Predicting Pipeline
# To properly evaluate the performance of each model you've chosen, it's important that you create a training and predicting pipeline that allows you to quickly and effectively train models using various sizes of training data and perform predictions on the testing data. Your implementation here will be used in the following section.
# In the code block below, you will need to implement the following:
#  - Import `fbeta_score` and `accuracy_score` from [`sklearn.metrics`](http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics).
#  - Fit the learner to the sampled training data and record the training time.
#  - Perform predictions on the test data `X_test`, and also on the first 300 training points `X_train[:300]`.
#    - Record the total prediction time.
#  - Calculate the accuracy score for both the training subset and testing set.
#  - Calculate the F-score for both the training subset and testing set.
#    - Make sure that you set the `beta` parameter!

def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''
    
    #check if the sample size is less than the training data size
    if(sample_size>len(X_train)):
        raise Exception('sample size exceeds maximum number of data points')
        
    results = {}
    
    #lets convert series/dataframes to matrices because some algorithms want us to ravel() data in the labels as a contiguous 1D array lets do it.
    if(False == isinstance(y_train, (list, np.ndarray))):
        y_train=y_train.values
        y_train.ravel()
    #do the same for the y_test
    if(False == isinstance(y_test, (list, np.ndarray))):
        y_test=y_test.values
        y_test.ravel()
    
    #Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time() # Get start time
    learner = learner.fit(X_train.iloc[:sample_size,:],y_train[:sample_size])
    end = time() # Get end time
    
    #Calculate the training time
    results['train_time'] = abs(end - start)

    # Get the predictions on the test set(X_test),
    # then get predictions on the first 300 training samples(X_train) using .predict()
    start = time() # Get start time
    predictions_test = learner.predict(X_test) 
    predictions_train = learner.predict(X_train.iloc[:300,:])
    end = time() # Get end time
    
    # TODO: Calculate the total prediction time
    results['pred_time'] = abs(end - start)
            
    # TODO: Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300],predictions_train)
        
    # TODO: Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test,predictions_test)
    
    # TODO: Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:300],predictions_train,beta=0.5)
        
    # TODO: Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test,predictions_test,beta=0.5)
       
    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
        
    # Return the results
    return results

def train_test():
    # Load the Census dataset
    data = pd.read_csv("census.csv")

    # Total number of records
    n_records = len(data)

    # Number of records where individual's income is more than $50,000
    n_greater_50k = sum(data['income'].str.count(">50K"))

    # Number of records where individual's income is at most $50,000
    n_at_most_50k =  sum(data['income'].str.count("<=50K"))

    # Percentage of individuals whose income is more than $50,000
    greater_percent =  (n_greater_50k/n_records)*100

    # Print the results
    print("Total number of records: {}".format(n_records))
    print("Individuals making more than $50,000: {}".format(n_greater_50k))
    print("Individuals making at most $50,000: {}".format(n_at_most_50k))
    print("Percentage of individuals making more than $50,000: {}%".format(greater_percent))

    # ## Preparing the Data
    # Before data can be used as input for machine learning algorithms, it often must be cleaned, formatted, and restructured — this is typically known as **preprocessing**. Fortunately, for this dataset, there are no invalid or missing entries we must deal with, however, there are some qualities about certain features that must be adjusted. This preprocessing can help tremendously with the outcome and predictive power of nearly all learning algorithms.

    # ### Transforming Skewed Continuous Features
    # A dataset may sometimes contain at least one feature whose values tend to lie near a single number, but will also have a non-trivial number of vastly larger or smaller values than that single number.  Algorithms can be sensitive to such distributions of values and can underperform if the range is not properly normalized. With the census dataset two features fit this description: '`capital-gain'` and `'capital-loss'`. 
    # 
    # Run the code cell below to plot a histogram of these two features. Note the range of the values present and how they are distributed.
    # Split the data into features and target label
    income_raw = data[['income']]
    features_raw = data.drop('income', axis = 1)

    # Visualize skewed continuous features of original data
    vs.distribution(data)

    # For highly-skewed feature distributions such as `'capital-gain'` and `'capital-loss'`, it is common practice to apply a <a href="https://en.wikipedia.org/wiki/Data_transformation_(statistics)">logarithmic transformation</a> on the data so that the very large and very small values do not negatively affect the performance of a learning algorithm. Using a logarithmic transformation significantly reduces the range of values caused by outliers. Care must be taken when applying this transformation however: The logarithm of `0` is undefined, so we must translate the values by a small amount above `0` to apply the the logarithm successfully.
    # 
    # Run the code cell below to perform a transformation on the data and visualize the results. Again, note the range of values and how they are distributed. 
    # Log-transform the skewed features
    skewed = ['capital-gain', 'capital-loss']
    features_log_transformed = pd.DataFrame(data = features_raw)
    features_log_transformed[skewed] = features_raw[skewed].apply(lambda x: np.log(x + 1))

    # Visualize the new log distributions
    vs.distribution(features_log_transformed, transformed = True)


    # ### Normalizing Numerical Features
    # In addition to performing transformations on features that are highly skewed, it is often good practice to perform some type of scaling on numerical features. Applying a scaling to the data does not change the shape of each feature's distribution (such as `'capital-gain'` or `'capital-loss'` above); however, normalization ensures that each feature is treated equally when applying supervised learners. Note that once scaling is applied, observing the data in its raw form will no longer have the same original meaning, as exampled below.
    # 
    # normalize each numerical feature. We will use [`sklearn.preprocessing.MinMaxScaler`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html) for this.

    
    # Initialize a scaler, then apply it to the features
    scaler = MinMaxScaler() # default=(0, 1)
    numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']

    features_log_minmax_transform = pd.DataFrame(data = features_log_transformed)
    features_log_minmax_transform[numerical] = scaler.fit_transform(features_log_transformed[numerical])

    # Show an example of a record with scaling applied
    display(features_log_minmax_transform.head(n = 5))


    # ### Implementation: Data Preprocessing
    # 
    # From the table in **Exploring the Data** above, we can see there are several features for each record that are non-numeric. Typically, learning algorithms expect input to be numeric, which requires that non-numeric features (called *categorical variables*) be converted. One popular way to convert categorical variables is by using the **one-hot encoding** scheme. One-hot encoding creates a _"dummy"_ variable for each possible category of each non-numeric feature. For example, assume `someFeature` has three possible entries: `A`, `B`, or `C`. We then encode this feature into `someFeature_A`, `someFeature_B` and `someFeature_C`.
    # 
    # |     | someFeature |                             | someFeature_A | someFeature_B | someFeature_C |
    # | :-: | :-:         |                             | :-----------: | :-----------: | :-----------: |
    # | 0   |  B          |                             |        0      |       1       |        0      |
    # | 1   |  C          | ----> one-hot encode ---->  |        0      |       0       |        1      |
    # | 2   |  A          |                             |        1      |       0       |        0      |
    # 
    # Additionally, as with the non-numeric features, we need to convert the non-numeric target label, `'income'` to numerical values for the learning algorithm to work. Since there are only two possible categories for this label ("<=50K" and ">50K"), we can avoid using one-hot encoding and simply encode these two categories as `0` and `1`, respectively. In code cell below, you will need to implement the following:
    #  - Use [`pandas.get_dummies()`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html?highlight=get_dummies#pandas.get_dummies) to perform one-hot encoding on the `'features_log_minmax_transform'` data.
    #  - Convert the target label `'income_raw'` to numerical entries.
    #    - Set records with "<=50K" to `0` and records with ">50K" to `1`.

    # One-hot encode the 'features_log_minmax_transform' data using pandas.get_dummies()
    features_final = pd.get_dummies(features_log_minmax_transform)

    # Encode the 'income_raw' data to numerical values
    income = income_raw.where(income_raw['income'].str.contains("<=50K"), 1)
    income.where(income['income'].str.contains(">50K"),0,inplace=True)
    # change type to integer from type object
    income=income['income'].astype(int)
    #check all is good.
    print(income.dtypes)
    #if everything went well mean in the describe will show the percentage of 1's or >50 which should equal 24.78% 
    income.describe()

    # Print the number of features after one-hot encoding
    encoded = list(features_final.columns)
    print("{} total features after one-hot encoding.".format(len(encoded)))

    # ### Shuffle and Split Data
    # Now all _categorical variables_ have been converted into numerical features, and all numerical features have been normalized. As always, we will now split the data (both features and their labels) into training and test sets. 80% of the data will be used for training and 20% for testing.
    # Split the 'features' and 'income' data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features_final, 
                                                    income, 
                                                    test_size = 0.2, 
                                                    random_state = 0)
    # Show the results of the split
    #print("Training set has {} samples.".format(X_train.shape[0]))
    #print("Testing set has {} samples.".format(X_test.shape[0]))
    # ----
    # ## Evaluating Model Performance
    # In this section, we will investigate four different algorithms, and determine which is best at modeling the data. Three of these algorithms will be supervised learners of your choice, and the fourth algorithm is known as a *naive predictor*.
    
    # ### Metrics and the Naive Predictor
    # *CharityML*, equipped with their research, knows individuals that make more than \$50,000 are most likely to donate to their charity. Because of this, *CharityML* is particularly interested in predicting who makes more than \$50,000 accurately. It would seem that using **accuracy** as a metric for evaluating a particular model's performace would be appropriate. Additionally, identifying someone that *does not* make more than \$50,000 as someone who does would be detrimental to *CharityML*, since they are looking to find individuals willing to donate. Therefore, a model's ability to precisely predict those that make more than \$50,000 is *more important* than the model's ability to **recall** those individuals. We can use **F-beta score** as a metric that considers both precision and recall:
    # 
    # $$ F_{\beta} = (1 + \beta^2) \cdot \frac{precision \cdot recall}{\left( \beta^2 \cdot precision \right) + recall} $$
    # 
    # In particular, when $\beta = 0.5$, more emphasis is placed on precision. This is called the **F$_{0.5}$ score** (or F-score for simplicity).
    #   
    # Looking at the distribution of classes (those who make at most \$50,000, and those who make more), it's clear most individuals do not make more than \$50,000. This can greatly affect **accuracy**, since we could simply say *"this person does not make more than \$50,000"* and generally be right, without ever looking at the data! Making such a statement would be called **naive**, since we have not considered any information to substantiate the claim. It is always important to consider the *naive prediction* for your data, to help establish a benchmark for whether a model is performing well. That been said, using that prediction would be pointless: If we predicted all people made less than \$50,000, *CharityML* would identify no one as donors. 
    # 
    # 
    # #### Note: Recap of accuracy, precision, recall
    # 
    # ** Accuracy ** measures how often the classifier makes the correct prediction. It’s the ratio of the number of correct predictions to the total number of predictions (the number of test data points).
    # 
    # ** Precision ** tells us what proportion of messages we classified as spam, actually were spam.
    # It is a ratio of true positives(words classified as spam, and which are actually spam) to all positives(all words classified as spam, irrespective of whether that was the correct classificatio), in other words it is the ratio of
    # 
    # `[True Positives/(True Positives + False Positives)]`
    # 
    # ** Recall(sensitivity)** tells us what proportion of messages that actually were spam were classified by us as spam.
    # It is a ratio of true positives(words classified as spam, and which are actually spam) to all the words that were actually spam, in other words it is the ratio of
    # 
    # `[True Positives/(True Positives + False Negatives)]`
    # 
    # For classification problems that are skewed in their classification distributions like in our case, for example if we had a 100 text messages and only 2 were spam and the rest 98 weren't, accuracy by itself is not a very good metric. We could classify 90 messages as not spam(including the 2 that were spam but we classify them as not spam, hence they would be false negatives) and 10 as spam(all 10 false positives) and still get a reasonably good accuracy score. For such cases, precision and recall come in very handy. These two metrics can be combined to get the F1 score, which is weighted average(harmonic mean) of the precision and recall scores. This score can range from 0 to 1, with 1 being the best possible F1 score(we take the harmonic mean as we are dealing with ratios).

    # ### Question 1 - Naive Predictor Performace
    # * If we chose a model that always predicted an individual made more than $50,000, what would  that model's accuracy and F-score be on this dataset? You must use the code cell below and assign your results to `'accuracy'` and `'fscore'` to be used later.
    # 
    # ** Please note ** that the the purpose of generating a naive predictor is simply to show what a base model without any intelligence would look like. In the real world, ideally your base model would be either the results of a previous model or could be based on a research paper upon which you are looking to improve. When there is no benchmark model set, getting a result better than random choice is a place you could start from.
    # 
    # ** HINT: ** 
    # 
    # * When we have a model that always predicts '1' (i.e. the individual makes more than 50k) then our model will have no True Negatives(TN) or False Negatives(FN) as we are not making any negative('0' value) predictions. Therefore our Accuracy in this case becomes the same as our Precision(True Positives/(True Positives + False Positives)) as every prediction that we have made with value '1' that should have '0' becomes a False Positive; therefore our denominator in this case is the total number of records we have in total. 
    # * Our Recall score(True Positives/(True Positives + False Negatives)) in this setting becomes 1 as we have no False Negatives.

    TP = np.sum(income) # Counting the ones as this is the naive case. Note that 'income' is the 'income_raw' data 
    #encoded to numerical values done in the data preprocessing step.
    FP = income.count() - TP # Specific to the naive case
    TN = 0 # No predicted negatives in the naive case
    FN = 0 # No predicted negatives in the naive case    
    # Calculate accuracy, precision and recall
    accuracy = (TP+TN)/(TP+TN+FN+FP)
    recall = TP/(TP+FN)
    precision = TP/(TP+FP)

    # Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
    fscore = (1+0.25)*(precision*recall)/(0.25*precision + recall)

    # ###  Supervised Learning Models
    # **The following are some of the supervised learning models that are currently available in** [`scikit-learn`](http://scikit-learn.org/stable/supervised_learning.html) **that you may choose from:**
    # - Gaussian Naive Bayes (GaussianNB)
    # - Decision Trees
    # - Ensemble Methods (Bagging, AdaBoost, Random Forest, Gradient Boosting)
    # - K-Nearest Neighbors (KNeighbors)
    # - Stochastic Gradient Descent Classifier (SGDC)
    # - Support Vector Machines (SVM)
    # - Logistic Regression

    # ### Question 2 - Model Application
    # List three of the supervised learning models above that are appropriate for this problem that you will test on the census data. For each model chosen
    # 
    # - Describe one real-world application in industry where the model can be applied. 
    # - What are the strengths of the model; when does it perform well?
    # - What are the weaknesses of the model; when does it perform poorly?
    # - What makes this model a good candidate for the problem, given what you know about the data?
    # 
    # ** HINT: **
    # 
    # Structure your answer in the same format as above^, with 4 parts for each of the three models you pick. Please include references with your answer.

    # **Answer:**
    # 
    # Before choosing the three models a look into the data reveals useful information on which kind of classifier may do well. 
    # The data has 5 numerical features apart from that all others were categorically one hot encoded which blew the feature Set to 100+ features. 
    # This makes the data a good candidate for decision trees, Ensemble methods such as random forest and Boosting methods.
    # The Naive Bayes is probably not going to do really well because the categories have data that are be dependant on each other. it will not work the best.  
    # Decision trees were dropped because Adaboost is used with decision tree as weak learners to overcome its inherent problem of overfitting.
    # 
    # **Random Forest**
    # - Identify the patient’s disease(diabetic retinopathy) by analyzing the patient’s medical record [1]
    # - individual decision trees typically exhibit high variance and tend to overfit, In random forests, each tree in the ensemble is built from a sample drawn with replacement (i.e., a bootstrap 
    #   sample) from the training set. additionally when splitting each node during the construction of a tree, best split is found from all input features or a random subset of max_features 
    #   making 2 sources of randomness which can reduce high variance and help in solving the overfitting problem.[2]
    # - prediction models arent easily interpretable like a decision tree is. and computational costs are high for training as well as predictions as predictions can take time which may make it unideal   for certain applications. [3]
    # - a lot of categorical data which makes decision trees a nice option. and Random forests are just decision trees that are made in a different way to avoid overfitting. 
    # 
    # **AdaBoost Classifier**
    # - Detection of pedestrians using patterns of motion and appearance [4].
    # - it is fast, simple and easy to program. Also, it has the flexibility to be combined with any machine learning algorithm; It has been  
    #   extended to learning problems beyond binary classification and it is versatile as it can be used with text or numeric data.[5]
    # - AdaBoost is constructed from empirical evidence and particularly vulnerable to uniform noise. ; Weak classifiers being too weak can lead to low margins and overfitting.[5]
    # - Adaboost is a boosting method of having weaklearners designed to individually underfit. For each successive iteration, the sample weights are individually modified and the learning algorithm is 
    #   reapplied to the reweighted data. this would perform better than a single decision tree [6]    
    # 
    # **Gradient Boosting Classifier**
    # - Flight Arrival Delay Prediction Using Gradient Boosting Classifier [7].
    # - AdaBoost can be seen as a special case with a particular loss function. Hence, gradient boosting is much more flexible; The classifier works best when minimum data cleaning is required on the 
    #   data such as the one present in this project.[8]
    # - If features are strongly correlated to each other in the data; the linear classifier can tend to over-predict if none of the features are dropped. This is somewhat mitigated by using L1 or L2 
    #   regularization, but not completely eliminated.[8]
    # - Again a lot of categorical data which suits the decision tree classifier. the Gradient Boosting classifier is a generalises the AdaBoost classifier hence, it has the potential to perform better 
    #   than AdaBoost Classifier.
    # 
    # References
    # 
    # [1] https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4062420/
    # 
    # [2] https://scikit-learn.org/stable/modules/ensemble.html#id7
    # 
    # [3] https://www.oreilly.com/library/view/hands-on-machine-learning/9781789346411/e17de38e-421e-4577-afc3-efdd4e02a468.xhtml
    # 
    # [4] Viola, Jones and Snow, "Detecting pedestrians using patterns of motion and appearance," Proceedings Ninth IEEE International Conference on Computer Vision, Nice, France, 2003, pp. 734-741         vol.2, doi: 10.1109/ICCV.2003.1238422.
    # 
    # [5] https://www.educba.com/adaboost-algorithm/
    # 
    # [6] https://scikit-learn.org/stable/modules/ensemble.html#adaboost
    # 
    # [7] https://link.springer.com/chapter/10.1007/978-981-13-1498-8_57
    # 
    # [8] https://www.quora.com/Why-does-Gradient-boosting-work-so-well-for-so-many-Kaggle-problems


    # ### Implementation: Initial Model Evaluation
    # In the code cell, you will need to implement the following:
    # - Import the three supervised learning models you've discussed in the previous section.
    # - Initialize the three models and store them in `'clf_A'`, `'clf_B'`, and `'clf_C'`.
    #   - Use a `'random_state'` for each model you use, if provided.
    #   - **Note:** Use the default settings for each model — you will tune one specific model in a later section.
    # - Calculate the number of records equal to 1%, 10%, and 100% of the training data.
    #   - Store those values in `'samples_1'`, `'samples_10'`, and `'samples_100'` respectively.
    # 
    # **Note:** Depending on which algorithms you chose, the following implementation may take some time to run!

    # Initialize the three models
    clf_A = RandomForestClassifier()
    clf_B = AdaBoostClassifier()
    clf_C = GradientBoostingClassifier()

    # Calculate the number of samples for 1%, 10%, and 100% of the training data
    samples_100 = len(X_train)
    samples_10 = int(samples_100*0.1)
    samples_1 = int(samples_100*0.01)
    
    # Collect results on the learners
    results = {}
    for clf in [clf_A, clf_B, clf_C]:
        clf_name = clf.__class__.__name__
        results[clf_name] = {}
        for i, samples in enumerate([samples_1, samples_10, samples_100]):
            results[clf_name][i] =         train_predict(clf, samples, X_train, y_train, X_test, y_test)
    # Run metrics visualization for the three supervised learning models chosen
    vs.evaluate(results, accuracy, fscore)
    
    # ----
    # ## Improving Results
    # In this final section, you will choose from the three supervised learning models the *best* model to use on the student data. You will then perform a grid search optimization for the model over the entire training set (`X_train` and `y_train`) by tuning at least one parameter to improve upon the untuned model's F-score. 
    
    # ### Question 3 - Choosing the Best Model
    # 
    # * Based on the evaluation you performed earlier, in one to two paragraphs, explain to *CharityML* which of the three models you believe to be most appropriate for the task of identifying individuals that make more than \$50,000. 
    # 
    # ** HINT: ** 
    # Look at the graph at the bottom left from the cell above(the visualization created by `vs.evaluate(results, accuracy, fscore)`) and check the F score for the testing set when 100% of the training set is used. Which model has the highest score? Your answer should include discussion of the:
    # * metrics - F score on the testing when 100% of the training data is used, 
    # * prediction/training time
    # * the algorithm's suitability for the data.

    # **Answer:**
    # 
    # - Gradient Boosting Classifier performed well on the test set with an F-Score of 74% and will be used further. 
    # 
    # **Explanation**
    # - metrics : From the results, it was evident that the Random forest classifier is overfitting to the training data as a major decrease in performance is noted from the training accuracy(94%) to  
    #   testing accuracy(84%). Additionally, the Fscore of 67.9% is the least of the three classifiers. Thus, the random forest classifier was dropped.
    #   Between the Adaboost and Gradient Boosting Classifier(GBC) the Gradient boosting classifier(Fscore = 74%) is chosen as it marginally performs better than the Adaboost(72%) in terms of Fscore.
    #   
    # - Prediction/Training Time : In terms of prediction time, The GBC(0.05s) is 5.8 times faster in prediction than the Adaboost(0.29s). 
    #   Note: Initially the experimental Histogram Gradient Boosting classifier was earmarked for selection of the three models which is much faster to train and predict according to sklearn 
    #   documentation for large samples (n>10k) but, it was still experimental in Sklearn when this project was submitted so the Histogram Gradient Boosting was not used.[HSTGBC]
    #  
    # - Suitability : The data at hand is definitely well suited for a decision tree type classifier. However, the decision tree was not used due to its overfitting nature. Furthermore, Random forests 
    #   that curb this problem was also found to overfit. Hence, the decision of using a boosting classifier is a good choice. The reason why AdaBoost was checked against GBC is because Gradient 
    #   Boosting is a generic algorithm to find approximate solutions to the additive modeling problem, while AdaBoost can be seen as a special case with a particular loss function. Hence, gradient 
    #   boosting is much more flexible and can give better result.[ADBGBC] Additionally, the fundamental reason why Boosting technique is used is that the Weak learners are designed to underfit and 
    #   many such learners are used so that the problem of overfitting from a single classifier is reduced due to the randomness of the estimators. 
    #   
    #   Thus, the Gradient Boosting Classifier was further explored.
    # 
    # References:
    # 
    # [HSTGBC] https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html#sklearn.ensemble.HistGradientBoostingClassifier
    # 
    # [ADBGBC] https://datascience.stackexchange.com/questions/39193/adaboost-vs-gradient-boosting

    # ### Question 4 - Describing the Model in Layman's Terms
    # 
    # * In one to two paragraphs, explain to *CharityML*, in layman's terms, how the final model chosen is supposed to work. Be sure that you are describing the major qualities of the model, such as how the model is trained and how the model makes a prediction. Avoid using advanced mathematical jargon, such as describing equations.
    # 
    # ** HINT: **
    # 
    # When explaining your model, if using external resources please include all citations.
    
    # **Answer: ** 
    # 
    # in Layman terms the data present has some numerical values and it also has categorical values. so, a classifier that can handle both must be used. Ensembles and trees offer such an option.
    # 
    # **Chosen model**
    # Since, the data has features that fit a sort of rule-based classification problem, a kind of an if-else decision on the features can be drawn. thus, a decision tree could be imagined.
    # It is already established well that the decision trees cannot describe the data which is feature intensive without reaching a large depth which then causes the model to overfit. This was the premise of using a boosting classifier model. the model can be thought of building small decision trees iteratively with small depths like say (1 to 4 nodes tall) focusing on few features of the data and then in the end their results of the classification are collected and added so that they all together will classify using the data as a whole as accurately as possible hopefully without over-fitting.
    # 
    # **Data pre-processing**
    # After making sure that the features are normalized within a range and there are no missing values or the missing values are imputed, the model can be trained. To effectively train the model and observe how well it learns the features of the dataset, a pipeline to train was created. this pipeline will now take in the training data and test data and make splices in the data so that each of the model can be observed on how it "learns" when data is fed in fractional quantities such as  1% 10% and 100% of total data. this way models that don't learn features at all may be dropped or if the rate of learning is not sufficient enough may be noted to change the learning rate if needed. 
    # 
    # **Model**
    # 
    # Model used here is the Gradient Boosting Classifier. Think of the classifier as having a smaller depth decision tree and each focused on only a certain unique features. these learners are weak as in they can classify somewhat well with the features that they are exposed to but individually cannot classify using the entire data.
    # 
    # Gradient boosting involves three main steps. 
    # The first step that is required is that a loss function be optimized. The loss function must be diﬀerentiable. A loss function measures how well a machine learning model fits the data of a certain phenomenon. Different loss function may be used depending on the type of problem. Different loss function can be used on speech or image recognition, predicting the price of real estate, and describing user behavior on a web site. The loss function depends on the type of problem. For example, regression may use a squared error and classiﬁcation may use logarithmic loss.
    # 
    # The second step is the use of a weak learner. 
    # In gradient boosters, the weak learner is a decision tree. Speciﬁcally regression trees are used that output real values for splits and whose output can be added together, allowing subsequent models outputs to be added to correct the residuals in the predictions of the previous iteration. The algorithms for classification problems and for regression problems use a different algorithm, however, they both use the same approach for splitting the data into groups. That approach is regression decision trees. Even classification problems use regression decision trees. In regression decision trees, the final answer is a range of real numbers, this makes it’s relatively simple to split the data based on the remaining error at each step. Steps are taken to ensure the weak learner remain weak yet is still constructed in a greedy fashion. It is common to constrain the weak learners in sundry ways. Often, weak learners can be constrained using a maximum number of layers, nodes, splits or leaf nodes.
    # 
    # The third step is combing many weak learners in an additive fashion. Decision trees are added one at a time. A gradient descent procedure is used to minimize the loss when adding trees. That’s the gradient part of gradient boosters. Gradient descent optimization in the machine learning world is typically used to find the parameters associated with a single model that optimizes some loss function. In contrast, gradient boosters are meta-models consisting of multiple weak models whose output is added together to get an overall prediction. The gradient descent optimization occurs on the output of the model and not the parameters of the weak models.
    # 
    # see the picture in the link : https://qph.fs.quoracdn.net/main-qimg-5a75dffb1240830bff827f1238669fe0 
    # in the picture, we can see that gradient boosting adds sub-models incrementally to minimize a loss function. Earlier we said that gradient boosting involved three main steps. In the picture the weak learner being used is a decision tree. Secondly, the trees are added sequentially. Lastly, the error of the model is being reduced.
    # 
    # Once, the Model is fitted to the training set and the labels using the default parameters, we make a list of parameters which we feel may impact the performance such as the maximum depth of the tree max_depth. we must be careful not to over-do this parameter. if we have a large depth then the model will not generalize well and we may find our model overfitting to the training set as it "memorizes" the training set. we make sure the depth is in the range of 3 and 6. number of estimators or number of these weak learners are chosen in such a way that they lie between 50 and 300. this way the GridSearch can choose between lot of estimators and shallow individual trees or less number of estimators with deep trees. we set the learning rate to 1 or 2 to see if the factor of increase in the learning rate impacts the performance of the classifier. we leave the optimizer to be the default.
    # 
    # before we fit the grid search with the model parameters, we input the scoring method, i.e. Fscore with beta as 0.5 so that the Grid search rates the best classification parameters based on the value of this score. we proceed to fit the model with the training data. then we make predictions on the test data to see if the model accuracy and Fscore improves compared to the default model.
    # we may reiterate the parameter settings until we can explain the nature of the increase or decrease in the performance metrics. 
    # 
    # Thus, we have successfully trained and evaluated our model. 
    # Furthermore, we can find top features that the model prioritizes and uses and this is known as the best features. this is useful in order to reduce the features that are not quite relevant and improve the time taken to train the model and time taken to make predictions.However, it would come with an acceptable trade-off with the performance.
    # 
    # Reference:
    # https://www.quora.com/What-is-an-intuitive-explanation-of-Gradient-Boosting
    
    # ### Implementation: Model Tuning
    # Fine tune the chosen model. Use grid search (`GridSearchCV`) with at least one important parameter tuned with at least 3 different values. You will need to use the entire training set for this. In the code cell below, you will need to implement the following:
    # - Import [`sklearn.grid_search.GridSearchCV`](http://scikit-learn.org/0.17/modules/generated/sklearn.grid_search.GridSearchCV.html) and [`sklearn.metrics.make_scorer`](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html).
    # - Initialize the classifier you've chosen and store it in `clf`.
    #  - Set a `random_state` if one is available to the same state you set before.
    # - Create a dictionary of parameters you wish to tune for the chosen model.
    #  - Example: `parameters = {'parameter' : [list of values]}`.
    #  - **Note:** Avoid tuning the `max_features` parameter of your learner if that parameter is available!
    # - Use `make_scorer` to create an `fbeta_score` scoring object (with $\beta = 0.5$).
    # - Perform grid search on the classifier `clf` using the `'scorer'`, and store it in `grid_obj`.
    # - Fit the grid search object to the training data (`X_train`, `y_train`), and store it in `grid_fit`.
    # 
    # **Note:** Depending on the algorithm chosen and the parameter list, the following implementation may take some time to run!

    # Initialize the classifier
    clf = GradientBoostingClassifier(random_state=42)
    
    # Create the parameters list you wish to tune, using a dictionary if needed.
    parameters = {'max_depth': [5], #[3,4,5,6],
                  'n_estimators' : [100]#,150]
                  }
    # Make an fbeta_score scoring object using make_scorer()
    scorer =  make_scorer(fbeta_score, beta=0.5)
    
    # TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
    grid_obj = GridSearchCV(clf,param_grid=parameters,scoring=scorer)

    # Fit the grid search object to the training data and find the optimal parameters using fit()
    grid_fit = grid_obj.fit(X_train, y_train)

    # Get the estimator
    best_clf = grid_fit.best_estimator_

    # Make predictions using the unoptimized and model
    predictions = (clf.fit(X_train, y_train)).predict(X_test)
    best_predictions = best_clf.predict(X_test)

    # Report the before-and-afterscores
    print("Unoptimized model\n------")
    print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
    print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
    print("\nOptimized Model\n------")
    print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
    print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
    
    # ### Question 5 - Final Model Evaluation
    # 
    # * What is your optimized model's accuracy and F-score on the testing data? 
    # * Are these scores better or worse than the unoptimized model? 
    # * How do the results from your optimized model compare to the naive predictor benchmarks you found earlier in **Question 1**?_  
    # 
    # **Note:** Fill in the table below with your results, and then provide discussion in the **Answer** box.
    
    # #### Results:
    # 
    # |     Metric     | Benchmark Model   |Unoptimized Model | Optimized Model |
    # | :------------: | :---------------: |:---------------: | :-------------: | 
    # | Accuracy Score |        24.78%     |       86.3%      |       86.9%     |
    # | F-score        |        29.17%     |       73.95%     |       74.8%     |
    # 

    # **Answer:**
    # 
    # the optimized model accuracy and the Fscore have improved as seen in the above table.
    # 
    # the optimized as well as the unoptimized models were better compared with the naive predictor values.
    # 
    # It was found from the Grid search that the model with max_depth of 5, 100 estimators and learning rate of 1 was found to be optimal.
    # This makes sense, the depth of the tree should be as low as possible in order to not overfit to the training set and computational needs is proportional to the number of estimators.
    # 

    # ----
    # ## Feature Importance
    # 
    # An important task when performing supervised learning on a dataset like the census data we study here is determining which features provide the most predictive power. By focusing on the relationship between only a few crucial features and the target label we simplify our understanding of the phenomenon, which is most always a useful thing to do. In the case of this project, that means we wish to identify a small number of features that most strongly predict whether an individual makes at most or more than \$50,000.
    # 
    # Choose a scikit-learn classifier (e.g., adaboost, random forests) that has a `feature_importance_` attribute, which is a function that ranks the importance of features according to the chosen classifier.  In the next python cell fit this classifier to training set and use this attribute to determine the top 5 most important features for the census dataset.

    # ### Question 6 - Feature Relevance Observation
    # When **Exploring the Data**, it was shown there are thirteen available features for each individual on record in the census data. Of these thirteen records, which five features do you believe to be most important for prediction, and in what order would you rank them and why?

    # **Answer:**
    # 
    # - Education qualification/Education_num : Since this field was found to impact the type of job people ended up with, it ultimately would impact whether he/she can make more than 50k an year
    # 
    # - Age/Occupation : Age has impact on the type of occupation like higher age groups tend to earn better due to experience/senior positions. 
    # 
    # - Marital status: this needs another feature like Age to make it effective. this looks to be correlated as cluster of age groups are single, married and divorced/widowed. Though i am not sure of 
    #   the source of the data, in some cases come tax benefits(EU countries and also US) on income for married people which can also positively impact the income of the individual.
    # 
    # - hours per week : overtime for white collar jobs and daily wage workers working high hours per week in a country like the USA could make >50K a year assuming a middle aged person. 
    # 
    # - country : pay scales in the countries against cost of living and economic conditions can impact salaries.
    # 

    # ### Implementation - Extracting Feature Importance
    # Choose a `scikit-learn` supervised learning algorithm that has a `feature_importance_` attribute availble for it. This attribute is a function that ranks the importance of each feature when making predictions based on the chosen algorithm.
    # 
    # In the code cell below, you will need to implement the following:
    #  - Import a supervised learning model from sklearn if it is different from the three used earlier.
    #  - Train the supervised model on the entire training set.
    #  - Extract the feature importances using `'.feature_importances_'`.

    # Import a supervised learning model that has 'feature_importances_'
    # the best model has this already
    
    # Train the supervised model on the training set using .fit(X_train, y_train)
    model = grid_fit.best_estimator_.fit(X_train, y_train)
    
    # Extract the feature importances using .feature_importances_ 
    importances = grid_fit.best_estimator_.feature_importances_

    # Plot
    vs.feature_plot(importances, X_train, y_train)


    # ### Question 7 - Extracting Feature Importance
    # 
    # Observe the visualization created above which displays the five most relevant features for predicting if an individual makes at most or above \$50,000.  
    # * How do these five features compare to the five features you discussed in **Question 6**?
    # * If you were close to the same answer, how does this visualization confirm your thoughts? 
    # * If you were not close, why do you think these features are more relevant?
    
    # **Answer:**
    # 
    # - Some of the items matched my expectation such as age ,Marital status and education_num but, maybe not in the order that I 
    #   expected.
    #   
    # 
    # - When I did the exploratory data analysis, I was going through the capital loss and gain features. Though, I do stock trading 
    #   and I am familiar with this terminology, I did not expect it to have a huge impact as these are usually sporadic income 
    #   sources. However, it could be that these income sources may be high enough to make the person observe huge 
    #   profits(ocassional, but can happen) driving net income >50K.
    # 
    # 
    # - I expected the Education/Age to have a stronger correlation compared with marital status but, it does not seem so. It could 
    #   be that the marital status allowed for more tax benefits in the mid-age groups thus, these groups (~40 to ~60 yrs) probably 
    #   enjoyed a reduced tax on their large paychecks. It could have also made the age column a bit less relevant as it could 
    #   in a way redundantly explain the cluster of age groups that are living with a spouse and not.

    # ### Feature Selection
    # How does a model perform if we only use a subset of all the available features in the data? With less features required to train, the expectation is that training and prediction time is much lower — at the cost of performance metrics. From the visualization above, we see that the top five most important features contribute more than half of the importance of **all** features present in the data. This hints that we can attempt to *reduce the feature space* and simplify the information required for the model to learn. The code cell below will use the same optimized model you found earlier, and train it on the same training set *with only the top five important features*. 

    # Reduce the feature space
    X_train_reduced = X_train[X_train.columns.values[(np.argsort(importances)[::-1])[:5]]]
    X_test_reduced = X_test[X_test.columns.values[(np.argsort(importances)[::-1])[:5]]]

    start = time() # Get start time
    # Train on the "best" model found from grid search earlier on full data to calculate time
    clf = (clone(best_clf)).fit(X_train, y_train)
    end = time() # Get end time
    #Calculate the training time
    deltaT = abs(end - start)

    start = time() # Get start time
    # Train on the "best" model found from grid search earlier
    clf = (clone(best_clf)).fit(X_train_reduced, y_train)
    end = time() # Get end time
    #Calculate the training time
    deltaT_new = abs(end - start)

    # Make new predictions
    reduced_predictions = clf.predict(X_test_reduced)

    # Report scores from the final model using both versions of data
    print("Final Model trained on full data\n------")
    print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
    print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta = 0.5)))
    print("Time taken to train :  {:.4f}".format(deltaT))
    print("\nFinal Model trained on reduced data\n------")
    print("Accuracy on testing data: {:.4f}".format(accuracy_score(y_test, reduced_predictions)))
    print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, reduced_predictions, beta = 0.5)))
    print("Time taken to train :  {:.4f}".format(deltaT_new))

    # ### Question 8 - Effects of Feature Selection
    # 
    # * How does the final model's F-score and accuracy score on the reduced data using only five features compare to those same scores when all features are used?
    # * If training time was a factor, would you consider using the reduced data as your training set?

    # **Answer:**
    # 
    # - When the feature set was reduced the accuracy reduced by ~1% and the Fscore was reduced by ~2.5%.
    # - If it was a factor then yes, I would consider the reduced feature set because It is observed that the reduced feature set model is ~11.5 times faster in training. If time is the most important aspect and that the trade-off of this performance is preferable for having feature sets that best explain the data, then the reduced data set model makes more sense to not just work as a classification model but, also, as a model that can replace our benchmark naive predictor. This opens avenues to exhaustively tune other hyperparameters in order to improve this model further if possible. 
    #   
    # ## Future Outlook
    # 
    # However, if i had the choice to use Histogram Gradient Boosting Classifer then I think we may not require this trade-off. It would solve the problem of the time taken to train. Again, i did not want to use an experimental classifier which is not yet certified as stable. Just if you are interested you can look at the training time differences between default classifiers of gradient boost (\~15s) and histogram gradient boost(\~5s) which means the HistGradientBoostingClassifier is (~3) times more efficient than the Gradient Boosting Classifier.
    # It also performs a bit better than the gradient boosting classifier   

    # This is an optional cell. It need not be executed. 
    try:
        # explicitly require this experimental feature
        from sklearn.experimental import enable_hist_gradient_boosting  # noqa
        # now you can import normally from ensemble
        from sklearn.ensemble import HistGradientBoostingClassifier
        clf_A = HistGradientBoostingClassifier()
        clf_B = GradientBoostingClassifier()
        # Calculate the number of samples for 1%, 10%, and 100% of the training data
        samples_100 = len(X_train)
        samples_10 = int(samples_100*0.1)
        samples_1 = int(samples_100*0.01)
        # Collect results on the learners
        results = {}
        for clf in [clf_A, clf_B]:
            clf_name = clf.__class__.__name__
            results[clf_name] = {}
            for i, samples in enumerate([samples_1, samples_10, samples_100]):
                results[clf_name][i] =             train_predict(clf, samples, X_train, y_train, X_test, y_test)

        # Run metrics visualization for the three supervised learning models chosen
        vs.evaluate(results, accuracy, fscore)
        for i in results:
            print("Training_time : "+i+' : '+str(results[i][2]['train_time']))
            print("Pred_time : "+i+' : '+str(results[i][2]['pred_time']))
    except ImportError:
        print('this feature wont work with your current version of sklearn')
    return accuracy_score(y_test, reduced_predictions), fbeta_score(y_test, reduced_predictions, beta = 0.5)


