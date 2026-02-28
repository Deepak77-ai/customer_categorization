
# Customer Personality Segmentation

## Problem statement

In this data science project, you will build a machine learning system which will be able predict the personality of the customer using machine learning algorithms. This project will be very usefull for malls, various stores and companies which are product based. Based on customer's personal details and purchase details, we can cluster them and we can predict the customer's cluster number using classification techniques.

## Solution Proposed

Now the question is how to dynamically predict the cluster of the customer ?. One of the approaches which we can use of machine learning approach, where we can cluster the customer based on the details we have and predict the cluster type based on the domain knowledge and leverage previous customer data to predict the cluster.

Dataset used
 <html>
<a href="https://github.com/entbappy/Branching-tutorial/blob/master/marketing_campaign.zip"> Dataset Link</a>
</html>



## Tech Stack Used

1. Python
2. Machine learning algorithms
3. stream-lit


## How to run

go to my live stream-lit app --> https://customercategorization-r6fh22kyxtv7zyctvtteay.streamlit.app/

Step 1. Cloning the repository.

```

https://github.com/Deepak77-ai/customer_categorization/
```

Step 2. Create a conda environment.

```

conda create --prefix venv python=3.7 -y

```

```

conda activate venv/

```

Step 3. Install the requirements

```

pip install -r requirements.txt

```


Step 4. Run the application server 

```

"python app.py" or "streamlit run app.py"

```

## Project Architecture -

![WhatsApp Image 2022-09-22 at 15 29 19](https://user-images.githubusercontent.com/71321529/192722336-54016f79-89ef-4c8c-9d71-a6e91ebab03f.jpeg)


## Models Used

* [K-Means](https://www.javatpoint.com/k-means-clustering-algorithm-in-machine-learning)
* [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

From these above models after hyperparameter optimization we selected these two models which were K-Means for clustering and Logistic Regression for classification and used the following in Pipeline.

* GridSearchCV is used for Hyperparameter Optimization in the pipeline.


## Conclusion

- This Project can be used in real-life by Users.


