# Restaurant-Visitor-Forecasting

Restaurants need to know how many customers to expect each day to effectively purchase ingredients and schedule staff members. This forecast isn't easy to make because many unpredictable factors affect restaurant attendance, like weather and local competition.
In this problem, we're challenged to reservation and visitation data to predict the total number of visitors to a restaurant for future dates. This information will help restaurants be much more efficient and allow them to focus on creating an enjoyable dining experience for their customers.
[Link](https://www.kaggle.com/c/restaurant-visitor-forecasting/overview) to the Kaggle competition.

__Refer to this [report](RVF_report.pdf) for a detailed analysis of the above machine learning problem and insights into our approach.__

- Dataset (along with the cleaned data `fe_train` and `fe_test`) has been maintained in the [`dataset` folder](dataset).
- [`RVF-complete`](RVF-complete) is the complete ipynb notebook(contains EDA, feature engineering, model building and submissions).


### Install Requirements
```bash
pip3 install -r requirements.txt
```

### Exploratory Data Analysis
`EDA.ipynb` contains a detailed EDA.

### For Feature Engineering
```
python3 feature_engineering.py
```
This will generate `fe_train.py` and `fe_test.py`, the cleaned train and test files respectively.

### Build models (as required)
Available Models:
- XGBRegressor 
- LGBMRegressor
- KNN

To run:
```
python3 model.py
```
This will generate `submission.csv` in the format specified by the competition.

