# Restaurant-Visitor-Forecasting

Restaurants need to know how many customers to expect each day to effectively purchase ingredients and schedule staff members. This forecast isn't easy to make because many unpredictable factors affect restaurant attendance, like weather and local competition.
In this problem, we're challenged to reservation and visitation data to predict the total number of visitors to a restaurant for future dates. This information will help restaurants be much more efficient and allow them to focus on creating an enjoyable dining experience for their customers.

__Refer to this [report](RVF_report.pdf) for a detailed analysis of the above machine learning problem and insights into our approach.__

Dataset (along with the cleaned data) has been maintained in the [`dataset` folder](datset)

### Install Requirements
```bash
pip3 install -r requirements.txt
```

### For Feature Engineering
```
python3 feature_engineering.py
```

### Build models (as required)
Available Models:
- XGBRegressor 
- LGBMRegressor
- KNN

To run:
```
python3 model.py
```

### Exploratory Data Analysis
Open the `EDA.ipynb` for a detailed EDA
