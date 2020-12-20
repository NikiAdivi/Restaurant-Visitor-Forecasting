import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class FeatureEngineering:
    
    def __init__(self):
        self.air_visits = pd.read_csv("../input/restaurant-visitor-forecasting/train.csv")
        self.air_reservations = pd.read_csv("../input/restaurant-visitor-forecasting/MetaData/MetaData/air_reserve.csv")
        self.hpg_reservations = pd.read_csv("../input/restaurant-visitor-forecasting/MetaData/MetaData/hpg_reserve.csv")
        self.air_stores = pd.read_csv("../input/restaurant-visitor-forecasting/MetaData/MetaData/air_store_info.csv")
        self.date_data = pd.read_csv("/kaggle/input/restaurant-visitor-forecasting/MetaData/MetaData/date_info.csv")
        self.test = pd.read_csv("/kaggle/input/restaurant-visitor-forecasting/test.csv")
        self.store_IDs = pd.read_csv("../input/restaurant-visitor-forecasting/MetaData/MetaData/store_id_relation.csv")
        
    def getProcessedData(self):
        return self.air_visits, self.test
    
    
    def remove_2016_Data(self):
        #From index 47699, i.e. dates after July 1, 2016 follow the periodic trend.  
        self.air_visits = self.air_visits.sort_values('visit_date').reset_index(drop = True).iloc[47699:,:]
     
    def add_next_holiday_flg(self):
        #The dates are in order, so the next_holiday_flg will just be the holiday_flg column shifted 1 down.
        hol_date = self.date_data["holiday_flg"].copy()
        #Drop the first row of the holiday_flg column
        hol_date.drop(hol_date.head(1).index, inplace = True) 
        #Add a row at the end, For convenience, the flg is set to 0.
        hol_date.loc[len(hol_date)+1] = 0 
        
        self.date_data["next_day_holiday_flg"] = hol_date.reset_index(drop = True)
        
        self.air_visits = self.air_visits.merge(self.date_data, left_on="visit_date", right_on="calendar_date").drop("calendar_date", axis = 1)
        self.test = self.test.merge(self.date_data, left_on="visit_date", right_on="calendar_date").drop("calendar_date", axis = 1)
        
    def mergeStoreInfo(self):
        self.air_visits = self.air_visits.merge(self.air_stores, on="air_store_id")
        self.test = self.test.merge(self.air_stores, on="air_store_id")
        
    #Size refers to the no. of stores of particular genre in a particular area. Accounts for local competition
    def addSizeFeature(self):
        air_competition = self.air_stores.groupby(["air_genre_name", "air_area_name"], as_index = False).size()
        
        self.air_visits = self.air_visits.merge(air_competition, on = ["air_genre_name", "air_area_name"])
        self.test = self.test.merge(air_competition, on = ["air_genre_name", "air_area_name"])
        
    def parseDate(self):
        self.air_visits['visit_date'] = pd.to_datetime(self.air_visits['visit_date'],format='%Y-%m-%d')
        self.air_visits['month'] = self.air_visits['visit_date'].dt.month
        self.air_visits['year'] = self.air_visits['visit_date'].dt.year
        self.air_visits['dow'] = self.air_visits['visit_date'].dt.dayofweek
        
        self.test['visit_date'] = pd.to_datetime(self.test['visit_date'],format='%Y-%m-%d')
        self.test['month'] = self.test['visit_date'].dt.month
        self.test['year'] = self.test['visit_date'].dt.year
        self.test['dow'] = self.test['visit_date'].dt.dayofweek
        
    def create_area_genre_encoding(self):
        #Label encoding based on the no. of visitors
        
        col = "air_area_name"
        area_encode = self.air_visits.groupby(col, as_index = False)['visitors'].sum().sort_values(by = "visitors").reset_index(drop = True)
        area_encode["area_index_col"] = range(1, len(area_encode) + 1)
        area_encode.drop(columns = "visitors", inplace = True)
        
        col = "air_genre_name"
        genre_encode = self.air_visits.groupby(col, as_index = False)['visitors'].sum().sort_values(by = "visitors").reset_index(drop = True)
        genre_encode["genre_index_col"] = range(1, len(genre_encode) + 1)
        genre_encode.drop(columns = "visitors", inplace = True)
        
        return area_encode, genre_encode
        
    def encodeAreaGenre(self):
        area_encode, genre_encode = self.create_area_genre_encoding()
        
        self.air_visits = self.air_visits.merge(area_encode, on = "air_area_name").drop(columns = "air_area_name")
        self.air_visits = self.air_visits.merge(genre_encode, on = "air_genre_name").drop(columns = "air_genre_name")   

        self.test = self.test.merge(area_encode, on = "air_area_name").drop(columns = "air_area_name")
        self.test = self.test.merge(genre_encode, on = "air_genre_name").drop(columns = "air_genre_name")     
    
    #Adds min mean max median count_observations for all the unique stores
    def addAggregateFunctions(self):
        #Find all unique stores from the test data.
        unique_stores = self.test['air_store_id'].unique()
        stores = pd.concat([pd.DataFrame({'air_store_id': unique_stores, 'dow': [i]*len(unique_stores)}) for i in range(7)], axis=0, ignore_index=True).reset_index(drop=True)

        tmp = self.air_visits.groupby(['air_store_id','dow'], as_index=False)['visitors'].min().rename(columns={'visitors':'min_visitors'})
        stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
        tmp = self.air_visits.groupby(['air_store_id','dow'], as_index=False)['visitors'].mean().rename(columns={'visitors':'mean_visitors'})
        stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
        tmp = self.air_visits.groupby(['air_store_id','dow'], as_index=False)['visitors'].median().rename(columns={'visitors':'median_visitors'})
        stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
        tmp = self.air_visits.groupby(['air_store_id','dow'], as_index=False)['visitors'].max().rename(columns={'visitors':'max_visitors'})
        stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow'])
        tmp = self.air_visits.groupby(['air_store_id','dow'], as_index=False)['visitors'].count().rename(columns={'visitors':'count_observations'})
        stores = pd.merge(stores, tmp, how='left', on=['air_store_id','dow']) 

        self.air_visits = pd.merge(self.air_visits, stores, how='left', on=['air_store_id','dow'])
        self.test = pd.merge(self.test, stores, how='left', on=['air_store_id','dow'])
        
        
    def oneHotEncode_dayOfWeek(self):
        self.air_visits = pd.get_dummies(data = self.air_visits, columns = ["dow"], prefix = ["dow"])
        self.air_visits.drop(columns = "day_of_week", inplace = True)
        
        self.test = pd.get_dummies(data = self.test, columns = ["dow"], prefix = ["dow"])
        self.test.drop(columns = "day_of_week", inplace = True)
        
    def preprocess_airReservations(self):
        self.air_reservations['visit_datetime'] = pd.to_datetime(self.air_reservations['visit_datetime'])
        self.air_reservations['visit_date'] = self.air_reservations['visit_datetime'].dt.date
        self.air_reservations['reserve_datetime'] = pd.to_datetime(self.air_reservations['reserve_datetime'])
        self.air_reservations['time_diff'] = self.air_reservations.apply(lambda x: (x['visit_datetime'] - x['reserve_datetime']).days, axis=1)
        self.air_reservations = self.air_reservations.groupby(['air_store_id', 'visit_date'], as_index=False)[['time_diff', 'reserve_visitors']].sum()

    def mergeStoreRelation(self):
        self.hpg_reservations = pd.merge(self.hpg_reservations, self.store_IDs, how='inner', on=['hpg_store_id'])
    
    def preprocess_hpgReservations(self):
        self.mergeStoreRelation()
        self.hpg_reservations['visit_datetime'] = pd.to_datetime(self.hpg_reservations['visit_datetime'])
        self.hpg_reservations['visit_date'] = self.hpg_reservations['visit_datetime'].dt.date
        self.hpg_reservations['reserve_datetime'] = pd.to_datetime(self.hpg_reservations['reserve_datetime'])
        self.hpg_reservations['time_diff'] = self.hpg_reservations.apply(lambda x: (x['visit_datetime'] - x['reserve_datetime']).days, axis=1)
        self.hpg_reservations = self.hpg_reservations.groupby(['air_store_id', 'visit_date'], as_index=False)[['time_diff', 'reserve_visitors']].sum()
        
    def mergeReservationData(self):
        #First merge hpg reservations with store_id_relation(to get corresponding air_store_ids)
        self.air_visits['visit_date'] = self.air_visits['visit_date'].dt.date
        self.test['visit_date'] = self.test['visit_date'].dt.date
        
        self.air_visits = pd.merge(self.air_visits, self.air_reservations, how='left', on=['air_store_id','visit_date'])
        self.air_visits = pd.merge(self.air_visits, self.hpg_reservations, how='left', on=['air_store_id','visit_date'])
        
        self.test = pd.merge(self.test, self.air_reservations, how='left', on=['air_store_id','visit_date'])
        self.test = pd.merge(self.test, self.hpg_reservations, how='left', on=['air_store_id','visit_date'])
      
    def write_fe_csv(self):
        self.air_visits.to_csv('./fe_train.csv', index=False)
        self.test.to_csv('./fe_test.csv', index=False)
        
def main(fe):
    fe.remove_2016_Data()
    fe.add_next_holiday_flg()
    fe.mergeStoreInfo()
    fe.addSizeFeature()
    fe.parseDate()
    fe.encodeAreaGenre()
    fe.addAggregateFunctions()
    fe.oneHotEncode_dayOfWeek()
    fe.preprocess_airReservations()
    fe.preprocess_hpgReservations()
    fe.mergeReservationData()
    fe.write_fe_csv()

        
if __name__ == '__main__':
    fe = FeatureEngineering()
    main(fe)
    data, test = fe.getProcessedData()

    print(data.head())
    print(data.columns)
    print(test.head())