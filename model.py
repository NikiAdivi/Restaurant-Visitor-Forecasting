from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error as mse
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor

class Model:
    
    def __init__(self):
        self.train = pd.read_csv('./fe_train.csv')
        self.test = pd.read_csv('./fe_test.csv')
        
        self.x_train, self.y_train, self.x_valid, self.y_valid = None, None, None, None
    
    def splitData(self):      
        y = self.train['visitors']
        X = self.train.drop(columns = "visitors",axis = 1)
        
        split = 179000 #For a time-series split
        self.x_train, self.y_train, self.x_valid, self.y_valid = X[:split], y[:split], X[split:], y[split:]
    
    def RMSLE(self, actual, pred):
        return mse(actual, pred)**0.5

    def xgboost(self):
        col = [c for c in self.test if c not in ['id', 'air_store_id', 'visit_date','visitors', 'longitude', 'latitude']]
        xgb_m = XGBRegressor(
            max_depth=8,
            learning_rate=0.01,
            n_estimators=500,
            gamma=0,
            min_child_weight=1,
            subsample=1,
            colsample_bytree=1,
            scale_pos_weight=1,
            seed=24,verbosity = 1)
        
        return xgb_m,col
    
    def lightGBM(self):
        col = [c for c in self.test if c not in ['id', 'air_store_id', 'visit_date','visitors', 'longitude', 'latitude']]
        params = {
            'learning_rate': 0.02,
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'sub_feature': 0.5417895574997428,
            'num_leaves': 105,
            'min_data': 42,
            'min_hessian': 1,
            'verbose': -1,
            'n_estimators': 500,
        }

        model = LGBMRegressor(**params)
        return model,col
    
    def KNNs(self):
        col = [c for c in self.x_train if c not in ['id', 'air_store_id', 'visit_date','visitors', 'latitude', 'longitude','size', 'reserve_visitors_x', 'reserve_visitors_y', 'time_diff_x', 'time_diff_y']]
        knn_m = KNeighborsRegressor(n_neighbors=170)
        return knn_m, col 
    
    def fitData(self, model, col):
        model.fit(self.x_train[col], np.log1p(self.y_train.values))
        
    def predictValidationSet(self, model, col):
        y_pred_train = model.predict(self.x_train[col])
        y_pred = model.predict(self.x_valid[col])

        print('RMSE LGBRegressor train: ', self.RMSLE(np.log1p(self.y_train.values), y_pred_train))
        print('RMSE LGBRegressor test: ', self.RMSLE(np.log1p(self.y_valid.values), y_pred))

    def predictTest(self, model, col):
        test_pred = model.predict(self.test[col])
        
        self.test["id"] = pd.DataFrame(self.test["air_store_id"] + "_" + self.test['visit_date'].astype(str))
        self.test["visitors"] = np.expm1(test_pred)

    def makeSubmission(self, model):
        self.test[['id','visitors']].to_csv('submission.csv', index=False)

def main():
    model_obj = Model()
    model_obj.splitData()
    
    model,col = model_obj.lightGBM()
    
    model_obj.fitData(model, col)
    model_obj.predictTest(model, col)
    model_obj.predictValidationSet(model, col)
    model_obj.makeSubmission(model)
    
if __name__ == '__main__':
    main()