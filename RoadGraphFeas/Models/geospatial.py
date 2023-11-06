# models that includes geo autocorrelation 
from pykrige.rk import RegressionKriging, Krige
import gpboost as gpb
import os
import numpy as np
from mgwr.gwr import GWR as Mod_GWR
from mgwr.sel_bw import Sel_BW
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import  check_is_fitted



class KrigeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, outer_params=None, inner_params=None):
        self.outer_params = outer_params
        self.inner_params = inner_params

    def split_coords(self, X, cols = ["x", "y"]):
        coords = X[cols]
        X=X.drop(cols, axis=1)
        return np.array(X), np.array(coords)
    
    def fit(self, X, y):
        X, coords = self.split_coords(X)
        self.outer_params['regression_model'].set_params(**self.inner_params)
        self.pykridge_model_ = RegressionKriging(**self.outer_params)
        self.pykridge_model_.fit(X,coords,y)
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X, coords = self.split_coords(X)
        predictions = self.pykridge_model_.predict(X, coords)
        return predictions
    
class OrdinaryKriging(BaseEstimator, RegressorMixin):
    def __init__(self, nlags=5, variogram_model='gaussian'):
        self.nlags = nlags
        self.variogram_model = variogram_model

    def fit(self, coords, y):
        self.model_ = Krige(nlags=self.nlags, variogram_model = self.variogram_model)
        self.model_.fit(np.array(coords),np.array(y))
        return self
    
    def predict(self, coords):
        predictions = self.model_.predict(np.array(coords))
        return predictions


class GPBoost(BaseEstimator, RegressorMixin):
    def __init__(self, outer_params = None, inner_params  = None):
        self.outer_params = outer_params
        self.inner_params = inner_params
    
    def split_coords(self, X, cols = ["x", "y"]):
        coords = X[cols]
        X=X.drop(cols, axis=1)
        return np.array(X), np.array(coords)
    
    def fit(self, X, y):
        assert isinstance(self.outer_params, dict)
        self.cov_function = self.outer_params["cov_function"]
       
        print("self.cov_function", self.cov_function)
        X, coords  = self.split_coords(X)
        data_train = gpb.Dataset(X, y)
        gpm_ = gpb.GPModel(cov_function = self.cov_function,
                             gp_coords=coords, 
                             gp_approx= "vecchia" # sample 
                             )
        self.gpbst_ = gpb.train(params=self.inner_params, train_set=data_train,
                gp_model=gpm_, num_boost_round=30) # default 30 to reduce the calculation time 
        return self
    
    def predict(self, X):
        check_is_fitted(self)
        X, coords  = self.split_coords(X)
        return self.gpbst_.predict(data=X, gp_coords_pred=coords, 
                        predict_var=False, pred_latent=False)['response_mean']
                        
#------------------------- Geospatial Random Forests

class GWR(BaseEstimator, RegressorMixin):
    def __init__(self, constant=True, kernel='gaussian', bw=None):
        self.constant = constant
        self.kernel = kernel
        self.bw = bw
    def split_coords(self, X, cols = ["x", "y"]):
        coords = X[cols]
        X=X.drop(cols, axis=1)
        rng = np.random.RandomState(23)
        rand = rng.randn(X.shape[0], X.shape[1])/10000 # to prevent a singular matrix
        return np.array(X)+rand, np.array(coords)
    
    def fit(self, X, y):
        X, coords = self.split_coords(X)
        if self.bw is None:
            self.bw = Sel_BW(coords, y.reshape((-1, 1)), X).search()
        self.model_ = Mod_GWR(coords, y.reshape((-1, 1)), X, self.bw, constant=self.constant, kernel=self.kernel)
        gwr_results = self.model_.fit()
        self.scale = gwr_results.scale
        self.residuals = gwr_results.resid_response 
        return self

    def predict(self, X):
        check_is_fitted(self)
        X, coords = self.split_coords(X)
        pred = self.model_.predict(coords, X, exog_scale=self.scale, exog_resid=self.residuals
               ).predictions
        return pred

