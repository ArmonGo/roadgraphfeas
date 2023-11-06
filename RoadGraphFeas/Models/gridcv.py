from sklearn.model_selection import GridSearchCV
import copy

def cross_val(estimator, grid_params, X_train, y_train, 
                scoring = ["neg_mean_absolute_error", "neg_mean_squared_error", "neg_median_absolute_error", "explained_variance"], 
                standard="neg_median_absolute_error",
                cv = 5, return_train_score = False):
    grid_model = GridSearchCV(estimator=estimator,
                            param_grid=grid_params,
                            scoring=scoring,
                            cv=cv,
                            refit= standard, 
                            return_train_score = return_train_score)
    grid_model.fit(X_train, y_train)
    return grid_model


def return_best_params(model_name, grid_model):
    best_param = copy.deepcopy(grid_model.best_params_)
    if model_name == 'GWR':
        try:
            best_param["bw"] =  grid_model.best_estimator_.bw
        except:
            print("not available to record bw")
    if model_name.split('-')[0] == 'Kriging' and len(model_name.split('-')) >1:
        try:
            del best_param['outer_params']['regression_model']
        except:
            print("regression model params not available")
    return best_param

def param_format(params):
    param_use = copy.deepcopy(params)
    for k,v in param_use.items():
        param_use[k]= [v]
    return param_use

def log(model_name, grid_model, repeat_cnt, attr_choice):
    records = {}
    records["cv_res"] = copy.deepcopy(grid_model.cv_results_)
    records["model_name"] = model_name
    records["repeat_cnt"] = repeat_cnt
    records["attr_choice"] = attr_choice
    records["best_estimator"] = grid_model.best_estimator_
    return records
