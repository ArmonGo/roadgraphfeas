# import all the package 
# graph 
from GraphConstruction.dataloader import *
from GraphConstruction.overpassloader import OverpassRoadLoader
from GraphConstruction.roadgraph import *
from GraphConstruction.transformer import *
# features
from StreamingFeasLoader.streamingdata import StreamDataLoader
# data split
from TrainTestSet.traintestloader import TrainTestLoader
# models
# from Models.direct import DirectModelLoader
from Models.geospatial import *
from Models.gridcv import *
# other packages
import pandas as pd
import copy
from sklearn.model_selection import ParameterGrid
from datetime import date
import sys
import time 
import contextlib
import dill
import shutil
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge 
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# more config less internal logic
M_REGRESSOR = {  'Lr_Ridge' : Ridge,
                'RF' : RandomForestRegressor, 
                'Lightgbm' : LGBMRegressor, 
                "SVM" : SVR,  
                "KNNuni" : KNeighborsRegressor, 
                "KNNdis" : KNeighborsRegressor, 
                'GPBoost-Lightgbm' : GPBoost, 
                "GWR" :  GWR, 
                "Kriging" : OrdinaryKriging, 
                "Kriging-KNNuni" : KrigeRegressor, 
                "Kriging-Lr_Ridge" : KrigeRegressor, 
                "Kriging-RF" : KrigeRegressor, 
                "Kriging-Lightgbm" : KrigeRegressor, 
                "Kriging-SVM" : KrigeRegressor
                }


class ParamsController:
    def __init__(self, params_grid, m_regressor):
        """initialize the best params"""
        self.params_grid = params_grid
        self.m_regressor = m_regressor
        # initialize the best params
        self.best_params_set = {}
        for k in params_grid.keys():
            self.best_params_set[k] = {}
        self.model_ls = list(self.params_grid.keys())

    def fetch_estimator_params(self, o_m_name, best):
        estimator = self.m_regressor[o_m_name]() # estimator always renew
        if not best: # do not use the best params
            param_base = copy.deepcopy(self.params_grid[o_m_name]["o_m_params"]) # as the base
            if self.params_grid[o_m_name]["i_m_renew"]: # if update the inner model
                    param_base["regression_model"] = [self.m_regressor[self.params_grid[o_m_name]["i_m_name"]]()] # add the regressor
            if self.params_grid[o_m_name]["i_m_name"] is not None:
                param = {"outer_params" : list(ParameterGrid([param_base]))} # outer model params
                param['inner_params'] = list(ParameterGrid([ {k : [v] for k, v in self.best_params_set[self.params_grid[o_m_name]["i_m_name"]].items()}])) # take the best params of inner model
            else:
                param = self.params_grid[o_m_name]["o_m_params"]
        else:
            param = self.best_params_set[o_m_name]
            if self.params_grid[o_m_name]["i_m_renew"]: 
                param["outer_params"]["regression_model"] = self.m_regressor[self.params_grid[o_m_name]["i_m_name"]]() # update the regression model but keep the params
        return estimator, param
            
    def update(self, o_m_name, best_params):
        self.best_params_set[o_m_name] = best_params

    
def write(item, save_path, f_name, file_type):
    if file_type == "csv":
        item.to_csv(save_path + f_name)
    elif file_type == "pickle":
        with open(save_path + f_name, 'wb') as fp:
            dill.dump(item, fp)
    return f_name + " has been written..."

def read(save_path, f_name, file_type):
    if file_type == "csv":
        item = pd.read_csv(save_path+f_name, index_col=0)
    elif file_type == "pickle":
        with open(save_path+f_name, 'rb') as fp:
            item = dill.load(fp)   
    return item 

def check(save_path, f_name):
     f_ls =  os.listdir(save_path)
     if f_name in f_ls:
          return False # If run the model again 
     else:
          return True

def progressbar(it, t_start, prefix="", size=60, out=sys.stdout): # Python3.3+
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print("{}[{}{}] {}/{}".format(prefix + " time_consume " + str(int(time.time() - t_start)) + "-" + str(it[j-1]) +":", "#"*x, "."*(size-x), j, count), 
                end='\r', file=out, flush=True)
    # show(0)
    t_start = time.time() 
    for i, item in enumerate(it):
        yield item
        nm = item
        taken = int(time.time() - t_start)
        t_start = time.time()
        print(f"{prefix} | {nm} took: {taken}", file=out)
    print("\n", flush=True, file=out)


class ExperimentLoader:
    def __init__(self, location_centriod, radius, province, save_path, data_path, country,
                  subg_radius= 150, graph_weight="all", # radius need to be a dictionary
                  grid_params={},
                  repeat=10, scaler_choice="minmax",  
                  developing_size = 0.3, 
                  evaluation_size=0.6, test_size=0.1,
                  normaliza_target = True
                  ):
          self.location_centriod = location_centriod
          self.radius = radius
          self.province = province
          self.save_path = save_path
          self.data_path = data_path
          self.graph_weight = graph_weight
          self.subg_radius = subg_radius
          self.repeat = repeat
          self.scaler_choice = scaler_choice
          self.country = country
          self.developing_size = developing_size
          self.evaluation_size = evaluation_size
          self.test_size = test_size
          self.normaliza_target = normaliza_target
          self.dataloader = self.data_loader()
          # prepare the data
          self.__data_preparation__()
          self.paramgenrator = ParamsController(grid_params, M_REGRESSOR) # write in the py file
          
    
    def data_loader(self):
        
        if self.country == "China":
            return EXampleBeijingLoader # using the example case for beijing 
        else:
            return "country name is not in the list!"
    
    def features_extract(self):
        feature_loader = StreamDataLoader(self.location_centriod, self.radius, self.subg_radius,self.dataloader, 
                 self.data_path, self.province, weight=self.graph_weight, features_choice = "all")
        df = feature_loader.load()
        features_dict = feature_loader.feas_ls_log(df)
        print("features done!")
        return df, features_dict
    
    def train_test_split(self, df):
        dataloader = TrainTestLoader(self.repeat, self.developing_size, 
                 self.evaluation_size, self.test_size, self.scaler_choice, 
                                     self.features_dict, self.normaliza_target) 
        dataset = dataloader.simple_train_test(df) # split only once!!
        self.dataset = dataset
        

    def __data_preparation__(self):
        if check(self.save_path,  "dis_train_test_data.csv") and check(self.save_path,  "ori_dis_train_test_data.csv"): # set different graph feas
            df, features_dict = self.features_extract()
            df["x"] = list(map(lambda x : x[0], df["fea_xy"]))
            df["y"] = list(map(lambda x : x[1], df["fea_xy"]))
            write(features_dict, self.save_path, "feature_names_ls.pickle", "pickle") # also write the feature name
            write(df, self.save_path, "ori_dis_train_test_data.csv", "csv")  # original dataset
            self.features_dict = features_dict
            self.train_test_split( df)
            write(self.dataset, self.save_path, "dis_train_test_data.csv", file_type="csv")
        # split fail
        elif check(self.save_path,  "dis_train_test_data.csv"): # set different graph feas
            self.features_dict = read(self.save_path, "feature_names_ls.pickle", file_type="pickle") # read the dictionary
            df = read(self.save_path, "ori_dis_train_test_data.csv", file_type="csv")
            self.train_test_split(df)
            write(self.dataset, self.save_path, "dis_train_test_data.csv", file_type="csv")
    
        else:
            self.dataset = read(self.save_path, "dis_train_test_data.csv", file_type="csv")
            self.features_dict = read(self.save_path, "feature_names_ls.pickle", file_type="pickle") # read the dictionary
        print('data is ready!')
    

    def data_selection(self, data_split,  feature_choice):
        """ graph features without weight are used in both graph_weight and graph_no_weight"""
        if feature_choice is None:
            feature_ls = ["x","y"]
        elif feature_choice == 'hedonic':
            feature_ls = ["x","y"] + list(self.features_dict["node_feas"])
        elif feature_choice == "graph_no_weight":
            feature_ls = ["x","y"] + [i for i in self.features_dict["graph_feas"] if i.split("_")[-2] != "weight"]
        elif feature_choice == "hedonic+graph_no_weight":
            feature_ls = ["x","y"] + list(self.features_dict["node_feas"]) +  \
            [i for i in self.features_dict["graph_feas"] if i.split("_")[-2] != "weight"]
        elif feature_choice == "graph_weight":
            feature_ls = ["x","y"] + [i for i in self.features_dict["graph_feas"] if i.split("_")[-2] != "None"]
        elif feature_choice == "hedonic+graph_weight":
            feature_ls = ["x","y"] + list(self.features_dict["node_feas"]) +  \
            [i for i in self.features_dict["graph_feas"] if i.split("_")[-2] != "None"]
        elif feature_choice == "all":
            feature_ls = ["x","y"] + list(self.features_dict["node_feas"]) + list(self.features_dict["graph_feas"])
        else:
            raise ValueError   
        print("feature_ls and choice: ", feature_ls)
        X = self.dataset[feature_ls][self.dataset["data_split"] == data_split]
        y = np.array(self.dataset[["target"]][self.dataset["data_split"] == data_split]).reshape(-1)
        return X, y


    def parameter_selection(self, attr_choice):
        gc.collect() # force the garbage collection 
        start_t = time.time()
        for m in progressbar(self.paramgenrator.model_ls, start_t, f" {attr_choice} computing ", 40):    
            if (m in ["GWR", "GPBoost-Lightgbm"] \
                or (len(m.split('-'))>1 and m.split('-')[0]== "Kriging")) \
                and attr_choice is None: # not available for only x and y
                pass
            elif attr_choice is not None and m=="Kriging":
                pass # kriging only run once for x and y 
            else:
                X, y = self.data_selection(data_split=0, feature_choice=attr_choice)
                print("model: ", m)
                estimator, param = self.paramgenrator.fetch_estimator_params(o_m_name=m, best = False)
                with contextlib.redirect_stdout(None): 
                    cv_m = cross_val(estimator, grid_params=param, X_train=X, y_train=y, 
                                standard="neg_median_absolute_error",
                                cv = 5, return_train_score = False)
                    bps = return_best_params(m, cv_m)
                    self.paramgenrator.update( m, bps) # update the best params
        return "Parameters for " + "model" + str(m) + "has been done!"
    
    def dump_rec(self, cur_path, params, obj):
        with open( cur_path + '_'.join(params) + '.pickle', 'wb') as f:
            dill.dump(obj, f)

    def evaluation_collection(self, attr_choice, cur_path, dump_file = True):
        for r in range(self.repeat):
            cv = KFold(n_splits=10, shuffle=True, random_state=r) # shuffle the data
            gc.collect() # force the garbage collection 
            start_t = time.time()
            for m in progressbar(self.paramgenrator.model_ls, start_t, f"repeat {r} {attr_choice} computing ", 40):    
                if (m in ["GWR", "GPBoost-Lightgbm"] \
                    or (len(m.split('-'))>1 and m.split('-')[0]== "Kriging")) \
                    and attr_choice is None: # not available for only x and y
                    pass
                elif attr_choice is not None and m=="Kriging":
                    pass # kriging only run once for x and y 
                else:
                    X, y = self.data_selection(data_split=1, feature_choice=attr_choice) # evalutaion
                    print("model: ", m)
                    estimator, param = self.paramgenrator.fetch_estimator_params(o_m_name=m, best = True) # use the best param
                    print('param', param)
                    with contextlib.redirect_stdout(None): 
                        cv_m = cross_val(estimator, grid_params=param_format(param), X_train=X, y_train=y, 
                                    standard="neg_median_absolute_error",
                                    cv = cv, return_train_score = True) # collection all res
                    record = log(m, cv_m, r, attr_choice)
                    if dump_file:
                        write(record, cur_path,  '_'.join([str(r), str(attr_choice), m])+ '.pickle', file_type = "pickle")             
        return "evaluation done!"

    
    def run(self):
        res_f_name = "exp_results_" + str(date.today())           
        res_f_ls =  os.listdir(self.save_path)
        if res_f_name in res_f_ls:
            shutil.rmtree(self.save_path + res_f_name)
        os.mkdir(self.save_path + res_f_name)

        for attr_choice in [None, "hedonic", "hedonic+graph_no_weight",
                                    "hedonic+graph_weight", "graph_weight", "graph_no_weight", "all"]: 
            self.parameter_selection(attr_choice)
            self.evaluation_collection(attr_choice, cur_path=self.save_path + res_f_name + '/')
        return 'done!'

