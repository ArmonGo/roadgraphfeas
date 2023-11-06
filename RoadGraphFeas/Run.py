from Experiments.experiment import * 
from IPython.utils import io
import time 
import contextlib

# take china-beijing as an example
countrys = ["China"]
centroids = [(116.3792, 39.9111) ]
data_paths = ['./Beijing_DATAPATH'
              ]
radius = [20000]

# o_m_name, i_m_name, i_m_renew, o_m_params 
model_param_setting = {"Lightgbm": { "i_m_name": None, 
                "i_m_renew": False, 
                "o_m_params": {"learning_rate": [0.01, 0.05, 0.001, 0.005],
                              "reg_alpha": np.arange(0.0,1, 0.1),
                              "reg_lambda": np.arange(0.0,1, 0.1)
                              }
             },
"RF": { "i_m_name": None, 
                "i_m_renew": False, 
                "o_m_params": {"min_samples_split": [2,5],
                              "min_samples_leaf": [2,3,5 ]
                              }
             },

"Lr_Ridge": { "i_m_name": None, 
                "i_m_renew": False, 
                "o_m_params": {"alpha": np.arange(0.1,1,0.1)}
             },
"KNNuni": { "i_m_name": None, 
                "i_m_renew": False, 
                "o_m_params": {"n_neighbors": range(5,105,5),
                                "weights": ["uniform"]}
             },
"KNNdis": { "i_m_name": None, 
                "i_m_renew": False, 
                "o_m_params": {"n_neighbors": range(5,105,5),
                                "weights": ["distance"]}
             },
"GWR": { "i_m_name": None, 
                "i_m_renew": False, 
                "o_m_params": {"constant": [True]
                              }
             },
"SVM": { "i_m_name": None, 
                "i_m_renew": False, 
                "o_m_params": {"C": range(1,105,10), # change the param from 5 to 10
                               "epsilon": np.arange(0.1, 1, 0.1)
                              }
             },

"Kriging": { "i_m_name": None, 
                "i_m_renew": False, 
                "o_m_params": {"nlags" : range(10,110,10),
                                "variogram_model":[ "gaussian", "spherical", "linear", "power"] }
             },
# combined model
"Kriging-KNNuni": { "i_m_name": "KNNuni", 
                "i_m_renew": True, 
                "o_m_params": {"nlags" : range(10,110,10),
                                "variogram_model":[ "gaussian", "spherical", "linear", "power"] }
             },
"Kriging-Lr_Ridge": { "i_m_name": "Lr_Ridge", 
                "i_m_renew": True, 
                "o_m_params": {"nlags" : range(10,110,10),
                                "variogram_model":[ "gaussian", "spherical", "linear", "power"] }
             },
"Kriging-RF": { "i_m_name": "RF", 
                "i_m_renew": True, 
                "o_m_params": {"nlags" : range(10,110,10),
                                "variogram_model":[ "gaussian", "spherical", "linear", "power"] }
             },
"Kriging-Lightgbm": { "i_m_name": "Lightgbm", 
                "i_m_renew": True, 
                "o_m_params": {"nlags" : range(10,110,10),
                                "variogram_model":[ "gaussian", "spherical", "linear", "power"] }
             },

"Kriging-SVM": { "i_m_name": "SVM", 
                "i_m_renew": True, 
                "o_m_params": {"nlags" : range(10,110,10),
                                "variogram_model":[ "gaussian", "spherical", "linear", "power"] }
             },
"GPBoost-Lightgbm": { "i_m_name": "Lightgbm", 
                "i_m_renew": False, 
                "o_m_params": {"cov_function": ["exponential", "gaussian"]}
             }

}


for cnt in range(1): # example only 
    save_path = './YOUR_RESULTS_SAVE_PATH/' + countrys[cnt]+ '/'
    centroid = centroids[cnt]
    radiu = radius[cnt]
    print(countrys[cnt] + " begins!")
    engine= ExperimentLoader( location_centriod = centroid, radius=radiu,  
                  province= None, save_path = save_path, data_path=data_paths[cnt],
                  country = countrys[cnt],
                  subg_radius= 500, graph_weight="all", # radius need to be a dictionary
                  grid_params=model_param_setting,
                  repeat=10, scaler_choice="minmax", 
                  developing_size = 0.3, 
                  evaluation_size=0.6, test_size=0.1, 
                  normaliza_target=True) 
    # catch the output and display the progress
    start_time =  time.time()
    print("start time: ",start_time) 
    final_res = engine.run()
    end_time =  time.time()
    print("end time: ",end_time) 
    print("it takes ", end_time - start_time, "s")