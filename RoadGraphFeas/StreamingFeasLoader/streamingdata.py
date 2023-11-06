from StreamingFeasLoader.streamingfeas import *
from StreamingFeasLoader.streaminggraph import *

class StreamDataLoader:
    def __init__(self, location_centriod, radius, extract_radius,dataloader, 
                 data_path, province, weight="all", features_choice = "all"):
        self.location_centriod = location_centriod # for transformer
        self.radius = radius 
        self.extract_radius = extract_radius # for subgraph construction
        self.data_path = data_path
        self.province = province
        self.dataloader = dataloader
        self.weight = weight
        self.features_choice = features_choice
        self._extractor_set_()


    def _extractor_set_(self):
        self.graph_loader = StreamingGraphLoader(self.location_centriod, 
                                            self.radius, 
                                            self.extract_radius,
                                            self.dataloader, self.data_path, self.province)
        self.feas_loader = StreamingGraphFea(self.weight)
    
    def _streaming_loading_(self):
        feas_ls = []
        gen = self.graph_loader.generator()
        for G, target_node, fea_xy in gen:
            if G is not None: # the road could be none 
                feas_ls.append(self.feas_loader.streaming_feas_loading(G, target_node, features_choice = self.features_choice, append_fea ={"fea_xy": fea_xy}))
                
        return pd.DataFrame(feas_ls)
    
    def feas_ls_log(self, df):
        fea_ls = list(df.columns)
        fea_dict = { "graph_feas" : np.setdiff1d(np.array([i for i in fea_ls if i.split("_")[-1] =="g"]), 
                                                  np.array(["node_ix", "type", "xy", "target", "fea_xy", "uni_id"])) }
        fea_dict["node_feas"] =  np.setdiff1d(np.array([i for i in fea_ls if i.split("_")[-1] !="g"]), 
                                                  np.array(["node_ix", "type", "xy", "target", "fea_xy", "uni_id"]))
        return fea_dict

    def load(self):
        df = self._streaming_loading_()
        df["node_ix"] = list(range(len(df)))
        return df  