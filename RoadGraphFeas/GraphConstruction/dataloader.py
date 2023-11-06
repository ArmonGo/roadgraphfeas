# data reader
import json
import pandas as pd
from shapely import ops
from shapely.geometry import MultiPoint, Point
import sqlite3
# import fuckit
import numpy as np
import copy 
from scipy import stats
        

class EXampleBeijingLoader:
    def __init__(self, transformer):
        self.transformer = transformer

    def _process(self, df):
        points = []
        non_trans_points = []
        features = copy.deepcopy(df.loc[:, df.columns.difference(['lat', 'lon'])]) # exclude lat and lon
        ix = list(range(len(df.lon))) # mapping back to the features
        target_ix = []
        for lon, lat, index in zip(df.lon, df.lat, ix):
            points.append(
                Point(
                    self.transformer.transform(lon, lat)
                    if self.transformer
                    else (lon, lat)
                )
            )
            non_trans_points.append(Point(
                    (lon, lat)
                ))
            target_ix.append(index)
            
        points = MultiPoint(points)
        points = ops.transform(lambda *args: args[:2], points)
        # need one non trans for graph extraction
        non_trans_points = MultiPoint(non_trans_points)
        non_trans_points = ops.transform(lambda *args: args[:2], non_trans_points)
        features = features.iloc[target_ix, :] # filter the features
        return points, features, target_ix, non_trans_points

    def load(self, file_name, province=None):
        df_raw = pd.read_csv(file_name, encoding_errors='ignore') # use chinese inside but not utf8
        df = copy.deepcopy(df_raw[[ 'square', 'livingRoom', 'drawingRoom', 'kitchen',
       'bathRoom', 'floor', 'buildingType', 'constructionTime',
       'renovationCondition', 'buildingStructure', 'elevator',
       'fiveYearsProperty']])
        df["target"] = df_raw["price"]
        df["lon"] = df_raw["Lng"] 
        df["lat"] = df_raw['Lat'] 
        df["floor"] = df.floor.str.extract('(\d+)')
        df["floor"]  = pd.to_numeric(df["floor"], errors='coerce')
        df["tradeTime"] = pd.to_numeric(df_raw["tradeTime"].str.replace('-',''), errors='coerce')
        df["constructionTime"] = pd.to_numeric(df_raw["constructionTime"].str.replace('-',''), errors='coerce')
        
        for i in df.columns:
            df[i] = pd.to_numeric(df[i], errors='coerce')
        df['uni_id'] = df_raw["id"]
        df = df.dropna() # delete nan first!
        df = df.drop_duplicates(subset=["lat", "lon"], keep="last") # drop duplicates
        df = df[(stats.zscore(df["target"])<5) & (stats.zscore(df["target"])>-2)]
        return self._process(df)
       
