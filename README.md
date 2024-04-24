# Enhancing Geospatial Predictions with Feature Engineering on Road Networks: A Graph-Driven Approach

Abstract
Traditional geospatial predictive models for property valuation have naturally relied on coordinates as well as “hedonic” (internal and external) features. In particular, location-centric methods such as Geographically Weighted Regression (GWR) and Kriging have focused on intrinsic target characteristics together with distances between individual targets. However, especially in the context of heavily urbanized areas, these approaches might overlook crucial aspects arising from the underlying topological structure that presents itself in such areas. Concretely, in this work, we focus on the structure arising from the road network connecting properties. We introduce a novel though straightforward technique for feature engineering based on graphs constructed on a road network. We then extract relevant features from these and utilize those as inputs for predictive models, and assess their performance benefits when used together with a variety of both well-known geospatial models as well as state-of-art machine learning models. To this end, we present an exhaustive experiment using four different real-life data sets across various regions and exhibiting sizes outperforming many comparative works in the field. Our findings reveal that our feature engineering approach offers significant improvements in predictive performance. Finally, we apply Shapley values as an interpretability technique to confirm the reliability and effectiveness of our approach.

This repository contains the implementation for the models with graph features as presented in: Enhancing Geospatial Predictions with Feature Engineering on Road Networks: A Graph-Driven Approach

For the data sets used as an example, see

https://www.kaggle.com/datasets/ruiqurm/lianjia/

Three publicly available and one proprietary data sets are used in this research. Regarding all public data used in the paper, see: 

https://www.kaggle.com/datasets/ruiqurm/lianjia/

https://www.kaggle.com/datasets/harlfoxem/housesalesprediction

https://www.kaggle.com/datasets/syuzai/perth-house-prices.
