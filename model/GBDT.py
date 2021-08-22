#!/usr/bin/env python
from sklearn.ensemble import GradientBoostingRegressor

gbdt=GradientBoostingRegressor(n_estimators=50,criterion='mse')
