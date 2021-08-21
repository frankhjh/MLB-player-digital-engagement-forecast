#!/usr/bin/env python
from xgboost import XGBRegressor

xgb_model=XGBRegressor(learning_rate=0.1,
                        n_estimators=100,
                        objective='reg:squarederror')
