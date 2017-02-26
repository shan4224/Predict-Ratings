# Importing some libraries
import graphlab as gl
import numpy as np
import pandas as pd
import os
from  sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# Import Files
train = pd.read_csv('train_MLWARE2.csv')
test = pd.read_csv('test_MLWARE2.csv')

#Preparing Data frame 
trainId = train['ID']
train = train.drop(['ID'],axis = 1)
train.columns = ['user_id','item_id','rating']
trainSF = gl.SFrame(train)

testId = test['ID']
test = test.drop(['ID'],axis = 1)
test.columns = ['user_id','item_id']
testSF = gl.SFrame(test)

# Modeling : Graphlab - 100 models
mf_rating1 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 1, num_factors = 4)
mf_rating2 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 2, num_factors = 4)
mf_rating3 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 3, num_factors = 4)
mf_rating4 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 4, num_factors = 4)
mf_rating5 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 5, num_factors = 4)
mf_rating6 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 50, num_factors = 4)
mf_rating7 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 60, num_factors = 4)
mf_rating8 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 70, num_factors = 4)
mf_rating9 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 80, num_factors = 4)
mf_rating10 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 90, num_factors = 4)

mf_rating11 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 1000, num_factors = 8)
mf_rating12 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 2000, num_factors = 8)
mf_rating13 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 3000, num_factors = 8)
mf_rating14 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 4000, num_factors = 8)
mf_rating15 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 5000, num_factors = 8)
mf_rating16 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed =600, num_factors = 8)
mf_rating17 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 700, num_factors = 8)
mf_rating18 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 800, num_factors = 8)
mf_rating19 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 900, num_factors = 8)
mf_rating20 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 1001, num_factors = 8)

mf_rating21 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 1002, num_factors = 8)
mf_rating22 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 2001, num_factors = 8)
mf_rating23 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 3001, num_factors = 8)
mf_rating24 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 4001, num_factors = 8)
mf_rating25 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 5001, num_factors = 8)
mf_rating26 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed =6001, num_factors = 8)
mf_rating27 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 701, num_factors = 8)
mf_rating28 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 801, num_factors = 8)
mf_rating29 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 901, num_factors = 8)
mf_rating30 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 1001, num_factors = 8)

mf_rating31 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 1003, num_factors = 8)
mf_rating32 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 2003, num_factors = 8)
mf_rating33 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 3003, num_factors = 8)
mf_rating34 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 4003, num_factors = 8)
mf_rating35 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 5003, num_factors = 8)
mf_rating36 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed =6003, num_factors = 8)
mf_rating37 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 703, num_factors = 8)
mf_rating38 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 803, num_factors = 8)
mf_rating39 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 903, num_factors = 8)
mf_rating40 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 1003, num_factors = 8)
mf_rating41 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 1002, num_factors = 8)
mf_rating42 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 2001, num_factors = 8)
mf_rating43 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 3001, num_factors = 8)
mf_rating44 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 4001, num_factors = 8)
mf_rating45 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 5001, num_factors = 8)
mf_rating46 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed =6001, num_factors = 8)
mf_rating47 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 701, num_factors = 8)
mf_rating48 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 801, num_factors = 8)
mf_rating49 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 901, num_factors = 8)
mf_rating50 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 1001, num_factors = 8)
mf_rating51 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 10001, num_factors = 8)
mf_rating52 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 20001, num_factors = 8)
mf_rating53 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 30001, num_factors = 8)
mf_rating54 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 40001, num_factors = 8)
mf_rating55 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 50001, num_factors = 8)
mf_rating56 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed =60001, num_factors = 8)
mf_rating57 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 7001, num_factors = 8)
mf_rating58 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 8001, num_factors = 8)
mf_rating59 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 9001, num_factors = 8)
mf_rating60 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 10001, num_factors = 8)

mf_rating61 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 10002, num_factors = 8)
mf_rating62 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 20001, num_factors = 8)
mf_rating63 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 30001, num_factors = 8)
mf_rating64 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 40001, num_factors = 8)
mf_rating65 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 50001, num_factors = 8)
mf_rating66 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed =60001, num_factors = 8)
mf_rating67 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 70001, num_factors = 8)
mf_rating68 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 80001, num_factors = 8)
mf_rating69 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 90001, num_factors = 8)
mf_rating70 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 100001, num_factors = 8)
mf_rating71 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 10020, num_factors = 8)
mf_rating72 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 20010, num_factors = 8)
mf_rating73 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 30010, num_factors = 8)
mf_rating74 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 40010, num_factors = 8)
mf_rating75 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 50010, num_factors = 8)
mf_rating76 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed =60010, num_factors = 8)
mf_rating77 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 7010, num_factors = 8)
mf_rating78 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 8010, num_factors = 8)
mf_rating79 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 9010, num_factors = 8)
mf_rating80 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 10010, num_factors = 8)
mf_rating81 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 100200, num_factors = 8)
mf_rating82 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 200100, num_factors = 8)
mf_rating83 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 300100, num_factors = 8)
mf_rating84 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 400100, num_factors = 8)
mf_rating85 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 500100, num_factors = 8)
mf_rating86 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed =600100, num_factors = 8)
mf_rating87 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 70100, num_factors = 8)
mf_rating88 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 80100, num_factors = 8)
mf_rating89 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 90100, num_factors = 8)
mf_rating90 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 100100, num_factors = 8)
mf_rating91 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 1002000, num_factors = 8)
mf_rating92 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 2001000, num_factors = 8)
mf_rating93 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 3001000, num_factors = 8)
mf_rating94 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 4001000, num_factors = 8)
mf_rating95 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 5001000, num_factors = 8)
mf_rating96 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed =60010000, num_factors = 8)
mf_rating97 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 7010000, num_factors = 8)
mf_rating98 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 8010000, num_factors = 8)
mf_rating99 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 9010000, num_factors = 8)
mf_rating100 = gl.factorization_recommender.create(trainSF, target='rating',linear_regularization = 0.001,random_seed = 10010000, num_factors = 8)

## Result combiner : Simple Average
result = pd.DataFrame(testId)
result['rating'] = (mf_rating1.predict(testSF) + mf_rating2.predict(testSF) + \
                    mf_rating3.predict(testSF) + mf_rating4.predict(testSF) + \
                    mf_rating5.predict(testSF) + mf_rating6.predict(testSF) + \
                    mf_rating7.predict(testSF) + mf_rating8.predict(testSF) + \
                    mf_rating9.predict(testSF) + mf_rating10.predict(testSF) +
                    mf_rating11.predict(testSF) + mf_rating12.predict(testSF) + \
                    mf_rating13.predict(testSF) + mf_rating14.predict(testSF) + \
                    mf_rating15.predict(testSF) + mf_rating16.predict(testSF) + \
                    mf_rating17.predict(testSF) + mf_rating18.predict(testSF) + \
                    mf_rating19.predict(testSF) + mf_rating20.predict(testSF) + \
                    mf_rating21.predict(testSF) + mf_rating22.predict(testSF) + \
                    mf_rating23.predict(testSF) + mf_rating24.predict(testSF) + \
                    mf_rating25.predict(testSF) + mf_rating26.predict(testSF) + \
                    mf_rating27.predict(testSF) + mf_rating28.predict(testSF) + \
                    mf_rating29.predict(testSF) + mf_rating30.predict(testSF) +
                    mf_rating31.predict(testSF) + mf_rating32.predict(testSF) + \
                    mf_rating33.predict(testSF) + mf_rating34.predict(testSF) + \
                    mf_rating35.predict(testSF) + mf_rating36.predict(testSF) + \
                    mf_rating37.predict(testSF) + mf_rating38.predict(testSF) + \
                    mf_rating39.predict(testSF) + mf_rating40.predict(testSF) + 
                    mf_rating41.predict(testSF) + mf_rating42.predict(testSF) + \
                    mf_rating43.predict(testSF) + mf_rating44.predict(testSF) + \
                    mf_rating45.predict(testSF) + mf_rating46.predict(testSF) + \
                    mf_rating47.predict(testSF) + mf_rating48.predict(testSF) + \
                    mf_rating49.predict(testSF) + mf_rating50.predict(testSF) + \
                    mf_rating51.predict(testSF) + mf_rating52.predict(testSF) + \
                    mf_rating53.predict(testSF) + mf_rating54.predict(testSF) + \
                    mf_rating55.predict(testSF) + mf_rating56.predict(testSF) + \
                    mf_rating57.predict(testSF) + mf_rating58.predict(testSF) + \
                    mf_rating59.predict(testSF) + mf_rating60.predict(testSF) +
                    mf_rating61.predict(testSF) + mf_rating62.predict(testSF) + \
                    mf_rating63.predict(testSF) + mf_rating64.predict(testSF) + \
                    mf_rating65.predict(testSF) + mf_rating66.predict(testSF) + \
                    mf_rating67.predict(testSF) + mf_rating68.predict(testSF) + \
                    mf_rating69.predict(testSF) + mf_rating70.predict(testSF) + \
                    mf_rating71.predict(testSF) + mf_rating72.predict(testSF) + \
                    mf_rating73.predict(testSF) + mf_rating74.predict(testSF) + \
                    mf_rating75.predict(testSF) + mf_rating76.predict(testSF) + \
                    mf_rating77.predict(testSF) + mf_rating78.predict(testSF) + \
                    mf_rating79.predict(testSF) + mf_rating80.predict(testSF) +
                    mf_rating81.predict(testSF) + mf_rating82.predict(testSF) + \
                    mf_rating83.predict(testSF) + mf_rating84.predict(testSF) + \
                    mf_rating85.predict(testSF) + mf_rating86.predict(testSF) + \
                    mf_rating87.predict(testSF) + mf_rating88.predict(testSF) + \
                    mf_rating89.predict(testSF) + mf_rating90.predict(testSF) + 
                    mf_rating91.predict(testSF) + mf_rating92.predict(testSF) + \
                    mf_rating93.predict(testSF) + mf_rating94.predict(testSF) + \
                    mf_rating95.predict(testSF) + mf_rating96.predict(testSF) + \
                    mf_rating97.predict(testSF) + mf_rating98.predict(testSF) + \
                    mf_rating99.predict(testSF) + mf_rating100.predict(testSF)) / 100
					
# Keeping everything b/w 0 & 10					
result.ix[result['rating'] > 10,'rating'] = 10
result.ix[result['rating'] < 0,'rating'] = 0
result.to_csv('Avg_ensemble.csv',index = False)
