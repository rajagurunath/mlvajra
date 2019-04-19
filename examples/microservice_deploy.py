
#mnist experiment :QvMXCpIy

######### Training code ##################################################
import xgboost as xgb
app_name="mlvajra_release"
model_name="vajra-model"


train = xgb.DMatrix(get_test_point(), label=[0])
# We then create parameters, watchlist, and specify the number of rounds
# This is code that we use to build our XGBoost Model, and your code may differ.
param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
watchlist = [(dtrain, 'train')]
num_round = 2
bst = xgb.train(param, dtrain, num_round, watchlist)

def predict(xs):
    return bst.predict(xgb.DMatrix(xs))

##########################################################################


############################ DEPLOY ############################################
from mlvajra.deploy.clipper import mlDeploy
deploy=mlDeploy()
deploy.create_application(app_name,"integers","vajra-default")
deploy.deploy_model(model_name,predict,0.1,pkgs_to_install=["xgboost"],model_type='PYTHON')
deploy.conn.link_model_to_app(app_name,model_name)

###################################################################################


############################ inference time ###########################################

def get_test_point():
    return [np.random.randint(255) for _ in range(784)]

import requests, json
# Get Address
addr = deploy.conn.get_query_addr()
# Post Query
response = requests.post(
     "http://%s/%s/predict" % (addr, 'xgboost-test'),
     headers={"Content-type": "application/json"},
     data=json.dumps({
         'input': get_test_point()
     }))
result = response.json()
if response.status_code == requests.codes.ok and result["default"]:
     print('A default prediction was returned.')
elif response.status_code != requests.codes.ok:
    print(result)
    raise BenchmarkException(response.text)
else:
    print('Prediction Returned:', result)
###############################################################################################
deploy.conn.stop_all()


