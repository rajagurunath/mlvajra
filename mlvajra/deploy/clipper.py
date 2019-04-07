try :
    from clipper_admin import ClipperConnection, DockerContainerManager
    from clipper_admin.deployers import python as python_deployer
    from clipper_admin.deployers import pytorch
    from clipper_admin.deployers import pyspark
    from clipper_admin.deployers import tensorflow
except ImportError as e:
    print("install clipper_admin to have this deployment functionalities")
from enum import IntEnum
models= IntEnum('models', 'PYTHON PYTORCH PYSPARK TENSORFLOW')


class mlDeploy(object):
    def __init__(self,**kwargs):
        self.conn = ClipperConnection(DockerContainerManager())
        self.conn.start_clipper()
    def create_application(self,app_name,input_type,
                    default_output="default_output from model",slo_micros=100000):
        self.app_name=app_name
        self.input_type=input_type
        self.conn.register_application(self.app_name,input_type,default_output,slo_micros)
    def deploy_model(self,model_name,func_to_deploy,
                        version,pkgs_to_install,
                        model_type=models.PYTHON,
                        model=None,sess=None,sc=None):
        self.model_name=model_name
        if model_type=='PYTHON':
            python_deployer.deploy_python_closure(self.conn, name=self.model_name, version=version,
                         input_type=self.input_type, func=func_to_deploy, pkgs_to_install=pkgs_to_install)
        if model_type=='PYTORCH':
            pytorch.deploy_pytorch_model(self.conn, name=self.model_name, version=version,
                         input_type=self.input_type, func=func_to_deploy,pytorch_model=model)
        if model_type=='TENSORFLOW':
            tensorflow.deploy_tensorflow_model(self.conn, name=self.model_name, version=version,
                         input_type=self.input_type, func=func_to_deploy,tf_sess =sess)
        if model_type=='PYSPARK':
            pyspark.deploy_pyspark_model(self.conn, name=self.model_name, version=version,
                         input_type=self.input_type, func=func_to_deploy,pyspark_model=model,sc=sc)
        def link_app_and_model(self):
            self.conn.link_model_to_app(self.app_name,self.model_name)
        def stop_service(self):
            self.conn.stop_all()

        def model_roll_back(self,version):
            self.conn.set_model_version(version)
        def model_replicas(self,num_replicas):
            self.conn.set_num_replicas(num_replicas)

        def get_address(self):
            return self.conn.get_query_addr()











