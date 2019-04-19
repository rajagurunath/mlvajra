import tensorflow_data_validation as tfdv 
import tensorflow_estimator
import tensorflow_metadata
import tensorflow_model_analysis
import tensorflow_transform


__author__='Gurunath'

from datetime import timedelta
import os
import airflow
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.email_operator import EmailOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.bash_operator import BashOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.python_operator import PythonOperator,BranchPythonOperator
from datetime import datetime
from scripts import (SPARK_PULL_SERVICE,SPARK_PUSH_SERVICE,PRED_PREPROCESSING_SERVICE,
                     SPARK_PREDICTION,SPARK_TABLE_SUMMARY,SPARK_TRAINING,TRAIN_PREPROCESSING_SERVICE)
from airflow.sensors.training_plugin import TrainingSensor
from airflow.sensors.time_sensor import TimeSensor
from airflow.utils import timezone
# These args will get passed on to each operator ["sun","mon","tues","wed","thu","fri","sat"]
# You can override them on a per-task basis during operator initialization
default_args = {
    'owner': 'Prodapt_mlteam',
    'depends_on_past': False,
    'start_date': '2019-04-05',
    'email': ['n9979623@windstream.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    # 'queue': 'bash_queue',
    # 'pool': 'backfill',
    # 'priority_weight': 10,
    # 'end_date': datetime(2016, 1, 1),
    # 'wait_for_downstream': False,
    # 'dag': dag,
    #'adhoc':False,3417
    # 'sla': timedelta(hours=2),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
    # 'trigger_rule': u'all_success'
}
SUMMARY_TRIGGER_HOUR=23
TRAINING_TRIGGER_HOUR=23    #
TRAINING_TRIGGER_DAY=0    #TODO:sunday
nodename='CHCGILDTO6Y'
nodeid='MA6813400180'
section='h_OCHCTP'
dag = DAG(
    'CHCGILDTO6Y_test',
    default_args=default_args,
    description='gets the data and uses spark to fetch the data from hive tables ,preprocess and put the data in another hive tables',
    schedule_interval='@hourly',
    catchup=True,
)

def sg_trigger_or_not():
    if timezone.utcnow().time().hour==SUMMARY_TRIGGER_HOUR:
        return "kickoff_summaryGen"
    else :
        return "SummaryGen_skipped"

def training_trigger_or_not():
    if (timezone.utcnow().day==TRAINING_TRIGGER_DAY) and (timezone.utcnow().time().hour==TRAINING_TRIGGER_HOUR):
        return "TRAIN_PREPROCESSING"
    else :
        return "PRED_PREPROCESSING"



with dag:
        END_TIME="{{macros.dateutil.parser.parse(ts).replace(minute=0,second=0).strftime('%Y-%m-%d %H:%M:%S')}}"
        START_TIME="{{(macros.dateutil.parser.parse(ts)-macros.timedelta(hours=1)).replace(minute=0,second=0).strftime('%Y-%m-%d %H:%M:%S')}}"
        TRAIN_START_TIME="{{(macros.dateutil.parser.parse(ts)-macros.timedelta(hours=2400)).replace(minute=0,second=0).strftime('%Y-%m-%d %H:%M:%S')}}"

        #starttime="{{macros.datetime.strptime((macros.dateutil.parser.parse(ts)-macros.timedelta(hours=1)).replace(minute=0,second=0).strftime('%Y%m%d%H%M%S')}}"
        #endtime="{{datetime.strptime(macros.dateutil.parser.parse(ts).replace(minute=0,second=0).strftime('%Y%m%d%H%M%S')}}"
        starttime="{{(macros.dateutil.parser.parse(ts)-macros.timedelta(hours=1)).replace(minute=0,second=0).strftime('%Y%m%d%H%M%S')}}"    
        endtime="{{macros.dateutil.parser.parse(ts).replace(minute=0,second=0).strftime('%Y%m%d%H%M%S')}}"

        # -1 for starttime  and -9 from summary service
        TmpStartTime ="{{(macros.dateutil.parser.parse(ts)-macros.timedelta(hours=10)).replace(minute=0,second=0).strftime('%Y-%m-%d %H:%M:%S')}}"
        # (datetime.datetime.strptime(StartTime, '%Y-%m-%d %H:%M:%S')
        #             -datetime.timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S')

       
        # PULLHQL="SELECT nodeid, section, module, ts, measure, val, nodename \
        # FROM dnapm.performance_metrics\
        # where nodename='CHCGILDTO6Y'\
        # and section='h_OCHCTP'\
        # and measure in ('BerPreFecMax','PhaseCorrectionAve','PmdMin','Qmin','SoPmdAve')\
        # and module in ('13-L1-9','13-L1-5','13-L1-6','13-L1-10','13-L1-3','13-L1-4','13-L1-8','13-L1-2','13-L1-7','13-L1-1',\
        # '10-L1-9','10-L1-3','10-L1-1','10-L1-5','10-L1-7','10-L1-4','10-L1-10','10-L1-6','10-L1-8','10-L1-2',\
        # '11-L1-7','11-L1-3','11-L1-8','11-L1-9','11-L1-1','11-L1-6','11-L1-5','11-L1-2','11-L1-4','11-L1-10')\
        # and ts>='{}' and ts<'{}'".format(START_TIME,END_TIME)

        # PUSHHQL="insert into table coeusapp.ml_performance_metrics_prestg (nodeid, section, module,ts,measure,val,type)\
        #     select nodeid,trim(section),trim(module),ts,trim(measure),val,type from coeusapp.ml_performance_metrics \
        #     where type='actual' and unix_timestamp(ts) >=unix_timestamp('{}') and unix_timestamp(ts)<unix_timestamp('{}')".format(START_TIME,END_TIME)

        #SPARKHQL="select nodeid,section,module,ts,measure,val from coeusapp.ml_performance_metrics_prestg where measure<>'null'
        #  and (to_timestamp(ts)>=to_timestamp('{}')) and (to_timestamp(ts) <to_timestamp('{}'))".format(START_TIME,END_TIME)
        SPARKHQL="select nodeid,section,module,ts,valid,measure,val,nodename from dnapm.performance_metrics where \
                nodename in ('LSAJCAWZO2Y','ASBNVACYO3Y','MIAUFLWSO0Y','MIAUFLWSO3P-NE70191','WASHDC12O1Y',\
                'CHCGILDTO6Y','ATLNGAMAO4Y','CHCGILWUO7Y','NYCMNYZRO1Y')\
                and section='{}' and ts>'{}' and ts<'{}' and measure in ( 'BerPreFecMax',  'PhaseCorrectionAve',  'PmdMin', \
                 'Qmin',  'SoPmdAve' )".format(section,START_TIME,END_TIME)

        SPARKTRAINHQL="select nodeid,section,module,ts,valid,measure,val,nodename from dnapm.performance_metrics where \
                nodename in ('LSAJCAWZO2Y','ASBNVACYO3Y','MIAUFLWSO0Y','MIAUFLWSO3P-NE70191','WASHDC12O1Y',\
                'CHCGILDTO6Y','ATLNGAMAO4Y','CHCGILWUO7Y','NYCMNYZRO1Y')\
                and section='{}' and ts>'{}' and ts<'{}' and measure in ( 'BerPreFecMax',  'PhaseCorrectionAve',  'PmdMin', \
                'Qmin',  'SoPmdAve' )".format(section,TRAIN_START_TIME,END_TIME)
         
        PREDICTIONSQL="select nodeid, section, ts, module, preprocessed_output from coeusapp.ml_performance_metrics_stg WHERE \
            section='{}' and unix_timestamp(ts)>=unix_timestamp('{}') \
            and unix_timestamp(ts)<unix_timestamp('{}')".format(section,START_TIME,END_TIME)
        
        PredQuery = "SELECT * from coeusapp.ml_performance_metrics_prestg \
                WHERE type='predicted' and ts>='{}'  and ts <'{}'".format(nodeid,TmpStartTime,END_TIME)
            
        AlarmQuery = "SELECT * from coeusapp.ml_performance_metrics_prestg \
                WHERE type='alarm' and ts>='{}' and ts<'{}'".format(nodeid,START_TIME,END_TIME)
        #starttime=datetime.strptime(START_TIME,'%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M%S')    
        #endtime=datetime.strptime(END_TIME,'%Y-%m-%d %H:%M:%S').strftime('%Y%m%d%H%M%S')
        # pullservice = BashOperator(
        #                 task_id= 'pullservice',
        #                 bash_command=SPARK_PULL_SERVICE,
        #                 dag=dag,
        #                 env={'START_TIME':START_TIME,'END_TIME':END_TIME,'SQL':PULLHQL},
        #         )
        # pushservice = BashOperator(
        #                 task_id= 'pushservice',
        #                 bash_command=SPARK_PUSH_SERVICE,
        #                 dag=dag,
        #                 env={'START_TIME':START_TIME,'END_TIME':END_TIME,'SQL':PUSHHQL,
	# 		     'starttime':starttime,'endtime':endtime},
        #         )

        trainbranch=BranchPythonOperator(
                task_id='check_for_training',
                python_callable=training_trigger_or_not,
        )
        pred_preprocessing = BashOperator(
                        task_id= 'PRED_PREPROCESSING',
                        bash_command=PRED_PREPROCESSING_SERVICE,
                        dag=dag,
                        env={'START_TIME':START_TIME,'END_TIME':END_TIME,'SQL':SPARKHQL},
                )
        train_preprocessing=BashOperator(
                        task_id= 'TRAIN_PREPROCESSING',
                        bash_command=TRAIN_PREPROCESSING_SERVICE,
                        dag=dag,
                        env={'START_TIME':START_TIME,'END_TIME':END_TIME,'SQL':SPARKTRAINHQL},
        )
        # predictions = BashOperator(
        #         task_id= 'predictionService',
        #         bash_command=SPARK_PREDICTION,
        #         dag=dag,
        #         env={'START_TIME':START_TIME,'END_TIME':END_TIME,'SQL':PREDICTIONSQL},
        #         )
        predictionTrain = BashOperator(
                task_id= 'predictionService_after_training',
                bash_command=SPARK_PREDICTION,
                dag=dag,
                env={'START_TIME':START_TIME,'END_TIME':END_TIME,'SQL':PREDICTIONSQL},
                )
        predictionPred = BashOperator(
                task_id= 'predictionService',
                bash_command=SPARK_PREDICTION,
                dag=dag,
                env={'START_TIME':START_TIME,'END_TIME':END_TIME,'SQL':PREDICTIONSQL},
                )
        mltraining=BashOperator(
                task_id='ML_TRAINING',
                bash_command=SPARK_TRAINING,
                 dag=dag,
                 env={'START_TIME':START_TIME,'END_TIME':END_TIME,'nodeid':nodeid},
        )

        # training_sensor=TrainingSensor(
        #          task_id='trigger_training',
        #          target_day="thu",
        #          target_hour=8,
        # )
        # summaryBranch=BranchPythonOperator(
        #         task_id='check_for_summarygen',
        #         python_callable=sg_trigger_or_not,
        # )
    

        #SummaryGen_not_triggered=DummyOperator(task_id='SummaryGen_skipped')
        #trainingservice_not_triggered=DummyOperator(task_id='TrainingService_skipped')

        # summary_sensor=TimeSensor(
        #         task_id='trigger_summary',
        #         target_time=
        # )


trainbranch.set_downstream(train_preprocessing)
trainbranch.set_downstream(pred_preprocessing)
train_preprocessing.set_downstream([predictionTrain,mltraining])
pred_preprocessing.set_downstream(predictionPred)

# summaryBranch.set_upstream(predictionPred)
# #summaryBranch.set_upstream(predictionTrain)

# # preprocessing.set_upstream(pushservice)
# # pushservice.set_upstream(pullservice)
# #predictions.set_upstream(preprocessing)
# summaryBranch.set_downstream(summarygen)
# summaryBranch.set_downstream(SummaryGen_not_triggered)


#trainBranch.set_upstream(predictions)


# training_sensor.set_upstream(preprocessing)
# training.set_upstream(training_sensor)

#summarygen.set_upstream(predictions)

def AllTaskSuccess1():
    print("AllTaskSuccess")
    pass

def pullservice_failed1():
    print("pullservice_failed")
    pass

def preprocessing_failed1():
    print("preprocessing_failed")
    pass

def predictions_failed1():
    print("predictions_failed")
    pass

def training_failed1():
    print("Training failed")
    pass


# AllTaskSuccess = EmailOperator (
#     dag=dag,
#     trigger_rule=TriggerRule.ALL_SUCCESS,
#     task_id="AllTaskSuccess",
#     to=["n9979623@windstream.com"],
#     subject="All Task completed successfully",
#     html_content='<h3>All Task completed successfully" </h3>')

# AllTaskSuccess=PythonOperator(
#         task_id= 'AllTaskSuccess',
#         python_callable=AllTaskSuccess1,
#         dag=dag)

#AllTaskSuccess.set_upstream([preprocessing,predictions])


# preprocessing_failed=PythonOperator(
#         task_id= 'preprocessing_failed',
#         python_callable=preprocessing_failed1,
#         dag=dag)

# preprocessing_failed.set_upstream([preprocessing])

# # predictions_failed = EmailOperator (
# #      dag=dag,
# #      trigger_rule=TriggerRule.ONE_FAILED,
# #      task_id="predictions_failed",
# #      to=["n9979623@windstream.com"],
# #      subject="predictions_failed Failed",
# #      html_content='<h3>predictionsservice Failed</h3>')

# predictions_failed=PythonOperator(
#         task_id= 'predictions_failed',
#         python_callable=predictions_failed1,
#         dag=dag)


# predictions_failed.set_upstream([predictions])
# training_failed=PythonOperator(
#         task_id= 'training_failed1',
#         python_callable=training_failed1,
#         dag=dag)

# training_failed.set_upstream(training)
