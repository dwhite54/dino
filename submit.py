import os
import subprocess

import json
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig

sagemaker_session = sagemaker.Session()

name='vit-base-8-afex-1k'
local_output_dir = './output/'
s3_output_dir='s3://whteamd/{}/output/'.format(name)

tensorboard_output_config = tensorboard_output_config=TensorBoardOutputConfig(
                                s3_output_path=s3_output_dir,
                                container_local_output_path=local_output_dir
                            )

# epochs originally 300
hyperparameters = json.loads('''{
                                "arch": "vit_base", 
                                "patch_size": 8, 
                                "out_dim": 65536, 
                                "norm_last_layer": true, 
                                "warmup_teacher_temp": 0.03, 
                                "teacher_temp": 0.07, 
                                "warmup_teacher_temp_epochs": 50, 
                                "use_fp16": false, 
                                "weight_decay": 0.04, 
                                "weight_decay_end": 0.4, 
                                "clip_grad": 3.0, 
                                "batch_size_per_gpu": 6, 
                                "epochs": 1, 
                                "freeze_last_layer": 3, 
                                "lr": 0.0005, 
                                "warmup_epochs": 10, 
                                "min_lr": 2e-06,  
                                "global_crops_scale_low": 0.25, 
                                "global_crops_scale_high": 1.0, 
                                "local_crops_scale_low": 0.05, 
                                "local_crops_scale_high": 0.25, 
                                "local_crops_number": 10, 
                                "seed": 0, 
                                "num_workers": 10, 
                                "world_size": 176, 
                                "ngpus": 4, 
                                "nodes": 1, 
                                "optimizer": "adamw", 
                                "momentum_teacher": 0.996, 
                                "use_bn_in_head": false
                                }''')

pytorch_estimator = PyTorch('train.py',
                            role='arn:aws:iam::338458051672:role/service-role/AmazonSageMaker-ExecutionRole-20210629T090924',
                            instance_type='ml.p3.8xlarge', #the allowed instance types are: ml.p4d.24xlarge, ml.p3dn.24xlarge, and ml.p3.16xlarge
                            instance_count=1, #distribution={"smdistributed": { "dataparallel": { "enabled": True } } },
                            volume_size=30,
                            max_run=60*60*72,
                            input_mode='Pipe', #'File' or 'Pipe' 
                            output_path='s3://whteamd/{}/'.format(name),
                            source_dir='./src/',
                            base_job_name=name,
                            session=sagemaker_session,
                            use_spot_instances=False,# max_wait=60*60*72,
                            checkpoint_s3_uri=s3_output_dir+'checkpoints/',
                            checkpoint_local_path=local_output_dir+'checkpoints/', #model_uri='s3://whteamd/dino_vitbase8_pretrain_full_checkpoint.pth',
                            tensorboard_output_config=tensorboard_output_config,
                            framework_version='1.8.0',
                            py_version='py36',
                            hyperparameters = hyperparameters,
                                disable_profiler = True,
                                metric_definitions=[
                                    {'Name': 'train-loss', 'Regex': 'loss ([\d\.-]+)'}
                                ])