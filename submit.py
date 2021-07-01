import os
import subprocess

import json
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig

sagemaker_session = sagemaker.Session()

# bucket = sagemaker_session.default_bucket()
# prefix = "sagemaker/DEMO-pytorch-cnn-cifar10"

#role = sagemaker.get_execution_role()


# instance_type = "local"

# try:
#     if subprocess.call("nvidia-smi") == 0:
#         ## Set type to GPU if one is present
#         instance_type = "local_gpu"
# except:
#     pass

# print("Instance type = " + instance_type)

name='vit-base-8-afex-1k'
local_output_dir = '/home/ubuntu/dino/output/'
s3_output_dir='s3://whteamd/{}/output/'.format(name)

tensorboard_output_config = tensorboard_output_config=TensorBoardOutputConfig(
                                s3_output_path=s3_output_dir,
                                container_local_output_path=local_output_dir
                            )

# epochs: 300 to 1
# nodes: 22 to 2

pytorch_estimator = PyTorch('train.py',
                            role='arn:aws:iam::338458051672:role/service-role/AmazonSageMaker-ExecutionRole-20210629T090924',
                            instance_type='ml.p3.16xlarge', #the allowed instance types are: ml.p4d.24xlarge, ml.p3dn.24xlarge, and ml.p3.16xlarge
                            distribution={"smdistributed": { "dataparallel": { "enabled": True } } },
                            instance_count=2,
                            volume_size=30,
                            max_run=60*60*72,
                            input_mode='Pipe', #'File', 
                            output_path='s3://whteamd/{}/'.format(name),
                            base_job_name=name,
                            session=sagemaker_session,
                            use_spot_instances=False, #max_wait=60*20,
                            model_uri='s3://whteamd/dino_vitbase8_pretrain_full_checkpoint.pth',
                            checkpoint_s3_uri=s3_output_dir+'checkpoints/',
                            checkpoint_local_path=local_output_dir+'checkpoints/',
                            tensorboard_output_config=tensorboard_output_config,
                            source_dir='/home/ubuntu/dino/src/',
                            framework_version='1.8.0',
                            py_version='py36',
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
                                "global_crops_scale": [0.25, 1.0], 
                                "local_crops_scale": [0.05, 0.25], 
                                "local_crops_number": 10, 
                                "seed": 0, 
                                "num_workers": 10, 
                                "world_size": 176, 
                                "ngpus": 8, 
                                "nodes": 2, 
                                "optimizer": "adamw", 
                                "momentum_teacher": 0.996, 
                                "use_bn_in_head": false
                                }'''),
                                disable_profiler = True,
                                metric_definitions=[
                                    {'Name': 'train-loss', 'Regex': 'loss ([\d\.-]+)'}
                                ])
pytorch_estimator.fit({'train': 's3://cortex-poe/poe-sup-contrast/datasets/baby-dataset-0427-v3/splits/train/',
                       'test': 's3://cortex-poe/poe-sup-contrast/datasets/baby-dataset-0427-v3/splits/test/'})