# Copyright 2024 BDP Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import os.path

parser = argparse.ArgumentParser()
parser.add_argument('--path', default=r'results/v1/checkpoint-best-loss=0.3184.msgpack')
parser.add_argument('--devices', type=str, default='0', help='The GPU device ids.')
parser.add_argument('--n_predict', type=int, default=None, help='The GPU device ids.')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.devices

import os.path
import brainstate
import brainunit as u
import brainevent

brainevent.config.gpu_kernel_backend = 'pallas'
from models import DrosophilaSpikingNetworkTrainer, load_setting

settings = load_setting(os.path.dirname(args.path))
settings.batch_size = 4
if args.n_predict is not None:
    settings.n_fr_per_train = args.n_predict

brainstate.environ.set(dt=settings.dt * u.ms)
trainer = DrosophilaSpikingNetworkTrainer(settings)
trainer.f_eval(checkpoint_path=args.path)
