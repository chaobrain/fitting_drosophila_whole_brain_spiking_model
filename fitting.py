# Copyright 2025 BDP Ecosystem Limited. All Rights Reserved.
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


from setting import get_parser
settings = get_parser()

import brainevent
brainevent.config.gpu_kernel_backend = 'pallas'

import brainstate
import brainunit as u
from models import DrosophilaSpikingNetworkTrainer

brainstate.environ.set(dt=settings.dt * u.ms)

trainer = DrosophilaSpikingNetworkTrainer(settings)
checkpoint_path = 'results/630#multi-step#2017-10-26_1#100.0#0.99#mae#current#200#0.0001#0.7#csr#20.0#20.0#5#5#5#0.1#1#0.2#None##2025-08-16-23-26-28/checkpoint-best-loss=803.0625.msgpack'
checkpoint_path = 'results/630#multi-step#2017-10-26_1#100.0#0.99#mae#current#200#0.0001#0.7#csr#20.0#20.0#5#5#5#0.1#1#0.2#None##2025-08-16-23-26-28/epoch=1000/checkpoint-best-loss=0.6315.msgpack'
checkpoint_path = 'results/630#multi-step#2017-10-26_1#100.0#0.99#mae#current#200#0.0001#0.7#csr#20.0#20.0#5#5#5#0.1#1#0.2#None##2025-08-16-23-26-28/epoch=1000/n_fr_per_warmup=50-n_fr_per_train=50-n_fr_per_gap=50/checkpoint-best-loss=2.3257.msgpack'
checkpoint_path = None
trainer.f_train(train_epoch=settings.epoch, checkpoint_path=checkpoint_path)
