import torch
from collections import OrderedDict

my_model = torch.load('/home/xie.yim/repo/test_planarrecon/PlanarRecon/checkpoints/release/model_000068.ckpt')
official_model = torch.load('/home/xie.yim/repo/test_planarrecon/PlanarRecon/checkpoints/release/model_000068.ckpt')

official_model = official_model['model']
new_weight_dict = OrderedDict()
for key, value in official_model.items():
    if 'module.neucon_net.' in key:
        key = 'module.fragment_net' + key[17:]
    new_weight_dict[key] = value

my_model['model'] = new_weight_dict
torch.save(my_model, '/home/xie.yim/repo/test_planarrecon/PlanarRecon/checkpoints/release/model_000068_convert.ckpt')