### Setup
```
pip install -e . 
python download_convert_models.py 
# can modify to download different models, by default it downloads all 5 ViTs pretrained on ImageNet21k 
```
### Usage
``` 
from vit_animesion import ViT, ViTConfigExtended, PRETRAINED_CONFIGS 
model_name = 'B_16' 
def_config = PRETRAINED_CONFIGS['{}'.format(model_name)]['config'] 
configuration = ViTConfigExtended(**def_config) 
model = ViT(configuration, name=model_name, pretrained=True, 
load_repr_layer=False, ret_attn_scores=False) 
```
