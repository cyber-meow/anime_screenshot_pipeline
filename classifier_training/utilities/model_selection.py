import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from einops.layers.torch import Rearrange

from efficientnet_pytorch import EfficientNet
from vit_animesion import ViT, ViTConfigExtended, PRETRAINED_CONFIGS


def load_model(args, device):
    # initiates model and loss
    if args.model_name == 'shallow':
        model = ShallowNet(args)
    elif args.model_name in ['resnet18', 'resnet50', 'resnet152']:
        model = ResNet(args)
    elif args.model_name == 'efficientnetb0':
        model = EffNet(args)
    else:
        model = VisionTransformer(args)
    # print(model)

    if args.checkpoint_path:
        if args.load_partial_mode:
            model.model.load_partial(weights_path=args.checkpoint_path,
                                     pretrained_image_size=model.configuration.
                                     pretrained_image_size,
                                     pretrained_mode=args.load_partial_mode,
                                     verbose=True)
        else:
            state_dict = torch.load(args.checkpoint_path,
                                    map_location=torch.device('cpu'))
            # state_dict = torch.load(args.checkpoint_path)
            expected_missing_keys = []
            if args.transfer_learning:
                # Modifications to load partial state dict
                if ('inter_class_head.4.weight' in state_dict):
                    expected_missing_keys += [
                        'inter_class_head.4.weight', 'inter_class_head.4.bias'
                    ]
                if ('class_head.1.weight' in state_dict):
                    expected_missing_keys += [
                        'class_head.1.weight', 'class_head.1.bias'
                    ]
                if ('model.fc.weight' in state_dict):
                    expected_missing_keys += [
                        'model.fc.weight', 'model.fc.bias'
                    ]
                for key in expected_missing_keys:
                    state_dict.pop(key)
                    # print(key)
            ret = model.load_state_dict(state_dict, strict=False)
            print('''Missing keys when loading pretrained weights: {}
                    Expected missing keys: {}'''.format(
                ret.missing_keys, expected_missing_keys))
            print('Unexpected keys when loading pretrained weights: {}'.format(
                ret.unexpected_keys))
            print('Loaded from custom checkpoint.')

    model.to(device)

    return model


def freeze_layers(model):
    for param in model.parameters():
        param.requires_grad = False


class ShallowNet(nn.Module):
    def __init__(self, args, fc_neurons=512):
        # input size: batch_sizex3x224x224
        super(ShallowNet, self).__init__()
        self.num_classes = args.num_classes
        self.fc_neurons = fc_neurons
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(kernel_size=2,
                                                        stride=2))
        # output size from conv2d: batch_sizex16x220x220
        # output size from maxpool2d: batch_sizex16x110x110
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(kernel_size=2,
                                                        stride=2))
        # output size from conv2d: batch_sizex32x106x106
        # output size from maxpool2d: batch_sizex32x53x53
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(53 * 53 * 32, self.fc_neurons),
                                nn.ReLU(), nn.Dropout(p=0.2),
                                nn.Linear(self.fc_neurons, self.num_classes))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.fc(out)
        return out


class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()

        if args.model_name == 'resnet18':
            base_model = models.resnet18(pretrained=args.pretrained,
                                         progress=True)
        elif args.model_name == 'resnet50':
            base_model = models.resnet50(pretrained=args.pretrained,
                                         progress=True)
        elif args.model_name == 'resnet152':
            base_model = models.resnet152(pretrained=args.pretrained,
                                          progress=True)
        self.model = base_model

        # Initialize/freeze weights
        # originally for pretrained would freeze all layers except last
        # if args.pretrained:
        #    freeze_layers(self.model)
        # else:
        if not args.pretrained:
            self.init_weights()

        # Classifier head
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, args.num_classes)

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

        self.apply(_init)
        nn.init.constant_(self.model.fc.weight, 0)
        nn.init.constant_(self.model.fc.bias, 0)

    def forward(self, x):
        out = self.model(x)
        return out


class EffNet(nn.Module):
    def __init__(self, args):
        super(EffNet, self).__init__()

        if args.pretrained:
            self.model = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            self.model = EfficientNet.from_name('efficientnet-b0')

        if not args.pretrained:
            self.init_weights()

        # Classifier head
        num_features = self.model._fc.in_features
        self.model._fc = nn.Linear(num_features, args.num_classes)

    @torch.no_grad()
    def init_weights(self):
        def _init(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)

        self.apply(_init)
        nn.init.constant_(self.model._fc.weight, 0)
        nn.init.constant_(self.model._fc.bias, 0)

    def forward(self, x):
        out = self.model(x)
        return out


class VisionTransformer(nn.Module):
    def __init__(self, args):
        super(VisionTransformer, self).__init__()

        def_config = PRETRAINED_CONFIGS['{}'.format(args.model_name)]['config']
        self.configuration = ViTConfigExtended(**def_config)
        self.configuration.num_classes = args.num_classes
        self.configuration.image_size = args.image_size
        self.configuration.max_text_seq_len = args.max_text_seq_len
        if args.vocab_size:
            self.configuration.vocab_size = args.vocab_size

        load_fc_layer = not (args.interm_features_fc) and not (
            args.mask_schedule)
        self.configuration.load_fc_layer = load_fc_layer

        base_model = ViT(self.configuration,
                         name=args.model_name,
                         pretrained=args.pretrained,
                         load_fc_layer=load_fc_layer,
                         ret_interm_repr=args.interm_features_fc,
                         multimodal=args.multimodal,
                         ret_attn_scores=args.ret_attn_scores)
        self.model = base_model

        if not load_fc_layer:
            if args.interm_features_fc:
                self.inter_class_head = nn.Sequential(
                    nn.Linear(self.configuration.num_hidden_layers, 1),
                    Rearrange(' b d 1 -> b d'), nn.ReLU(),
                    nn.LayerNorm(self.configuration.hidden_size,
                                 eps=self.configuration.layer_norm_eps),
                    nn.Linear(self.configuration.hidden_size,
                              self.configuration.num_classes))
                if args.exclusion_loss:
                    self.exclusion_loss = nn.KLDivLoss(reduction='batchmean')
                    self.temperature = args.temperature
                    self.exc_layers_dist = args.exc_layers_dist
            else:  # original cls head but also doing mlm
                self.class_head = nn.Sequential(
                    nn.LayerNorm(self.configuration.hidden_size,
                                 eps=self.configuration.layer_norm_eps),
                    nn.Linear(self.configuration.hidden_size,
                              self.configuration.num_classes))

            if args.mask_schedule:
                # https://github.com/dhlee347/pytorchic-bert/blob/master/pretrain.py
                self.mlm_head = nn.Sequential(
                    nn.Linear(self.configuration.hidden_size,
                              self.configuration.hidden_size),
                    nn.GELU(),
                    nn.LayerNorm(self.configuration.hidden_size,
                                 eps=self.configuration.layer_norm_eps),
                )
                self.text_decoder = nn.Linear(self.configuration.hidden_size,
                                              self.configuration.vocab_size,
                                              bias=False)
                self.text_decoder.weight = self.model.text_embeddings.word_embeddings.weight
                self.decoder_bias = nn.Parameter(
                    torch.zeros(self.configuration.vocab_size))

    def forward(self, images, text=None, mask=None):
        """Breaks image into patches, applies transformer, applies MLP head.
        Args:
            images (tensor): `b,c,fh,fw`
            text (tensor): b, max_text_seq_len
            mask (bool tensor): (B(batch_size) x S(seq_len))
        """

        exclusion_loss = 0

        if hasattr(self, 'inter_class_head'):
            features, interm_features = self.model(images, text, mask)
        elif hasattr(self, 'class_head'):
            features = self.model(images, text, mask)
        else:
            logits = self.model(images, text)

        if hasattr(self, 'inter_class_head'):
            if hasattr(self, 'exclusion_loss'):
                for i in range(len(interm_features) - self.exc_layers_dist):
                    exclusion_loss += self.exclusion_loss(
                        F.log_softmax(interm_features[i][:, 0, :] /
                                      self.temperature,
                                      dim=1),
                        F.softmax(
                            interm_features[i + self.exc_layers_dist][:, 0, :]
                            / self.temperature,
                            dim=1))
            interm_features = torch.stack(interm_features, dim=-1)
            logits = self.inter_class_head(interm_features[:, 0])
        elif hasattr(self, 'class_head'):
            logits = self.class_head(features[:, 0])

        if hasattr(self, 'text_decoder'):
            predicted_text = self.mlm_head(
                features[:, -self.configuration.max_text_seq_len:, :])
            predicted_text = self.text_decoder(
                predicted_text) + self.decoder_bias

        if hasattr(self, 'text_decoder') and hasattr(self, 'exclusion_loss'):
            return logits, predicted_text, exclusion_loss
        elif hasattr(self, 'text_decoder'):
            return logits, predicted_text
        elif hasattr(self, 'exclusion_loss'):
            return logits, exclusion_loss
        return logits
