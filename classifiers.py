from abc import ABC, abstractmethod
import torch
from torch import nn
import torch.nn.functional as F


class BaseBackbone(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(self, input):
        pass

    @abstractmethod
    def save(self, path_):
        pass

    @abstractmethod
    def load(self, path_):
        pass

    @abstractmethod
    def get_num_params(self):
        pass

    @abstractmethod
    def get_out_dim(self):
        pass


class FFN(nn.Module, BaseBackbone):
    def __init__(self, in_dim, hidden_dim, out_dim, dropout=0.3):
        """"
        Args
            in_dim (int) dimension of input features
            n_class (int) number of classes
        """
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def save(self, path_):
        torch.save(self.state_dict(), path_)

    def load(self, path_):
        self.load_state_dict(torch.load(path_))

    def get_num_params(self):
        get_n = lambda i: sum(p.numel() for p in i.parameters() if p.requires_grad)
        return get_n(self.head)

    def forward(self, x):
        if isinstance(x, dict):
            x = list(x.values())[0]
            if isinstance(x, dict):  # image modality
                x = list(x.values())[0]

        return self.head(x)

    def __call__(self, input):
        return self.forward(input)
    
    def get_out_dim(self):
        return self.head[-1].out_features


class Attention(nn.Module, BaseBackbone):
    def __init__(self, dims, hidden_dim, n_layer, out_dim, lambda_=0.5, dropout=0.3, **args):
        """"
        Args
            dims (dict) dict of input dimension for each modality and features ({"m": {"f1": n1, "f2": n2}, ...}})
            hidden_dim (int) internal dimension of vectors
            n_layer (int) number of attention fusion layers
            out_dim (int) dimension of output features
            lambda_ (float) trade-off parameter for feature vector update
            dropout (float) dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layer = n_layer
        self.lambda_ = lambda_

        self.layers = nn.ModuleDict({
            "proj": nn.ModuleDict({m: nn.ModuleDict({k: nn.Linear(v, hidden_dim) for k,v in feat.items()}) if isinstance(feat, dict) else nn.Linear(feat, hidden_dim) for m, feat in dims.items()}),
            
            "FFN": nn.Sequential(
                nn.Linear(hidden_dim, 4*hidden_dim),
                nn.GELU(),
                nn.Linear(4*hidden_dim, hidden_dim),
                nn.Dropout(dropout),
            ),
            "ln": nn.LayerNorm(hidden_dim),
            "dropout": nn.Dropout(dropout),
            "out": nn.Linear(hidden_dim, out_dim),
        })

        self.queries = nn.ParameterDict({
            "q_mi": nn.ParameterDict({m: nn.Parameter(torch.randn(hidden_dim)) for m in dims.keys()}),
            "q": nn.Parameter(torch.randn(hidden_dim)),
        })

    def save(self, path_):
        torch.save(self.state_dict(), path_)

    def load(self, path_):
        self.load_state_dict(torch.load(path_))

    def get_num_params(self):
        get_n = lambda i: sum(p.numel() for p in i.parameters() if p.requires_grad)
        return get_n(self.layers) + get_n(self.queries)

    def forward(self, x: dict):
        """
        Args
            x (dict) dict of input tensors for each modality and features ({"m": {"f1": tensor1, "f2": tensor2}, ...}})
        Returns
            out (tensor) output logits
        """

        B = list(x["image"].values())[0].shape[0]  # batch size

        # project and stack modalities features
        proj = self.layers["proj"]
        for m in x.keys():
            if isinstance(x[m], dict):
                for k in x[m].keys():
                    x[m][k] = proj[m][k](x[m][k])
                x[m] = torch.stack(tuple(x[m].values()), dim=1)
            else:
                x[m] = proj[m](x[m])

        # initialize query vectors
        q_mi = {k: v.unsqueeze(dim=0).repeat(B, 1) for k,v in self.queries["q_mi"].items()}
        q = self.queries["q"].unsqueeze(dim=0).repeat(B, 1)
        
        for _ in range(self.n_layer):
            # intra-modality fusion
            for m in x.keys():
                if m == "image": # no intra-modality attention for clinical modality
                    attn_scores = torch.matmul(q_mi[m].unsqueeze(dim=-2), x[m].transpose(-2, -1)) / (self.hidden_dim ** 0.5)
                    attn_scores = F.softmax(attn_scores, dim=-1)
                    q_mi[m] = torch.matmul(attn_scores, x[m]).squeeze(dim=-2)
                
                    # propagate modality query into feature vectors
                    x[m] = self.lambda_ * x[m] + (1 - self.lambda_) * q_mi[m].unsqueeze(dim=-2).repeat(1, x[m].shape[1], 1)
                else:
                    # propagate modality query into feature vectors
                    x[m] = self.lambda_ * x[m] + (1 - self.lambda_) * q_mi[m]
                
                # layer norm and dropout
                x[m] = self.layers["dropout"](self.layers["ln"](x[m]))
                
                # FFN
                x[m] = self.layers["FFN"](x[m])

            # inter-modality fusion
            modality_features = torch.stack([q_mi[m] for m in x.keys()], dim=1)
            attn_scores = torch.matmul(q.unsqueeze(dim=-2), modality_features.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
            attn_scores = F.softmax(attn_scores, dim=-1)
            q = torch.matmul(attn_scores, modality_features).squeeze(dim=-2)

            # propagate fused modalities into modalities feature vectors
            for m in q_mi.keys():
                q_mi[m] = self.lambda_ * q_mi[m] + (1 - self.lambda_) * q
                q_mi[m] = self.layers["ln"](self.layers["dropout"](q_mi[m]))

        # classification head
        out = self.layers["out"](q)
        return out

    def __call__(self, input):
        return self.forward(input)
    
    def get_out_dim(self):
        return self.layers["out"].out_features
    

class Concat(nn.Module, BaseBackbone):
    def __init__(self, dims, hidden_dim, out_dim, dropout=0.3, **args):
        """"
        Args
            dims (dict) dict of dimension for each modality and features ({"m": {"f1": n1, "f2": n2}, ...}})
            hidden_dim (int) projection dimension after concatenation of all features of modality
            out_dim (int) dimension of output features
        """
        super().__init__()
        # project only modalities with multiple features
        self.proj = nn.ModuleDict({m: nn.Linear(sum(feat.values()), hidden_dim) if isinstance(feat, dict) else nn.Linear(feat, hidden_dim) for m, feat in dims.items()})
        self.ffn = FFN(len(dims)*hidden_dim, hidden_dim, out_dim, dropout=dropout)

    def save(self, path_):
        torch.save(self.state_dict(), path_)

    def load(self, path_):
        self.load_state_dict(torch.load(path_))

    def get_num_params(self):
        get_n = lambda i: sum(p.numel() for p in i.parameters() if p.requires_grad)
        return get_n(self.proj) + get_n(self.ffn)

    def forward(self, x):
        """
        Args
            x (dict) dict of input tensors for each modality and features ({"m": {"f1": tensor1, "f2": tensor2}, ...}})
        """
        proj = {m: self.proj[m](torch.cat(tuple(x[m].values()), dim=-1)) if isinstance(x[m], dict) else self.proj[m](x[m]) for m in x.keys()}
        x = self.ffn(torch.cat(tuple(proj.values()), dim=-1))
        return x

    def __call__(self, input):
        return self.forward(input)
    
    def get_out_dim(self):
        return self.ffn.get_out_dim()


class GatedModality(nn.Module, BaseBackbone):
    def __init__(self, dims, hidden_dim, out_dim, dropout=0.3, **args):
        """"
        Args
            dims (dict) dict of input dimension for each modality and features ({"m": {"f1": n1, "f2": n2}, ...}})
            hidden_dim (int) projection dimension after concatenation of all features of modality
            out_dim (int) dimension of output features
        """
        super().__init__()
        n_modality = len(dims)
        self.hidden_dim = hidden_dim
        self.ffn = nn.ModuleDict({m: FFN(sum(feat.values()), 3*hidden_dim, hidden_dim) if isinstance(feat, dict) else FFN(feat, 3*hidden_dim, hidden_dim) for m, feat in dims.items()})
        self.bn = nn.BatchNorm1d(n_modality * hidden_dim)
        self.gates = nn.Linear(n_modality * hidden_dim, n_modality)
        self.out = nn.Linear(n_modality * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def save(self, path_):
        torch.save(self.state_dict(), path_)

    def load(self, path_):
        self.load_state_dict(torch.load(path_))

    def get_num_params(self):
        get_n = lambda i: sum(p.numel() for p in i.parameters() if p.requires_grad)
        return get_n(self.ffn) + get_n(self.gates) + get_n(self.out)

    def forward(self, x):
        """
        Args
            x (dict) dict of input tensors for each modality and features ({"m": {"f1": tensor1, "f2": tensor2}, ...}})
        """
        modality_features = {m: self.ffn[m](torch.cat(tuple(x[m].values()), dim=-1)) if isinstance(x[m], dict) else self.ffn[m](x[m]) for m in x.keys()}
        modality_features = torch.cat(tuple(modality_features.values()), dim=-1)
        modality_features = self.bn(modality_features)
        gates = torch.sigmoid(self.gates(modality_features))
        gated_features = modality_features * gates.repeat_interleave(self.hidden_dim, dim=-1)
        out = self.out(self.dropout(gated_features))
        return out

    def __call__(self, input):
        return self.forward(input)
    
    def get_out_dim(self):
        return self.out.out_features


class CoxModel(nn.Module):
    """
    Class to build a model to train a backbone using Cox partial likelihood or ProtoNet loss.
    Args
        backbone (BaseBackbone) backbone to use for survival analysis
    """
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.get_out_dim(), 1)

    def forward(self, x):   
        return self.head(self.backbone(x))

    def __call__(self, input):
        return self.forward(input)

    
class Classifier(nn.Module):
    """
    Class to build a classifier based on a backbone. 
    Args
        backbone (BaseBackbone) backbone to use for classification
        freeze_backbone (bool) whether to freeze backbone parameters during training
    """
    def __init__(self, backbone, n_class=1, freeze_backbone=True):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.get_out_dim(), n_class)

        # freeze backbone parameters
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.head(self.backbone(x))

    def __call__(self, input):
        return self.forward(input)