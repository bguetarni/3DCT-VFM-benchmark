from abc import ABC, abstractmethod
from itertools import chain
import torch
from torch import nn
import torch.nn.functional as F
import sklearn

def unfold_dict_tensor(d):
    tensors = []
    for v in d.values():
        if isinstance(v, dict):
            tensors.extend(unfold_dict_tensor(v))
        else:
            tensors.append(v)
    return tensors

class Normalizer:
    def __init__(self, method=None):
        match method:
            case "scale":
                self.normalizer = sklearn.preprocessing.StandardScaler()
            case "mnimax":
                self.normalizer = sklearn.preprocessing.MinMaxScaler(feature_range=(0,1))
            case "unit":
                self.normalizer = sklearn.preprocessing.Normalizer()
            case _:
                self.normalizer = sklearn.preprocessing.StandardScaler()
    
    def fit_transform(self, X):
        X.loc[:,:] = self.normalizer.fit_transform(X.values)
        return X

    def transform(self, X):
        X.loc[:,:] = self.normalizer.transform(X.values)
        return X
    
    def get_params(self):
        return self.normalizer.__getstate__()

    def set_params(self, params):
        self.normalizer.__setstate__(params)

class BaseClassifier(ABC):
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


class MLP_head(nn.Module, BaseClassifier):
    def __init__(self, n_dim, n_class=1, dropout=0.1):
        """"
        Args
            n_dim (int) dimension of input features
            n_class (int) number of classes
        """
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(n_dim, 4*n_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4*n_dim, n_class),
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
            x = unfold_dict_tensor(x)[0]

        return self.head(x)

    def __call__(self, input):
        return self.forward(input)


class Attention(nn.Module, BaseClassifier):
    def __init__(self, dims, n_dim, n_layer, lambda_, dropout=0.5, n_class=1, **args):
        """"
        Args
            dims (dict) dict of dimension for each modality and features ({"m": {"f1": n1, "f2": n2}, ...}})
            n_dim (int) internal dimension of vectors similar to Transformer.n_dim
            n_class (int) number of classes
            lambda_ (float) trade-off parameter for feature vector update
            n_layer (int) number of attention fusion layers
        """
        super().__init__()
        self.n_dim = n_dim
        self.n_layer = n_layer
        self.lambda_ = lambda_

        self.layers = nn.ModuleDict({
            "proj": nn.ModuleDict({m: nn.ModuleDict({k: nn.Linear(v, n_dim) for k,v in feat.items()}) if isinstance(feat, dict) else nn.Linear(feat, n_dim) for m, feat in dims.items()}),
            
            "FFN": nn.Sequential(
                nn.Linear(n_dim, 4*n_dim),
                nn.GELU(),
                nn.Linear(4*n_dim, n_dim),
                nn.Dropout(dropout),
            ),
            "ln": nn.LayerNorm(n_dim),
            "dropout": nn.Dropout(dropout),
            "head": MLP_head(n_dim, n_class, dropout),
        })

        self.queries = nn.ParameterDict({
            "q_mi": nn.ParameterDict({m: nn.Parameter(torch.randn(n_dim)) for m in dims.keys()}),
            "q": nn.Parameter(torch.randn(n_dim)),
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
                x[m] = torch.stack(tuple(x[m].values()), dim=1)  # (B, n_feat, n_dim)
            else:
                x[m] = proj[m](x[m])  # (B, n_dim)

        # initialize query vectors
        q_mi = {k: v.unsqueeze(dim=0).repeat(B, 1) for k,v in self.queries["q_mi"].items()}  # (B, n_dim)
        q = self.queries["q"].unsqueeze(dim=0).repeat(B, 1)  # (B, n_dim)
        
        for _ in range(self.n_layer):
            # intra-modality fusion
            for m in x.keys():
                if m == "image": # no intra-modality attention for clinical modality
                    attn_scores = torch.matmul(q_mi[m].unsqueeze(dim=-2), x[m].transpose(-2, -1)) / (self.n_dim ** 0.5)  # (B, 1, n_feat)
                    attn_scores = F.softmax(attn_scores, dim=-1)
                    q_mi[m] = torch.matmul(attn_scores, x[m]).squeeze(dim=-2)  # (B, n_dim)
                
                    # propagate modality query into feature vectors
                    x[m] = self.lambda_ * x[m] + (1 - self.lambda_) * q_mi[m].unsqueeze(dim=-2).repeat(1, x[m].shape[1], 1)
                else:
                    # propagate modality query into feature vectors
                    x[m] = self.lambda_ * x[m] + (1 - self.lambda_) * q_mi[m]
                
                x[m] = self.layers["ln"](self.layers["dropout"](x[m]))   # dropout and layer norm
                
                # FFN
                x[m] = self.layers["FFN"](x[m])

            # inter-modality fusion
            modality_features = torch.stack([q_mi[m] for m in x.keys()], dim=1)  # (B, n_mod, n_dim)
            attn_scores = torch.matmul(q.unsqueeze(dim=-2), modality_features.transpose(-2, -1)) / (self.n_dim ** 0.5)  # (B, 1, n_mod)
            attn_scores = F.softmax(attn_scores, dim=-1)
            q = torch.matmul(attn_scores, modality_features).squeeze(dim=-2)  # (B, n_dim)

            # propagate fused modalities into modalities feature vectors
            for m in q_mi.keys():
                q_mi[m] = self.lambda_ * q_mi[m] + (1 - self.lambda_) * q
                q_mi[m] = self.layers["ln"](self.layers["dropout"](q_mi[m]))   # dropout and layer norm

        # classification head
        out = self.layers["head"](q)  # (B, n_class)
        return out

    def __call__(self, input):
        return self.forward(input)
    

class Concat(nn.Module, BaseClassifier):
    def __init__(self, dims, n_dim, dropout=0.1, n_class=1, **args):
        """"
        Args
            dims (dict) dict of dimension for each modality and features ({"m": {"f1": n1, "f2": n2}, ...}})
            n_dim (int) projection dimension after concatenation of all features of modality
            n_class (int) number of classes
        """
        super().__init__()
        # project only modalities with multiple features
        self.proj = nn.ModuleDict({m: nn.Linear(sum(feat.values()), n_dim) if isinstance(feat, dict) else nn.Linear(feat, n_dim) for m, feat in dims.items()})
        self.head = MLP_head(len(dims)*n_dim, n_class, dropout)

    def save(self, path_):
        torch.save(self.state_dict(), path_)

    def load(self, path_):
        self.load_state_dict(torch.load(path_))

    def get_num_params(self):
        get_n = lambda i: sum(p.numel() for p in i.parameters() if p.requires_grad)
        return get_n(self.proj) + get_n(self.head)

    def forward(self, x):
        """
        Args
            x (dict) dict of input tensors for each modality and features ({"m": {"f1": tensor1, "f2": tensor2}, ...}})
        """
        proj = {m: self.proj[m](torch.cat(tuple(x[m].values()), dim=-1)) if isinstance(x[m], dict) else self.proj[m](x[m]) for m in x.keys()}  # (B, n_dim)
        out = self.head(torch.cat(tuple(proj.values()), dim=-1))  # (B, n_class)
        return out

    def __call__(self, input):
        return self.forward(input)


class Linear(nn.Module, BaseClassifier):
    def __init__(self, n_dim, dropout=0.1, n_class=1, **args):
        """"
        Args
            n_dim (int) dimension of features
            n_class (int) number of classes
        """
        super().__init__()
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(n_dim, n_class))

    def save(self, path_):
        torch.save(self.state_dict(), path_)

    def load(self, path_):
        self.load_state_dict(torch.load(path_))

    def get_num_params(self):
        get_n = lambda i: sum(p.numel() for p in i.parameters() if p.requires_grad)
        return get_n(self.head)

    def forward(self, x):
        """
        Args
            x (torch.Tensor, dict) input tensor
        """
        if isinstance(x, dict):
            x = unfold_dict_tensor(x)[0]
        
        out = self.head(x)  # (B, n_class)
        return out

    def __call__(self, input):
        return self.forward(input)
