import tqdm
import torch
import torch.nn.functional as F
import pandas
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, balanced_accuracy_score, f1_score, log_loss, confusion_matrix


def send_to_device(x, device):
    if isinstance(x, dict):
        return {k: send_to_device(v, device) for k,v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    else:
        return torch.tensor(x).to(device)

def zero_division(num, den):
    try:
        if den == 0:
            return 0
        else:
            return num / den
    except TypeError:
        return 0
    
class BaseTrainer:
    def __init__(self):
        pass

    def evaluate(self, model, loader, batch_size, device):
        y_pred_proba = []
        y_pred = []
        y_true = []
        for batch in loader.batch_iterator(batch_size):
            if batch is StopIteration:
                break
            x, y, _ = batch
            with torch.no_grad():
                pred_proba = model(send_to_device(x, device)).to("cpu")
                pred_proba = F.sigmoid(pred_proba)
                y_pred_proba.append(pred_proba.flatten())
                y_pred.append(torch.round(pred_proba.flatten()))
                y_true.append(y)
        y_pred = torch.cat(y_pred, dim=0)
        y_pred_proba = torch.cat(y_pred_proba, dim=0)
        y_true = torch.cat(y_true, dim=0)
        return self.compute_metrics(y_pred, y_pred_proba, y_true), (y_true.cpu().numpy().tolist(), y_pred_proba.cpu().numpy().tolist())
    
    def compute_metrics(self, y_pred, y_pred_proba, y):
        cm = confusion_matrix(y, y_pred, labels=[0,1]).ravel()

        # handle case where only one label in y and y_pred
        # set all confusion matrix element to zero
        # keeping only the element relative to class present in y
        if len(cm) == 1:
            count = cm[0]
            cm = np.zeros((2,2), dtype=np.int64)
            i = int(np.unique(y).item())
            j = int(np.unique(y_pred).item())
            cm[i,j] = count
            cm = cm.ravel()

        tn, fp, fn, tp = cm
        m = {"acc": accuracy_score(y, y_pred),
                "auc": roc_auc_score(y, y_pred_proba),
                "balanced_accuracy": balanced_accuracy_score(y, y_pred),
                "f1_score": f1_score(y, y_pred, zero_division=0),
                "specificity": zero_division(tn, (tn + fp)),
                "sensitivity": zero_division(tp, (tp + fn)),
                "log_loss": log_loss(y, y_pred_proba, labels=[0,1])}
        return m

class FineTuneTrainer(BaseTrainer):
    def __init__(self, epochs, bsize, optimizer_params, lr_scheduler=False, class_weights=False, **kwargs):
        self.epochs = epochs
        self.bsize = bsize
        self.optimizer_params = optimizer_params
        self.lr_scheduler = lr_scheduler
        self.class_weights = class_weights

    def train(self, model, device, train_loader, valid_loader=None, test_loader=None):
        """
        Docstring for train
        
        :param model: (nn.Module) model to train
        :param device: (str, torch.device) device for model training
        :param train_loader: (DataLoader) dataloader for training data
        """
        # set model to train mode
        model.train()

        # define optimizer
        if self.optimizer_params["name"] == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=self.optimizer_params["lr"], weight_decay=0.1)
        else:
            opt = torch.optim.SGD(model.parameters(), lr=self.optimizer_params["lr"], weight_decay=0.1)

        # divide learning rate by 2 each 5 epochs
        if self.lr_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=5, gamma=0.5)
        else:
            scheduler = None

        if self.class_weights:
            train_loader.cw = train_loader.comput_class_weights()

        # send model to device
        model.to(device)
        
        # train/test loop
        print("training binary classifier...")
        metrics = []
        best_state_dict = None
        test_pred_proba = []
        for epoch in tqdm.trange(self.epochs, ncols=100):
            
            # =====================   train iteration   =====================
            model.train()   # set to train mode

            for batch in train_loader.batch_iterator(self.bsize, sample_weight=self.class_weights):
                # stop if StopIteration returned
                if batch is StopIteration:
                    break

                x, y, cw = batch

                if not(cw is None):
                    cw = cw.to(device=device)

                # compute batch loss and update model parameters with gradient clippping
                pred = F.sigmoid(model(send_to_device(x, device))).view(-1)
                y = y.view(*pred.shape).to(device=device, dtype=torch.float32)
                opt.zero_grad()
                loss = F.binary_cross_entropy(pred, y, weight=cw)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.)
                loss.backward()
                opt.step()
            # =========================================================================

            # =====================   validation/test iteration   =====================
            model.eval() # set to eval mode

            # train/val
            for split, loader in [("train", train_loader), ("valid", valid_loader), ("test", test_loader)]:
                split_metrics, (y_true, y_pred_proba) = self.evaluate(model, loader, self.bsize, device)
                for m, v in split_metrics.items():
                    metrics.append({"split": split, "metric": m, "value": v, "step": epoch})
                
                if split == "test":
                    test_pred_proba.append({"step": epoch, "y_true": y_true, "y_pred_proba": y_pred_proba})
            
            # save checkpoint if current validation loss is lowest
            validation_loss = pandas.DataFrame(metrics)
            validation_loss = validation_loss[(validation_loss["metric"] == "log_loss") & (validation_loss["split"] == "valid")]
            validation_loss = validation_loss["value"].values
            if validation_loss[-1] == min(validation_loss):
                best_state_dict = model.state_dict()
            # =========================================================================

            # update learning rate
            if scheduler:
                scheduler.step()
        
        return metrics, best_state_dict, test_pred_proba
        

class CoxProtoNetTrainer(BaseTrainer):
    def __init__(self, n_iter, bsize, optimizer_params, epsilon=1e-5, **kwargs):
        self.n_iter = n_iter
        self.bsize = bsize
        self.optimizer_params = optimizer_params
        self.epsilon = epsilon

    def train(self, model, device, train_loader):
        """
        Docstring for train
        
        :param model: (nn.Module) model to train
        :param device: (str, torch.device) device for model training
        :param train_loader: (DataLoader) dataloader for training data
        """
        # set model to train mode
        model.train()

        # define optimizer
        if self.optimizer_params["name"] == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=float(self.optimizer_params["initial_lr"]), weight_decay=0.1)
        else:
            opt = torch.optim.SGD(model.parameters(), lr=float(self.optimizer_params["initial_lr"]), weight_decay=0.1)
        
        if self.optimizer_params["lr_scheduler"]:
            # create learning rate scheduler (cosine annealing with warmup)
            warmup = self.optimizer_params["warmup"] / self.n_iter
            div_factor = float(self.optimizer_params["max_lr"]) / float(self.optimizer_params["initial_lr"])
            final_div_factor = float(self.optimizer_params["initial_lr"]) / float(self.optimizer_params["final_lr"])
            scheduler  = torch.optim.lr_scheduler.OneCycleLR(opt,
                                                             max_lr=float(self.optimizer_params["max_lr"]),
                                                             total_steps=self.n_iter, 
                                                             pct_start=warmup, 
                                                             anneal_strategy='cos',
                                                             div_factor=div_factor,
                                                             final_div_factor=final_div_factor)
        else:
            scheduler = None
        
        loss = []
        for _ in tqdm.trange(self.n_iter, ncols=100):
            batch = train_loader.get_random_batch(self.bsize)
            if batch is StopIteration:
                    break
            pos, neg = batch
            
            # compute later feature representation
            pos = model(send_to_device(pos, device))
            neg = model(send_to_device(neg, device))

            # compute class prototypes
            pos_proto = pos.mean(dim=0, keepdim=True)
            neg_proto = neg.mean(dim=0, keepdim=True)

            # distances between instances
            pairwise_dist = torch.sum((pos[:,None,:] - neg[None,:,:])**2, dim=-1).sqrt()
            
            # positive instances loss
            pos_proto_dist = self.prototype_dist(pos, pos_proto)
            pos_loss = -torch.log(torch.exp(-pos_proto_dist)/(torch.exp(-pos_proto_dist) + torch.exp(-pairwise_dist).sum(dim=1) + self.epsilon))

            # negative instances loss
            neg_proto_dist = self.prototype_dist(neg, neg_proto)
            neg_loss = -torch.log(torch.exp(-neg_proto_dist)/(torch.exp(-neg_proto_dist) + torch.exp(-pairwise_dist).sum(dim=0) + self.epsilon))

            # total loss
            cox_proto_loss = torch.cat((pos_loss, neg_loss)).mean()
            
            # update model parameters with gradient clippping
            opt.zero_grad()
            cox_proto_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.)
            opt.step()
            if scheduler:
                scheduler.step()
            loss.append(cox_proto_loss.cpu().detach().item())
        return loss
    
    def prototype_dist(self, instances, proto):
        return torch.sum((instances - proto)**2, dim=-1).sqrt()
    
class ProtoNetTrainer:
    def __init__(self, n_iter, bsize, optimizer_params, epsilon=1e-5, **kwargs):
        self.n_iter = n_iter
        self.bsize = bsize
        self.optimizer_params = optimizer_params
        self.epsilon = epsilon
    
    def train(self, model, device, train_loader):
        """
        Docstring for train
        
        :param model: (nn.Module) model to train
        :param device: (str, torch.device) device for model training
        :param train_loader: (DataLoader) dataloader for training data
        """
        # set model to train mode
        model.train()

        # define optimizer
        if self.optimizer_params["name"] == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=float(self.optimizer_params["initial_lr"]), weight_decay=0.1)
        else:
            opt = torch.optim.SGD(model.parameters(), lr=float(self.optimizer_params["initial_lr"]), weight_decay=0.1)
        
        if self.optimizer_params["lr_scheduler"]:
            # create learning rate scheduler (cosine annealing with warmup)
            warmup = self.optimizer_params["warmup"] / self.n_iter
            div_factor = float(self.optimizer_params["max_lr"]) / float(self.optimizer_params["initial_lr"])
            final_div_factor = float(self.optimizer_params["initial_lr"]) / float(self.optimizer_params["final_lr"])
            scheduler  = torch.optim.lr_scheduler.OneCycleLR(opt,
                                                             max_lr=float(self.optimizer_params["max_lr"]),
                                                             total_steps=self.n_iter, 
                                                             pct_start=warmup, 
                                                             anneal_strategy='cos',
                                                             div_factor=div_factor,
                                                             final_div_factor=final_div_factor)
        else:
            scheduler = None
        
        loss = []
        for _ in tqdm.trange(self.n_iter, ncols=100):
            batch = train_loader.get_random_batch(self.bsize)
            if batch is StopIteration:
                    break
            
            (pos_queries, pos_proto), (neg_queries, neg_proto) = batch
            
            # computes queries feature
            pos_queries = model(send_to_device(pos_queries, device))
            neg_queries = model(send_to_device(neg_queries, device))

            # computes class prototypes feature
            proto = {m: {f: torch.cat((pos_proto[m][f], neg_proto[m][f]), dim=0) for f in pos_proto[m].keys()} if m == "image" else torch.cat((pos_proto[m], neg_proto[m]), dim=0) for m in pos_proto.keys()}
            proto = model(send_to_device(proto, device))
            pos_proto = proto[:1]
            neg_proto = proto[1:]

            # compute total loss
            N = pos_queries.shape[0] + neg_queries.shape[0]
            proto_loss = 0
            for q, proto in ((pos_queries, pos_proto), (neg_queries, neg_proto)):
                all_proto = torch.stack((pos_proto, neg_proto), dim=0)
                all_proto = torch.repeat_interleave(all_proto, q.shape[0], dim=1)
                total_dist = torch.exp(-self.dist(q, all_proto)).sum(dim=0)
                proto_loss += (1/N) * torch.sum(-torch.log(torch.exp(-self.dist(q, proto))/(total_dist + self.epsilon)))
            opt.zero_grad()
            proto_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.)
            opt.step()
            if scheduler:
                scheduler.step()
            loss.append(proto_loss.cpu().detach().item())
        return loss

    def dist(self, a, b, type="euclidean"):
        match type:   #TODO implement other distance functions
            case "euclidean":
                return torch.sum((a-b)**2, dim=-1).sqrt()
            case _:
                return torch.sum((a-b)**2, dim=-1).sqrt()
