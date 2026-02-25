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
        return self.compute_metrics(y_pred, y_pred_proba, y_true)
    
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
        train_metrics = []
        test_metrics = []
        best_state_dict = None
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                loss.backward()
                opt.step()
            # =========================================================================

            # =====================   validation/test iteration   =====================
            model.eval() # set to eval mode

            # train/val
            for split, loader in [("train", train_loader), ("valid", valid_loader)]:
                metrics = self.evaluate(model, loader, self.bsize, device)
                for m, v in metrics.items():
                    train_metrics.append({"split": split, "metric": m, "value": v, "step": epoch})
            
            # test
            metrics = self.evaluate(model, test_loader, self.bsize, device)
            for m, v in metrics.items():
                test_metrics.append({"split": "test", "metric": m, "value": v, "step": epoch})

            # save checkpoint if current validation loss is lowest
            validation_loss = pandas.DataFrame(train_metrics)
            validation_loss = validation_loss[(validation_loss["metric"] == "log_loss") & (validation_loss["split"] == "valid")]
            validation_loss = validation_loss["value"].values
            if validation_loss[-1] == min(validation_loss):
                best_state_dict = model.state_dict()
            # =========================================================================

            # update learning rate
            if scheduler:
                scheduler.step()
        
        return train_metrics, test_metrics, best_state_dict
        

class CoxTrainer(BaseTrainer):
    def __init__(self, cox_strategy, epochs, bsize, optimizer_params, epsilon=1e-5, **kwargs):
        self.cox_strategy = cox_strategy
        self.epochs = epochs
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

        if self.optimizer_params["name"] == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=float(self.optimizer_params["initial_lr"]), weight_decay=0.1)
        else:
            opt = torch.optim.SGD(model.parameters(), lr=float(self.optimizer_params["initial_lr"]), weight_decay=0.1)
        
        if self.optimizer_params["lr_scheduler"]:
            # create learning rate scheduler (cosine annealing with warmup)
            total_steps = int(self.epochs * np.ceil(len(train_loader) / self.bsize))
            warmup = self.optimizer_params["warmup"] / self.epochs
            div_factor = float(self.optimizer_params["max_lr"]) / float(self.optimizer_params["initial_lr"])
            final_div_factor = float(self.optimizer_params["initial_lr"]) / float(self.optimizer_params["final_lr"])
            scheduler  = torch.optim.lr_scheduler.OneCycleLR(opt, 
                                                             max_lr=float(self.optimizer_params["max_lr"]),
                                                             total_steps=total_steps, 
                                                             pct_start=warmup, 
                                                             anneal_strategy='cos',
                                                             div_factor=div_factor,
                                                             final_div_factor=final_div_factor)
        else:
            scheduler = None
        
        for _ in tqdm.trange(self.epochs, ncols=100):
            for batch in train_loader.batch_iterator(self.bsize):
                if batch is StopIteration:
                    break
                
                neg, pos = batch
                if self.cox_strategy == "1v1":
                    neg = model(send_to_device(neg, device))
                    pos = model(send_to_device(pos, device))
                    cox_loss = torch.mean(-torch.log(torch.exp(neg) / (torch.exp(pos) + self.epsilon)))
                else:
                    neg = model(send_to_device(neg, device))
                    predict_fn = lambda t: model(send_to_device(t, device))
                    exp_sum_fn = lambda t: torch.exp(t).sum(dim=0)
                    pos = map(predict_fn, pos)
                    pos = map(exp_sum_fn, pos)
                    pos = torch.stack(tuple(pos), dim=0)
                    cox_loss = torch.mean(-torch.log(torch.exp(neg) / (pos + self.epsilon)))
                opt.zero_grad()
                cox_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                opt.step()
                if scheduler:
                    scheduler.step()
    
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
            loss = 0
            for q, proto in ((pos_queries, pos_proto), (neg_queries, neg_proto)):
                all_proto = torch.stack((pos_proto, neg_proto), dim=0)
                all_proto = torch.repeat_interleave(all_proto, q.shape[0], dim=1)
                total_dist = torch.exp(-self.dist(q, all_proto)).sum(dim=0)
                loss += (1/N) * torch.sum(-torch.log(torch.exp(-self.dist(q, proto))/(total_dist + self.epsilon)))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
            opt.step()
            if scheduler:
                scheduler.step()

    def dist(self, a, b, type="euclidean"):
        match type:
            case "euclidean":
                return torch.sum((a-b)**2, dim=-1)
            case _:
                return torch.sum((a-b)**2, dim=-1)
