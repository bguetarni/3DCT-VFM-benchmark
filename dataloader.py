import random
from collections.abc import Iterable
from itertools import chain
from more_itertools import collapse, chunked
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

from datasets import cohorts_map, CATEGORICAL_CLINICAL_VARIABLES

def get_normalizer(method="scale"):
    match method:
        case "scale":
            return StandardScaler()
        case "minmax":
            return MinMaxScaler(feature_range=(0,1))
        case "unit":
            return Normalizer()
        case _:
            return StandardScaler()


class Sample:
    def __init__(self, id_, imaging=None, clinical=None, rfs=None, label=None):
        """
        Docstring for __init__
            id_ (str): sample/patient id
            imaging (dict): dictionary of imaging features, with keys as FM and values as feature values
            clinical (dict): dictionary of clinical features, with keys as feature names and values as feature values
            rfs (dict): dictionary with keys "T" and "delta" for RFS time and event indicator
            label (int): binary label for classification task
        """
        self.id_ = id_
        self.imaging = imaging
        self.clinical = clinical
        self.clinical_copy = clinical   # copy to keep track of original clinical variables after tensor converion
        self.rfs = rfs
        self.label = label

    def filter_imaging_features(self, extractors):
        if not(self.imaging is None):
            self.imaging = {k: v for k, v in self.imaging.items() if k in extractors}

    def get_features_dimension(self):
        dim = {}
        
        if self.imaging is None:
            dim.update({"image": None})
        else:
            dim.update({"image": {k: len(v) for k, v in self.imaging.items()}})
        
        if self.clinical is None:
            dim.update({"clinical": None})
        else:
            dim.update({"clinical": sum([len(v) if isinstance(v, Iterable) else 1 for v in self.clinical.values()])})
        
        return dim


class Data:
    def __init__(self, base_path, cohort, samples=None, **kwargs):
        self.base_path = base_path
        self.cohort = cohort
        self.samples = samples

    def __len__(self):
        return len(self.samples) if self.samples else 0

    def load(self, split=None):
        try:
            loader = cohorts_map[self.cohort]
        except KeyError:
            raise ValueError(f"dataset name {self.cohort} not recognized. See args.dataset argument choices.")        
        
        imaging, clinical, rfs = loader(None).get_features_labels(self.base_path)

        patients_id = set([*imaging.keys(), *clinical.keys(), *rfs.keys()])
        self.samples = []
        for p in patients_id:
            # check for RADCURE patients split
            if self.cohort == "radcure" and ((split == "training" and clinical[p]["RADCURE-challenge"] == "test") or \
                                             (split == "test" and str(clinical[p]["RADCURE-challenge"]) in ("training", "0"))):
                    # skip patient if in RADCURE test split while loading training split
                    continue

            img = imaging[p] if p in imaging.keys() else None
            cl = clinical[p] if p in clinical.keys() else None
            rf = rfs[p] if p in rfs.keys() else None
            self.samples.append(Sample(id_=p, imaging=img, clinical=cl, rfs=rf))

    def compute_labels(self, task):
        match task:
            case "rfs_2":
                T = 2
            case "rfs_5":
                T = 5
            case _:
                raise ValueError(f"task {task} not valid. Valid ones are [rfs_2, rfs_5]")
        
        for sample in self.samples:
            if sample.rfs is None or sample.rfs["T"] is None:
                label = None
            elif sample.rfs["T"] < T*365:
                if sample.rfs["delta"] == 1:
                    label = 1
                else:
                    label = None
            else:
                label = 0
            
            # update sample label
            sample.label = label

    def filter_clinical_features(self, clinical_variables):
        for sample in self.samples:
            if sample.clinical is None:
                continue
            sample.clinical = {k: v for k, v in sample.clinical.items() if k in clinical_variables}

    def filter_imaging_features(self, extractors):
        for sample in self.samples:
            sample.filter_imaging_features(extractors)

    def inclusion_criteria_clinical(self, clinical_variable, value, criteria=None):
        if value is None:
            # include patient even if inclusion criteria variable is missing !!
            return True

        match clinical_variable:
            case "dose":
                return 64 <= value and value < 80
            case "metastasis":
                return value == 0
            case "localisation":
                if criteria and "localisation" in criteria.keys():
                    return value in criteria["localisation"]
                else:
                    return value == 0
            case "treatment":
                return value in [0,1]
            case "surgery":
                return value == 0
            case _:
                # if variable not in inclusion criteria list, include it anyways
                return True
    
    def apply_inclusion_criteria(self, criteria=None):
        # filter patients according to inclusion criteria on clinical variables
        self.samples = [s for s in self.samples if all([self.inclusion_criteria_clinical(i, j, criteria) for i, j in s.clinical.items()])]

    def check_imaging(self, sample, exp_params):
        if sample.imaging is None:
            return False
        elif not all([i in sample.imaging.keys() for i in exp_params["extractors"]]):
            return False
        else:
            return True
        
    def check_clinical(self, sample, exp_params):
        if sample.clinical is None:
            return False
        elif not all([k in sample.clinical.keys() for k in exp_params["clinical"]]):
            return False
        elif any([sample.clinical[k] is None for k in exp_params["clinical"]]):
            return False
        else:
            return True

    def filter_patients(self, exp_params):
        # filter patients with missing data for selected modality or label
        # preserve patients with missing label but available RFS time for Cox model pre-training
        invalid_samples_id = []
        cox_samples_id = []
        for s in self.samples:
            if (exp_params["modality"] == "image" and not(self.check_imaging(s, exp_params))) or \
                    (exp_params["modality"] == "clinical" and not(self.check_clinical(s, exp_params))) or \
                        (exp_params["modality"] == "both" and (not(self.check_imaging(s, exp_params)) or not(self.check_clinical(s, exp_params)))):
                # missing data
                invalid_samples_id.append(s.id_)
            elif s.label is None:
                if not(s.rfs is None) and not(s.rfs["T"] is None):
                    cox_samples_id.append(s.id_)
                else:
                    invalid_samples_id.append(s.id_)
            else:
                pass

        # select samples for Cox pre-training
        cox_samples = [s for s in self.samples if s.id_ in cox_samples_id]

        # remove invalid and cox samples from dataset
        self.samples = [s for s in self.samples if not(s.id_ in invalid_samples_id or s.id_ in cox_samples_id)]

        return cox_samples

    def get_features_dimension(self):
        return self.samples[0].get_features_dimension()
    
    def normalize_image_features(self, normalizer=None):
        if isinstance(normalizer, dict):
            for k, norm in normalizer.items():
                x = np.stack([s.imaging[k] for s in self.samples], axis=0)
                x = norm.transform(x)
                for i, sample in enumerate(self.samples):
                    sample.imaging[k] = x[i]
        else:
            normalizer = {}
            for k in self.samples[0].imaging.keys():
                norm = get_normalizer(str(normalizer))
                x = np.stack([s.imaging[k] for s in self.samples], axis=0)
                x = norm.fit_transform(x)
                normalizer[k] = norm
                for i, sample in enumerate(self.samples):
                    sample.imaging[k] = x[i]
            
            return normalizer
        
    def normalize_clinical_features(self):
        for sample in self.samples:
            if "age" in sample.clinical.keys():
                sample.clinical["age"] /=  100.
            if "dose" in sample.clinical.keys():
                sample.clinical["dose"] /= 70.

    def get_max_clinical_values(self):
        max_values = {}
        for sample in self.samples:
            if not(sample.clinical is None):
                for k, v in sample.clinical.items():
                    if k in CATEGORICAL_CLINICAL_VARIABLES:
                        if k not in max_values.keys() or v > max_values[k]:
                            max_values[k] = v
        return max_values
    
    def one_hot_encode(self, value, max_value):
        if max_value > 1:
            one_hot = np.zeros(int(max_value) + 1)
            one_hot[int(value)] = 1
            return one_hot
        else:
            return value
    
    def one_hot_encode_clinical_features(self, max_clinical_values):
        for sample in self.samples:
            sample.clinical = {k: self.one_hot_encode(sample.clinical[k], max_clinical_values[k]) if k in CATEGORICAL_CLINICAL_VARIABLES else sample.clinical[k] for k in sample.clinical.keys()}
    
    def convert_to_tensor(self):
        for sample in self.samples:
            if not(sample.imaging is None):
                sample.imaging = {k: torch.tensor(v, dtype=torch.float32) for k, v in sample.imaging.items()}
            if not(sample.clinical is None):
                # concatenate clinical features by ordered key to ensure same order across samples and cohorts !!
                sample.clinical = torch.tensor(list(collapse([sample.clinical[k] for k in sorted(sample.clinical.keys())])), dtype=torch.float32)
        
    def undersampling(self):
        labels = [s.label for s in self.samples]
        label_counts = {i: labels.count(i) for i in set(labels)}
        n = min(label_counts.values())
        candidates = {i: [s for s in self.samples if s.label == i] for i in label_counts.keys()}
        selected_samples = {i: random.sample(candidates[i], n) for i in candidates.keys()}
        samples = list(chain.from_iterable(selected_samples.values()))
        return Data(base_path=self.base_path, cohort=self.cohort, samples=samples)

    def stack_to_batch(self):
        """
        Stack samples features in batch tensors for training
        This function requires that features are already converted to tensors see convert_to_tensor() function
        """
        # initialize empty batch dictionary
        batch = {}
        
        # stack imaging features
        if not(self.samples[0].imaging is None):
            batch.update({"image": {k: [] for k in self.samples[0].imaging.keys()}})
            for sample in self.samples:
                for k in sample.imaging.keys():
                    batch["image"][k].append(sample.imaging[k])
            batch["image"] = {k: torch.stack(v, dim=0) for k,v in batch["image"].items()}

        # stack clinical features
        if not(self.samples[0].clinical is None):
            batch["clinical"] = torch.stack([sample.clinical for sample in self.samples], dim=0)
        
        return batch

    def split_loc(self):
        loc_samples = {}
        for sample in self.samples:
            loc = sample.clinical_copy["localisation"]
            if loc not in loc_samples.keys():
                loc_samples[loc] = []
            loc_samples[loc].append(sample)
        
        loc_samples = {f"{self.cohort} ({cohorts_map['radcure'](None).localisation_mapping[k]})": v for k,v in loc_samples.items()}
        return loc_samples.items()


class DataLoader:
    def __init__(self, data, split, class_weights=False):
        """
        Docstring for __init__
        
        data (Data) : internal dataset to use for pre-training and training
        split (str) : split of dataloader
        class_weights (bool) : wether to return class weights for loss function during iteration over data
        """
        self.data = data
        self.split = split
        self.class_weights = class_weights

    def __len__(self):
        return len(self.data)

    def load(self):
        self.data.load(self.split)

    def preprocess(self, exp_params, task):
        """
        compute patients label according to task definition and RFS data
        remove patients with missing data or invalid label
        prepare data for training according to experiment parameters and task
        keep only modalities and features type defined in exp_params
        """
        
        # compute labels according to task definition
        self.data.compute_labels(task)
        
        # filter patients according to inclusion criteria on clinical variables
        # for RADCURE include other localisation in criteria to be able to evaluate model performance on different localisations
        if self.data.cohort == "radcure" and self.split == "test":
            self.data.apply_inclusion_criteria(criteria={"localisation": [0,1,2]})

        # one-hot encode and categorical and filter clinical variables
        # if RADCURE dataset, keep track of patients split for later use in split_loader()
        if self.data.cohort == "radcure":
            self.radcure_challenge = {s.id_: s.clinical["RADCURE-challenge"] for s in self.data.samples}
        self.data.filter_clinical_features(exp_params["clinical"])

        # select imaging features defined in exp_params  
        self.data.filter_imaging_features(exp_params["extractors"])

        # select modality(ies)
        if exp_params["modality"] == "image":   # remove clinical features
            for sample in self.data.samples:
                sample.clinical = None
        elif exp_params["modality"] == "clinical": # remove imaging features
            for sample in self.data.samples:
                sample.imaging = None
        else:
            pass
        
        # filter samples with missing data for selected modality or label
        # preserve internal patients with missing label but available RFS time for Cox model pre-training
        if self.split == "training":
            self.cox_samples = self.data.filter_patients(exp_params)
        else:
            self.data.filter_patients(exp_params)
    
    def compute_class_weights(self, t=0.01):
        labels = [s.label for s in self.data.samples if not(s.label is None)]
        freq = {i: labels.count(i) for i in set(labels)}
        cw = torch.zeros(len(freq), dtype=torch.float32)
        for i in freq.keys():   # populate values with inverse frequence
            i = int(i)
            cw[i] = 1 / freq[i]
        cw = torch.softmax(cw / t, dim=0)   # apply softmax with temperature to increase heterogeneity
        cw = (cw - cw.mean()) + 1   # center values around 1
        return cw
    
    def split_loader(self):
        """
        Split data into training and validation according to RADCURE challenge definition
        """
        training = []
        validation = []
        for sample in self.data.samples:
            if self.radcure_challenge[sample.id_] == "training":
                training.append(sample)
            else:
                validation.append(sample)
        
        training_loader = DataLoader(Data(self.data.base_path, self.data.cohort, training), "training", class_weights=self.class_weights)
        validation_loader = DataLoader(Data(self.data.base_path, self.data.cohort, validation), "validation")
        return training_loader, validation_loader
    
    def get_features_dimension(self):
        return self.data.get_features_dimension()
    
    def normalize_image_features(self, normalizer=None):
        if isinstance(normalizer, Normalizer):
            self.data.normalize_image_features(normalizer)
        else:
            normalizer = self.data.normalize_image_features(normalizer)
            return normalizer

    def normalize_clinical_features(self):
        self.data.normalize_clinical_features()
    
    def convert_to_tensor(self):
        self.data.convert_to_tensor()
    
    def undersampling(self):
        return DataLoader(data=self.data.undersampling(), split=self.split, class_weights=self.class_weights)
    
    def get_labels(self):
        return [s.label for s in self.data.samples]

    def get_max_clinical_values(self):
        return self.data.get_max_clinical_values()
    
    def one_hot_encode_clinical_features(self, max_clinical_values):
        self.data.one_hot_encode_clinical_features(max_clinical_values)

    def batch_iterator(self, batch_size, sample_weight=False, skip_singleton_batch=True):
        # shuffle samples before batching
        random.shuffle(self.data.samples)

        # batch iteration
        for batch in chunked(self.data.samples, batch_size):
            if skip_singleton_batch and len(batch) < 2:
                return StopIteration   # stop if batch size is 1 (BatchNorm error)
            
            x = Data(base_path=self.data.base_path, cohort=self.data.cohort, samples=batch).stack_to_batch()
            y = torch.tensor([s.label for s in batch])

            if sample_weight:
                cw = self.cw[y.to(dtype=torch.long)]
            else:
                cw = None

            yield x, y, cw
        return StopIteration

    def split_loc(self):
        if self.data.cohort != "radcure" or self.split != "test":
            raise ValueError("'split_loc()' function is only implemented for RADCURE test split")
        
        return [DataLoader(Data(self.data.base_path, cohort, samples), self.split) for cohort, samples in self.data.split_loc()]

class CoxProtoNetLoader:
    def __init__(self, data):
        self.data = data
    
    def prepare_data(self, task):
        match task:
            case "rfs_2":
                T = 2
            case "rfs_5":
                T = 5
            case _:
                raise ValueError(f"task {task} not valid. Valid ones are [rfs_2, rfs_5]")
        
        self.positives = [s for s in self.data.samples if not(s.rfs is None) and not(s.rfs["T"] is None) and s.rfs["T"] >= T*365]
        self.negatives = [s for s in self.data.samples if not(s.rfs is None) and not(s.rfs["T"] is None) and s.rfs["T"] < T*365 and s.rfs["delta"] == 1]

        if len(self.positives) == 0 or len(self.negatives) == 0:
            raise ValueError(f"No positive or negative sample for Cox pre-training with {task} task. Consider changing task or strategy.")

    def get_random_batch(self, batch_size):
        # take random samples and stack samples in batch
        n = min(len(self.positives), len(self.negatives), batch_size)
        pos = Data(base_path=self.data.base_path, cohort=self.data.cohort, samples=random.sample(self.positives, n)).stack_to_batch()
        neg = Data(base_path=self.data.base_path, cohort=self.data.cohort, samples=random.sample(self.negatives, n)).stack_to_batch()
        return pos, neg


class ProtoNetLoader:
    def __init__(self, data):
        self.data = data
    
    def prepare_data(self, task):
        match task:
            case "rfs_2":
                T = 2
            case "rfs_5":
                T = 5
            case _:
                raise ValueError(f"task {task} not valid. Valid ones are [rfs_2, rfs_5]")
        
        self.positives = [s for s in self.data.samples if not(s.rfs is None) and not(s.rfs["T"] is None) and s.rfs["T"] >= T*365]
        self.negatives = [s for s in self.data.samples if not(s.rfs is None) and not(s.rfs["T"] is None) and s.rfs["T"] < T*365 and s.rfs["delta"] == 1]

        if len(self.positives) == 0 or len(self.negatives) == 0:
            raise ValueError(f"No positive or negative sample for Cox pre-training with {task} task. Consider changing task.")
        
    def get_random_batch(self, batch_size):
        def average_dict(d):
            if isinstance(d, dict):
                return {k: average_dict(v) for k, v in d.items()}
            else:
                return torch.mean(d, dim=0, keepdim=True)
        
        n = min(len(self.positives), len(self.negatives), batch_size) // 2
        
        # randomly shuffle data
        random.shuffle(self.positives)
        random.shuffle(self.negatives)

        # queries
        pos_queries = Data(base_path=self.data.base_path, cohort=self.data.cohort, samples=self.positives[:n]).stack_to_batch()
        neg_queries = Data(base_path=self.data.base_path, cohort=self.data.cohort, samples=self.negatives[:n]).stack_to_batch()

        # prototypes
        pos_proto = Data(base_path=self.data.base_path, cohort=self.data.cohort, samples=self.positives[n:batch_size]).stack_to_batch()
        pos_proto = average_dict(pos_proto)
        neg_proto = Data(base_path=self.data.base_path, cohort=self.data.cohort, samples=self.negatives[n:batch_size]).stack_to_batch()
        neg_proto = average_dict(neg_proto)

        return (pos_queries, pos_proto), (neg_queries, neg_proto)
