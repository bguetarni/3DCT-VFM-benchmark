from monai.transforms import MapTransform, CropForegroundd, CenterSpatialCropd
import numpy as np

class BboxCropd(MapTransform):
    def __init__(self, keys, roi_size=(100,100,50), allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.roi_size = roi_size

    def __call__(self, data):
        if data["bbox"] is None:
            data = CropForegroundd(keys=["image"], source_key="image", allow_smaller=True)(data)
        else:
            data["bbox"] = self.fit_bbox_size(data["bbox"])
            xmin, ymin, zmin = data["bbox"][0]
            xmax, ymax, zmax = data["bbox"][1]
            # assume image is channel-first
            data["image"] = data["image"][:, xmin:xmax, ymin:ymax, zmin:zmax]
        
        return CenterSpatialCropd(keys=["image"], roi_size=self.roi_size)(data)

    def fit_bbox_size(self, bbox, size=None):
        # increase bbox to fit to specified size
        # bbox must be specified as ((xmin,ymin,zmin), (xmax,ymax,zmax))

        if size is None:
            size = self.roi_size
        
        bbox = np.asarray(bbox)
        assert all(bbox[1] >= bbox[0]), "Second argument of bbox must be greater than first"
        for i in range(len(bbox[0])):
            if bbox[1][i] - bbox[0][i] < size[i]:
                bbox[0][i] = max(0, bbox[0][i] - size[i]//2) # avoid negative values
                delta = np.abs(min(0, bbox[0][i] - size[i]//2))
                bbox[1][i] += size[i]//2 + delta   # compensate if we had to shift the min value
        return bbox
