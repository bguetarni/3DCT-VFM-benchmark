from monai.transforms import MapTransform, CropForegroundd
import numpy as np

class BboxCropd(MapTransform):
    def __call__(self, data):
        if data["bbox"] is None:
            return CropForegroundd(keys=["image"], source_key="image")(data)
        else:
            data["bbox"] = self.fit_bbox_size(data["bbox"])
            xmin, ymin, zmin = data["bbox"][0]
            xmax, ymax, zmax = data["bbox"][1]
            data["image"] = data["image"][xmin:xmax, ymin:ymax, zmin:zmax]
            return data

    def fit_bbox_size(self, bbox, size=(100,100,30)):
        # increase bbox to fit to specified size
        # bbox must be specified as ((xmin,ymin,zmin), (xmax,ymax,zmax))
        bbox = np.asarray(bbox)
        assert all(bbox[1] >= bbox[0]), "Second argument of bbox must be greater than first"
        for i in range(len(bbox[0])):
            if bbox[1][i] - bbox[0][i] < size[i]:
                bbox[0][i] = min(0, bbox[0][i] - size[i]//2) # avoid negative values
                bbox[1][i] += size[i]//2
        else:
            bbox = bbox
        return bbox