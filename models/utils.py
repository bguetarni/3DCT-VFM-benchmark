from monai.transforms import MapTransform
import numpy as np

class BboxCropd(MapTransform):
    def __call__(self, data):
        if data["bbox"] is None:
            data = data
        else:
            data["bbox"] = self.fit_bbox_size(data["bbox"])
            xmin, ymin, zmin = data["bbox"][0]
            xmax, ymax, zmax = data["bbox"][1]
            data["image"] = data["image"][xmin:xmax, ymin:ymax, zmin:zmax]
        return data

    def fit_bbox_size(self, bbox, size=(20, 50, 50)):
        # increase bbox to fit to specified size
        # bbox must be specified as ((xmin,ymin,zmin), (xmax,ymax,zmax))

        bbox = np.array(bbox[0]), np.array(bbox[1])
        assert all(bbox[1] >= bbox[0]), "Second argument of bbox must be greater than first"
        if any(bbox[1] - bbox[0] < np.array(size)):
            for i in range(len(bbox[0])):
                if bbox[0][i] - size[i]/2 < 0:
                    bbox[1][i] += size[i]
                else:
                    bbox[0][i] -= size[i]//2
                    bbox[1][i] += size[i]//2
        else:
            bbox = bbox
        return bbox