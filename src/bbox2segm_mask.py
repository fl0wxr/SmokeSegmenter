import numpy as np
from segment_anything import SamPredictor, sam_model_registry

from pdb import set_trace as pause


def bbox2segm_mask(img, bboxes, sam_fp):

    sam = sam_model_registry['default'](checkpoint = sam_fp)
    labels = [bbox[0] for bbox in bboxes]
    bboxes = np.array([bbox[1:] for bbox in bboxes])

    predictor = SamPredictor(sam)
    predictor.set_image(img)
    masks, _, _ = predictor.predict\
    (
        # point_coords = None,
        # point_labels = None,
        # box = bboxes,
        # multimask_output = True,
    )

    print(np.sum(masks))
    pause()

    return masks[2, ...]

