import numpy as np
from segment_anything import SamPredictor, sam_model_registry


def bbox2segm_mask(img: np.ndarray, bboxes: list[list], sam_fp: str, DEVICE) -> np.ndarray:
    '''
        Description:
            Applies segmentation based solely on bounding boxes. There is exactly one class (the smoke class) with its mask index set to 1.

        Args:
            img. Shape (H, W, C). The input image.
            bboxes. Outter length is n_bboxes, inner length is 5 where the first element is the class index, and the following 4 are the bounding box coordinates in the format xmin.
            sam_fp. File path of the corresponding SAM model.

        Returns:
            mask. Shape (H, W). Segmentation mask.
    '''

    if len(bboxes) == 0: return np.zeros(img.shape[:-1], dtype = np.uint8)

    ## Change to 'vit_h' instead of 'vit_l' if you use the default model
    sam = sam_model_registry['vit_l'](checkpoint = sam_fp)
    sam.to(device = DEVICE)
    predictor = SamPredictor(sam)

    labels = [bbox[0] for bbox in bboxes]

    ## Columns must be in xyxy format with shape (n_bboxes, 4)
    bboxes_coordinates = np.array([bbox[1:] for bbox in bboxes])[:, [0, 2, 1, 3]]

    predictor.set_image(img)

    masks = []
    for bbox_coordinates in bboxes_coordinates:

        ## Shape (1, H, W). Segmentation masks. The second dimension's slices are the same bounding boxes mask with different prediction confidence scores (decreasing order).
        ## (!) predictor.predict_torch doesn't seem to work nearly as well as predictor.predict
        mask, _, _ = predictor.predict\
        (
            point_coords = None,
            point_labels = None,
            box = bbox_coordinates,
            multimask_output = False,
        )

        masks.append(mask[0])

    mask = np.any(masks, axis = 0)

    ## Equivalent to the `any` approach
    # mask = np.zeros(masks[0].shape)
    # for mask_ in masks:
    #     mask = np.logical_or(mask, mask_)

    mask = mask.astype(np.uint8)

    return mask