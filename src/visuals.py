from matplotlib import pyplot as plt
from copy import deepcopy
import numpy as np
import cv2

from pdb import set_trace as pause


class SegmentationVisuals:

    def __init__(self):

        self.n_axes = 3
        self.fig, self.axes = plt.subplots(nrows = 1, ncols = self.n_axes)
        self.ax_titles = ('Image', 'Segmentation Mask', 'Combined')

    def combine_img_mask(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        '''
            Description:
                Generates an image based on `img` where the areas of `mask` that correspond to a class are highlighted. Supports only one class.

            Args:
                img. Shape (H, W, C).
                mask. Shape (H, W).

            Returns:
                combined. Shape (H, W, C).
        '''

        ## Can take values in the interval (0, 1)
        alpha = 0.5
        colors = [None, [255, 0, 0]]
        color_ClassIdcs = {}
        class_idcs = np.unique(mask).tolist()
        class_idcs.sort()
        color_ClassIdcs = {i: colors[i] for i in class_idcs}

        ## Each iteration computes a one vs all mask; and applies weighted sum on the input image until all disjoint mask segmentations form the entire area of the image.
        combined = deepcopy((1 - alpha) * img).astype(np.uint8)
        for class_ in class_idcs:
            binary_mask = mask == class_
            binary_mask_rgb = np.stack(3 * [binary_mask], axis = -1).astype(np.uint8)
            if class_ == 0:
                binary_mask_rgb = binary_mask_rgb * img
            else:
                for channel in range(3):
                    binary_mask_rgb[..., channel] = binary_mask_rgb[..., channel] * colors[class_][channel]
            combined += (alpha * binary_mask_rgb).astype(np.uint8)

        return combined

    def build_plt(self, img: np.ndarray, mask: np.ndarray, fig_title: str):
        '''
            Args:
                img. Shape (H, W, C).
                mask. Shape (H, W).
        '''

        self.fig_title = fig_title
        combined = self.combine_img_mask(img = img, mask = mask)

        for ax_idx, (ax, img) in enumerate(zip(self.axes, (img, mask, combined))):
            ax.imshow(X = img)
            ax.set_title(self.ax_titles[ax_idx])
            ax.axis('off')

        self.fig.suptitle(self.fig_title + '\nImage Resolution: (%d, %d)'%(img.shape[0], img.shape[1]))
        plt.show()

    def store_fig(self, fp):
        plt.savefig(fp, dpi=1200)
        plt.close()
        self.fig.clear()
