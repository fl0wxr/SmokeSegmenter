from matplotlib import pyplot as plt
from copy import deepcopy
import numpy as np
import cv2

from pdb import set_trace as pause


def masked_image(img: np.ndarray, bboxes: list[list], classes: list[str]) -> list[np.ndarray]:
    '''
        Args:
            img. Shape (H, W, C). Image.
            bboxes. Length equal to the instances number of bounding boxes. Coordinate system matches that of the digital image `img`.
                bboxes[i][0] -> class index
                bboxes[i][1] -> x left
                bboxes[i][2] -> x right
                bboxes[i][3] -> y up
                bboxes[i][4] -> y down

        Returns:
            masked_img. Image where the contours of the bounding boxes are placed on top of it. Each bounding box has text on it.
    '''

    masked_img = deepcopy(img)

    thickness = 2

    colors = [(0, 206, 209), (68, 252, 4), (255, 0, 0)] ## (228, 205, 0)

    for bbox in bboxes:

        color = colors[bbox[0]]
        bbox_text = '%s'%(classes[bbox[0]])

        ## Docs for cv2.rectangle -> https://docs.opencv.org/3.1.0/d6/d6e/group__imgproc__draw.html#ga07d2f74cadcf8e305e810ce8eed13bc9
        ## Bounding box
        cv2.rectangle(img = masked_img, pt1 = (bbox[1], bbox[3]), pt2 = (bbox[2], bbox[4]), color = color, thickness = thickness)

        (text_width, text_height), _ = cv2.getTextSize(bbox_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

        pt1 = (bbox[1] - 1, int(bbox[3] - text_height * 1.4) - 1)
        pt2 = (bbox[1] + text_width + 2, bbox[3])
        org = (bbox[1] + 2, bbox[3] - 5)
        if pt1[1] < 0:
            pt1 = (bbox[1] - 1, 2 + int(bbox[3] + text_height * 1.4))
            pt2 = (bbox[1] + text_width + 2, bbox[3])
            org = (bbox[1] + 2, bbox[3] + 3 + text_height)

        ## Text box
        cv2.rectangle(img = masked_img, pt1 = pt1, pt2 = pt2, color = color, thickness = -1)

        ## Text
        cv2.putText(img = masked_img, text = bbox_text, org = org, fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 0.8, color = (0, 0, 0), thickness = 2)

    return masked_img

class SegmVisuals:

    def __init__(self, classes):

        self.n_axes = 2
        self.classes = classes
        self.n_classes = len(self.classes)
        self.fig, self.axes = plt.subplots(nrows = 1, ncols = self.n_axes, figsize = (12, 6.3))
        self.ax_titles = ('Image', 'Combined') #, 'Segmentation Mask'

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
        colors = [None, (0, 206, 209)] #(228, 205, 0)
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

    def build_plt(self, img: np.ndarray, mask: np.ndarray, bboxes: None or list[list], fig_title: str):
        '''
            Args:
                img. Shape (H, W, C). Image.
                mask. Shape (H, W). Mask.
                bboxes. NoneType or Length equal to the instances number of bounding boxes. Coordinate system matches that of the digital image `img`.
                    bboxes[i][0] -> class index
                    bboxes[i][1] -> x left
                    bboxes[i][2] -> x right
                    bboxes[i][3] -> y up
                    bboxes[i][4] -> y down
                fig_title. Title of figure.
        '''

        self.fig_title = fig_title
        combined = self.combine_img_mask(img = img, mask = mask)
        if bboxes != None: combined = masked_image(img = combined, bboxes = bboxes, classes = self.classes)

        for ax_idx, (ax, img_) in enumerate(zip(self.axes, (img, combined))):
            ax.imshow(X = img_)
            ax.set_title(self.ax_titles[ax_idx])
            ax.axis('off')

        self.fig.suptitle(self.fig_title + '\nImage Resolution: (%d, %d)'%(img.shape[0], img.shape[1]))

    def display(self):

        plt.show()

    def store_fig(self, fp = '../produced_plots/fig0.jpg'):

        plt.savefig(fp, dpi = 1200)
        plt.close()
        self.fig.clear()

class DetVisuals:

    def __init__(self, classes):

        self.n_axes = 2
        self.classes = classes
        self.n_classes = len(self.classes)
        self.fig, self.axes = plt.subplots(nrows = 1, ncols = self.n_axes, figsize = (12, 6.3))
        self.ax_titles = ('Image', 'Bounding Boxes')

    def build_plt(self, img: np.ndarray, bboxes: list[list], fig_title: str, confidence_scores: list[float] = None):
        '''
            Args:
                img. Shape (H, W, C). Image.
                bboxes. Length equal to the instances number of bounding boxes. Coordinate system matches that of the digital image `img`.
                    norm_vertex_bboxes[i][0] -> class index
                    norm_vertex_bboxes[i][1] -> x left
                    norm_vertex_bboxes[i][2] -> x right
                    norm_vertex_bboxes[i][3] -> y up
                    norm_vertex_bboxes[i][4] -> y down
                fig_title. Title of figure.
                confidence_scores. Length equal to the instances number of bounding boxes. Contains confidence score for each bounding box.
        '''

        if bboxes != []:

            self.confidence_scores = confidence_scores
            if self.confidence_scores == None:
                self.confidence_scores = len(bboxes) * [None]

            self.fig_title = fig_title
            masked_img = masked_image(img = img, bboxes = bboxes, classes = self.classes)

            for ax_idx, (ax, img_) in enumerate(zip(self.axes, (img, masked_img))):
                ax.imshow(X = img_)
                ax.set_title(self.ax_titles[ax_idx])
                ax.axis('off')

        else:

            for ax_idx, (ax, img_) in enumerate(zip(self.axes, (img, img))):
                ax.imshow(X = img_)
                ax.set_title(self.ax_titles[ax_idx])
                ax.axis('off')

        self.fig.patch.set_facecolor('gray')

        self.fig.suptitle(self.fig_title + '\nImage Resolution: (%d, %d)'%(img.shape[0], img.shape[1]))

    def display(self):

        plt.show()

    def store_fig(self, fp = '../produced_plots/fig0.jpg'):

        plt.savefig(fp, dpi = 1200)
        plt.close()
        self.fig.clear()