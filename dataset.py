import os
import matplotlib.pyplot as plt
import cv2
import matplotlib.image as mpimg
import numpy as np
import pandas as pd


class FacialKeypointsDataset(object):
    def __init__(self, csv_file, root_dir, transform=None):
        """
                Args:
                    csv_file (string): Path to the csv file with annotations.
                    root_dir (string): Directory with all the images.
                    transform (callable, optional): Optional transform to be applied
                        on a sample.
        """
        self.key_plt_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_plt_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir, self.key_plt_frame.iloc[idx, 0])

        image = mpimg.imread(image_name)

        if image.shape[2] == 4:
            image = image[:, :, 0:3]

        key_pts = self.key_plt_frame.iloc[idx, 1:]
        key_pts = key_pts.astype("float").values.reshape((-1, 2))
        sample = {"image": image, "keypoints": key_pts}

        return sample


class Normalize(object):
    """Convert a color image to grayscale and normalize the color range to [0,1]."""

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        image_copy = np.copy(image)
        key_pts_copy = np.copy(key_pts)

        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_RGB2GRAY)

        image_copy = image_copy / 255

        key_pts_copy = (key_pts_copy - 100) / 50

        return {"image": image_copy, "keypoints": key_pts_copy}


class Rescale(object):
    """Rescale the image in a sample to a given size.

        Args:
            output_size (tuple or int): Desired output size. If tuple, output is
                matched to output_size. If int, smaller of image edges is matched
                to output_size keeping aspect ratio the same.
        """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(image, (new_w, new_h))

        key_pts = key_pts * [new_w / w, new_h / h]

        return {"image": img, "keypoints": key_pts}


class RamdonCorp(object):
    """Crop randomly the image in a sample.

        Args:
            output_size (tuple or int): Desired output size. If int, square crop
                is made.
        """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, key_pts = sample['image'], sample['keypoints']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top:top + new_h, left:left + new_w]
        key_pts = key_pts - [left, top]

        return {"image": image, "keypoints": key_pts}


def show_keypoints(image, key_pts, gt_pts=None):
    plt.imshow(image)
    plt.scatter(key_pts[:, 0], key_pts[:, 1], s=20, marker='.', c='m')
    if gt_pts is not None:
        plt.scatter(gt_pts[:, 0], gt_pts[:, 1], s=20, marker='.', c='g')


def visualize_output(test_images, test_outputs, gt_pts=None, batch_size=10):
    for i in range(batch_size):
        # plt.figure(figsize=(20, 10))
        ax = plt.subplot(1, batch_size, i+1)

        image = test_images[i]
        predicted_key_pts = test_outputs[i]
        predicted_key_pts = predicted_key_pts*50+100
        predicted_key_pts = predicted_key_pts.reshape((-1, 2)).astype('int')

        if gt_pts is not None:
            ground_truth_pts = gt_pts[i]
            ground_truth_pts = ground_truth_pts*50 + 100
            ground_truth_pts = ground_truth_pts.reshape((-1, 2)).astype('int')
            show_keypoints(np.squeeze(image), predicted_key_pts, ground_truth_pts)
        else:
            show_keypoints(np.squeeze(image), predicted_key_pts)
        plt.axis('off')
    plt.show()