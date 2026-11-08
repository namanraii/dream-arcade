import numpy as np

from dreamarcade.eval.metrics import mae, precision_recall, psnr, ssim
from dreamarcade.train.ppo_dream import human_normalized


def test_identical_frames_score_perfectly():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[10:20, 10:14] = 255
    assert psnr(image, image) > 40
    assert ssim(image, image) > 0.99
    assert mae(image, image) == 0


def test_precision_recall_and_hns():
    target = np.array([0, 1, 1, 2])
    pred = np.array([0, 1, 0, 2])
    report = precision_recall(pred, target)
    assert report["2"]["precision"] == 1
    assert report["2"]["recall"] == 1
    assert abs(human_normalized(14.6) - 1) < 1e-9
    assert abs(human_normalized(-20.7)) < 1e-9
