"""Box geometry, in original image pixels."""


def clip_box(box, image_size):
    w, h = image_size
    return (max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3]))


def pad_box(box, frac, image_size):
    px, py = (box[2] - box[0]) * frac, (box[3] - box[1]) * frac
    return clip_box((box[0] - px, box[1] - py, box[2] + px, box[3] + py), image_size)


def area_frac(box, image_size):
    w, h = image_size
    return (box[2] - box[0]) * (box[3] - box[1]) / (w * h)


def crop_ok(image, box, min_side=8):
    crop = image.crop(tuple(round(v) for v in box))
    return crop if min(crop.size) >= min_side else None
