import os
from PIL import Image
import numpy as np
import copy
import random
import pandas as pd


IMAGES = {
    "objects": [
       ('antiquemug', 200, 0.02),
       ('antiquespoon', 180, 0.02),
       ('avocado', 200, 0.02),
       ('tomato', 180, 0.02),
       ('baseball', 150, 0.02),
       ('baseballbat', 230, 0.02),
       ('book', 200, 0.02),
       ('flashlight', 200, 0.02),
       ('button', 130, 0.02),
       ('thread', 200, 0.02),
       ('flower', 150, 0.02),
       ('bagel', 150, 0.02),
       ('eraser', 180, 0.02),
       ('pencil', 200, 0.02),
       ('fork', 200, 0.05),
       ('knife', 200, 0.05),
       ('greenbottle', 200, 0.05),
       ('corkscrew', 150, 0.05),
       ('lemon', 180, 0.02),
       ('apple', 160, 0.02),
       ('orange', 160, 0.02),
       ('pear', 200, 0.02),
       ('razor', 200, 0.02),
       ('sharpener', 200, 0.02),
       ('shoe', 200, 0.02),
       ('socks', 200, 0.02),
       ('stapler', 180, 0.03),
       ('magnifyingglass', 200, 0.03),
       ('table', 200, 0.03),
       ('chair', 200, 0.03),
       ('tape', 180, 0.02),
       ('scissors', 230, 0.02),
    ],
    "objects_holdout": [
       ('barrel', 170, 0.02),
       ('brush', 180, 0.02),
       ('burger', 180, 0.02),
       ('calculator', 180, 0.02),
       ('compass', 180, 0.02),
       ('dart', 180, 0.02),
       ('fan', 200, 0.02),
       ('leaf', 200, 0.02),
       ('pinecone', 170, 0.02),
       ('shrimp', 190, 0.02),
    ]
}

long_objects = ['antiquespoon', 'baseballbat', 'banana', 'flashlight', 'fork', 'knife', 'magnifyingglass', 'pencil', 'razor',
                'scissors', 'socks']


def manual_correct_obj_name(obj_name):
    if obj_name.startswith('antique'):
        return obj_name.replace('antique', '')
    elif obj_name == 'coffee':
        return 'mug'
    elif obj_name == 'greenbottle':
        return 'bottle'
    elif obj_name == 'baseballbat':
        return 'bat'
    return obj_name


def paste_image_on_canvas(canvas, img, pos, occlusion_tolerance=None):
    if canvas is None:
        return None
    # Calculate new positions based on the image dimensions to center the images at the specified coordinates
    new_x = pos[0] - img.width // 2
    new_y = pos[1] - img.height // 2

    before_pixel = count_foreground_pixel(canvas) + count_foreground_pixel(img)

    # Paste the image onto the canvas at the calculated position
    canvas.paste(img, (new_x, new_y), img)

    after_pixel = count_foreground_pixel(canvas)
    # Determine the prefix based on occlusion criteria
    pixel_difference = abs(after_pixel - before_pixel)
    if occlusion_tolerance is not None and (pixel_difference / after_pixel) > occlusion_tolerance:
        del canvas
        return None

    return canvas


def convert_transparent_pixels(canvas):
    canvas_np = np.array(canvas)
    alpha_channel = canvas_np[:, :, 3]
    mask = alpha_channel < 200

    canvas_np[mask] = [255, 255, 255, 255]  # White pixel with full alpha

    return Image.fromarray(canvas_np, 'RGBA')


def is_too_close(p1, p2, threshold):
    return np.linalg.norm(np.array(p1) - np.array(p2)) < threshold


def generate_no_relation_pos(x_range, y_range, prev_obj_pos, max_trial_num=1000, min_no_rel_distance=300):
    too_close = True
    count = 0
    while too_close and count < max_trial_num:
        count += 1
        obj_pos = (np.random.randint(x_range[0], x_range[1]), np.random.randint(y_range[0], y_range[1]))

        too_close = False
        for pos in prev_obj_pos:
            if is_too_close(obj_pos, pos, min_no_rel_distance):
                too_close = True
                continue
    if too_close:
        return None
    return obj_pos


def create_image_with_relation(objs, pos, canvas_size, num_extra_choice=1, occlusion_tolerance=0.05, padding=50, min_no_rel_distance=200):
    assert len(objs) == num_extra_choice + 5
    obj_sizes = [(objs[i].width, objs[i].height) for i in range(len(objs))]
    canvas = Image.new('RGBA', canvas_size, (0, 0, 0, 0))
    canvas = paste_image_on_canvas(canvas, objs[0], pos, occlusion_tolerance)

    obj1_width = objs[0].width
    obj1_height = objs[0].height

    # second object (top)
    obj2_pos = (pos[0], pos[1] - obj1_height // 2 - objs[1].height // 2)
    canvas = paste_image_on_canvas(canvas, objs[1], obj2_pos, occlusion_tolerance)
    # third object (down)
    obj3_pos = (pos[0], pos[1] + obj1_height // 2 + objs[2].height // 2)
    canvas = paste_image_on_canvas(canvas, objs[2], obj3_pos, occlusion_tolerance)

    # left side next to
    obj4_pos = (pos[0] - obj1_width // 2 - objs[3].width // 2, pos[1])
    canvas = paste_image_on_canvas(canvas, objs[3], obj4_pos, occlusion_tolerance)
    # right side next to
    obj5_pos = (pos[0] + obj1_width // 2 + objs[4].width // 2, pos[1])
    canvas = paste_image_on_canvas(canvas, objs[4], obj5_pos, occlusion_tolerance)

    if canvas is None:
        return None, None, None

    x_range = (padding, canvas_size[0] - padding)
    y_range = (padding, canvas_size[1] - padding)
    cur_canvas = None

    # extra objects (no relation)
    count = 0
    while cur_canvas is None and count < 1000:
        count += 1
        cur_canvas = copy.deepcopy(canvas)
        prev_obj_pos = [pos, obj2_pos, obj3_pos, obj4_pos, obj5_pos]
        extra_obj_pos_list = []
        for _ in range(num_extra_choice):
            extra_obj_pos = generate_no_relation_pos(x_range, y_range, prev_obj_pos, min_no_rel_distance)
            if extra_obj_pos is not None:
                prev_obj_pos.append(extra_obj_pos)
                extra_obj_pos_list.append(extra_obj_pos)
            else:
                continue

        for e_i in range(num_extra_choice):
            cur_canvas = paste_image_on_canvas(cur_canvas, objs[5 + e_i], extra_obj_pos_list[e_i], occlusion_tolerance)
            if cur_canvas is None:
                break

    if cur_canvas is None:
        return None, None, None

    return convert_transparent_pixels(cur_canvas), prev_obj_pos, obj_sizes


def read_obj_image(image_path, long_side=300):
    try:
        image = Image.open(image_path + '.png')
    except:
        image = Image.open(image_path + '.jpg')

    # Determine the new size maintaining the aspect ratio
    width, height = image.size
    if width > height:
        new_width = long_side
        new_height = int((height / width) * long_side)
    else:
        new_height = long_side
        new_width = int((width / height) * long_side)

    # Resize the image
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return resized_image, (new_height, new_width)


def count_foreground_pixel(image):
    # Ensure the image is in RGBA mode
    if image.mode != 'RGBA':
        image = image.convert('RGBA')

    # Get the alpha channel
    alpha = image.split()[-1]

    # Convert alpha channel to numpy array
    alpha_array = np.array(alpha)

    # Count the number of non-zero alpha values (foreground pixels)
    return np.sum(alpha_array > 0)


def save_metadata(metadata, objs, objs_pos, objs_sizes, image_name):
    metadata['anchor'].append(objs[0])
    metadata['above_object'].append(objs[1])
    metadata['below_object'].append(objs[2])
    metadata['left_object'].append(objs[3])
    metadata['right_object'].append(objs[4])
    metadata['random_object1'].append(objs[5])

    metadata['anchor_pos'].append(f"{objs_pos[0][0]} {objs_pos[0][1]}")
    metadata['above_object_pos'].append(f"{objs_pos[1][0]} {objs_pos[1][1]}")
    metadata['below_object_pos'].append(f"{objs_pos[2][0]} {objs_pos[2][1]}")
    metadata['left_object_pos'].append(f"{objs_pos[3][0]} {objs_pos[3][1]}")
    metadata['right_object_pos'].append(f"{objs_pos[4][0]} {objs_pos[4][1]}")
    metadata['random_object1_pos'].append(f"{objs_pos[5][0]} {objs_pos[5][1]}")

    metadata['anchor_size'].append(f"{objs_sizes[0][0]} {objs_sizes[0][1]}")
    metadata['above_object_size'].append(f"{objs_sizes[1][0]} {objs_sizes[1][1]}")
    metadata['below_object_size'].append(f"{objs_sizes[2][0]} {objs_sizes[2][1]}")
    metadata['left_object_size'].append(f"{objs_sizes[3][0]} {objs_sizes[3][1]}")
    metadata['right_object_size'].append(f"{objs_sizes[4][0]} {objs_sizes[4][1]}")
    metadata['random_object1_size'].append(f"{objs_sizes[5][0]} {objs_sizes[5][1]}")

    metadata['image_name'].append(os.path.basename(image_name))
    return metadata


def create_stimuli(image_folder, images, canvas_size, output_path, num_trials,
                   num_extra_choice=1, center_range=200, padding=50, jitter=False, min_no_rel_distance=300):
    metadata = {'image_name': [],
                'anchor': [], 'above_object': [], 'below_object': [], 'left_object': [], 'right_object': [], 'random_object1': [],
                'anchor_pos': [], 'above_object_pos': [], 'below_object_pos': [], 'left_object_pos': [], 'right_object_pos': [], 'random_object1_pos': [],
                'anchor_size': [], 'above_object_size': [], 'below_object_size': [], 'left_object_size': [], 'right_object_size': [], 'random_object1_size': [],
                }

    center_x_range = (canvas_size[0] // 2 - center_range, canvas_size[0] // 2 + center_range + 1)
    center_y_range = (canvas_size[1] // 2 - center_range, canvas_size[1] // 2 + center_range + 1)

    count = 0
    failed_images = 0

    while count < num_trials:
        print(f"Generating {count+1}-th trial...")

        selected_images = random.sample(images, num_extra_choice + 5)
        objs = []
        image_names = []
        occlusion_tolerance = 0
        # subject, above object, below object, left object, right object, random_object(s)
        for im_idx in range(len(selected_images)):
            image_name, long_side, tolerance = selected_images[im_idx]
            if jitter:
                long_side = int(np.random.normal(loc=long_side, scale=10))
            obj, _ = read_obj_image(os.path.join(image_folder, image_name), long_side=long_side)
            image_names.append(manual_correct_obj_name(image_name))
            objs.append(obj)
            occlusion_tolerance += tolerance

        pos = (np.random.randint(center_x_range[0], center_x_range[1]),
               np.random.randint(center_y_range[0], center_y_range[1]))

        cur_objs = copy.deepcopy(objs)

        # rotate long objects for above and below relation
        if selected_images[1][0] in long_objects:
            cur_objs[1] = cur_objs[1].rotate(90, expand=True)
        if selected_images[2][0] in long_objects:
            cur_objs[2] = cur_objs[2].rotate(90, expand=True)
        canvas, objs_pos_list, objs_sizes = create_image_with_relation(cur_objs, pos, canvas_size, num_extra_choice,
                                                                     occlusion_tolerance, padding, min_no_rel_distance)

        if canvas is not None:
            count += 1
            save_name = os.path.join(output_path, f'image_{count}.png')
            canvas.save(save_name, 'PNG')
            metadata = save_metadata(metadata, image_names, objs_pos_list, objs_sizes, save_name)
            print(f"Saved image for {count} trials!")
        else:
            failed_images += 1

    print(f"Ignored {failed_images} failed images in total.")

    return count, pd.DataFrame(metadata)


if __name__ == '__main__':
    seed = 1234
    np.random.seed(seed)
    random.seed(seed)

    jitter = True
    min_no_rel_distance = 300
    canvas_size = (800, 800)  # Width, height of the canvas

    splits = ['train', 'val', 'test']
    num_trials_mapping = {'train': 1000, 'eval': 4000, 'test': 1000}
    obj_sets = ['objects', 'objects_holdout']

    for obj_set in obj_sets:
        for split in splits:
            if obj_set == 'objects_holdout' and split != 'test':
                continue

            num_trials = num_trials_mapping[split]

            image_folder = f'data/{obj_set}'
            save_path = f'data/synthetic_spatial_relation_{split}' if obj_set == 'object' else 'data/synthetic_spatial_relation_novel_objects_test'
            os.makedirs(save_path, exist_ok=True)


            images = IMAGES[obj_set]
            num_image, metadata = create_stimuli(image_folder, images, canvas_size, save_path, num_trials, jitter=jitter, min_no_rel_distance=min_no_rel_distance)

            print(f"Generated {num_image} images for {obj_set} split {split} in total.")
            metadata.to_csv(os.path.join(save_path, 'metadata.csv'), index=False)
