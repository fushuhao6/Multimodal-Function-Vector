import torch

def construct_question(obj_pairs, query_obj, model_config, images=None):
    name = model_config['name_or_path'].lower()
    if 'flamingo' in name:
        context_image_questions = [
            f"<image>Q:{obj_pair[0]}. A:{obj_pair[1]}.<|endofchunk|>"
            for image_id, obj_pair in enumerate(obj_pairs)
        ]
        context_images_text = '\n'.join(context_image_questions)

        if len(obj_pairs) > 0:
            context_images_text = context_images_text + '\n'
        else:
            context_images_text = ''

        question = (
            f"{context_images_text}<image>Q:{query_obj}. A:"
        )
    elif any(keyword in name for keyword in ("qwen")):
        assert images is not None
        assert len(images) == len(obj_pairs) + 1
        content = []
        for i, obj_pair in enumerate(obj_pairs):
            content.append({
                "type": "image",
                "image": images[i],
            })
            content.append({
                "type": "text",
                "text": f"Q:{obj_pair[0]}. A:{obj_pair[1]}.\n"
            })

        content.append({
            "type": "image",
            "image": images[-1],
        })
        content.append({
            "type": "text",
            "text": f"Q:{query_obj}. A:"
        })
        question = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Use the following template to answer the question:\nQ:object1. A:object2"}]
            },
            {
                "role": "user",
                "content": content,
            }
        ]
    else:
        raise ValueError(f'Unsupported model config: {model_config["name_or_path"]}')
    return question
