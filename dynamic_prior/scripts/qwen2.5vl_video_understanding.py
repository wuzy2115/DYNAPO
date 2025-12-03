import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

import os
import math
import hashlib
import requests
from IPython.display import Markdown, display
import numpy as np
from PIL import Image
import decord
from decord import VideoReader, cpu
from openai import OpenAI

import json
import markdown
from bs4 import BeautifulSoup
from datetime import datetime


def download_video(url, dest_path):
    response = requests.get(url, stream=True)
    with open(dest_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8096):
            f.write(chunk)
    print(f"Video downloaded to {dest_path}")


def get_video_frames(video_path, num_frames=128, cache_dir='.cache'):
    os.makedirs(cache_dir, exist_ok=True)

    video_hash = hashlib.md5(video_path.encode('utf-8')).hexdigest()
    if video_path.startswith('http://') or video_path.startswith('https://'):
        video_file_path = os.path.join(cache_dir, f'{video_hash}.mp4')
        if not os.path.exists(video_file_path):
            download_video(video_path, video_file_path)
    else:
        video_file_path = video_path

    frames_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_frames.npy')
    timestamps_cache_file = os.path.join(cache_dir, f'{video_hash}_{num_frames}_timestamps.npy')

    if os.path.exists(frames_cache_file) and os.path.exists(timestamps_cache_file):
        frames = np.load(frames_cache_file)
        timestamps = np.load(timestamps_cache_file)
        return video_file_path, frames, timestamps

    vr = VideoReader(video_file_path, ctx=cpu(0))
    total_frames = len(vr)

    indices = np.linspace(0, total_frames - 1, num=num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()
    timestamps = np.array([vr.get_frame_timestamp(idx) for idx in indices])

    np.save(frames_cache_file, frames)
    np.save(timestamps_cache_file, timestamps)

    return video_file_path, frames, timestamps


def create_image_grid(images, num_columns=8):
    pil_images = [Image.fromarray(image) for image in images]
    num_rows = math.ceil(len(images) / num_columns)

    img_width, img_height = pil_images[0].size
    grid_width = num_columns * img_width
    grid_height = num_rows * img_height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    for idx, image in enumerate(pil_images):
        row_idx = idx // num_columns
        col_idx = idx % num_columns
        position = (col_idx * img_width, row_idx * img_height)
        grid_image.paste(image, position)

    return grid_image


def inference(video_path, prompt, max_new_tokens=2048, total_pixels=20480 * 28 * 28, min_pixels=16 * 28 * 28):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"video": video_path, "total_pixels": total_pixels, "min_pixels": min_pixels},
            ]
        },
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info([messages], return_video_kwargs=True)
    fps_inputs = video_kwargs['fps']
    print("video input:", video_inputs[0].shape)
    num_frames, _, resized_height, resized_width = video_inputs[0].shape
    print("num of video tokens:", int(num_frames / 2 * resized_height / 28 * resized_width / 28))
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, fps=fps_inputs, padding=True, return_tensors="pt")
    inputs = inputs.to('cuda')

    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
    output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return output_text[0]

def inference_with_api(
    video_path,
    prompt,
    sys_prompt = "You are a helpful assistant.",
    model_id = "qwen-vl-max-latest",
):
    client = OpenAI(
        api_key = os.getenv('DASHSCOPE_API_KEY'),
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )    
    messages = [
        {
            "role": "system",
            "content": [{"type":"text","text": sys_prompt}]
        },
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": video_path}},
                {"type": "text", "text": prompt},
            ]
        }
    ]
    completion = client.chat.completions.create(
        model = model_id,
        messages = messages,
    )
    print(completion)
    return completion.choices[0].message.content


def parse_json(response):
    html = markdown.markdown(response, extensions=['fenced_code'])
    soup = BeautifulSoup(html, 'html.parser')
    json_text = soup.find('code').text

    data = json.loads(json_text)
    return data


def time_to_seconds(time_str):
    time_obj = datetime.strptime(time_str, '%M:%S.%f')
    total_seconds = time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1_000_000
    return total_seconds



if __name__ == "__main__":
    model_path = "Qwen/Qwen2.5-VL-32B-Instruct"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    # video_url = "https://ofasys-multimodal-wlcb-3.oss-cn-wulanchabu.aliyuncs.com/sibo.ssb/datasets/cookbook/ead2e3f0e7f836c9ec51236befdaf2d843ac13a6.mp4"
    # prompt = "Localize a series of activity events in the video, output the start and end timestamp for each event, and describe each event with sentences. Provide the result in json format with 'mm:ss.ff' format for time depiction."

    video_url = "/data/zhuoyuan/dynpose-100k/lightspeed/video/0380_GODZROCK.mp4"
    # prompt = "Analysis the wholo video to find all moving objects in the video, output the start and end timestamp for each moving objects. Provide the result in json format with 'mm:ss.ff' format for time depiction."
    prompt = "Describe the video in detail. List all moving objects in the video, and describe the action of each object. Some objects might locate in the edge of the video, and some might only appear in a short period of time, you should notice these objects as well."
    video_path, frames, timestamps = get_video_frames(video_url, num_frames=64)

    response = inference(video_path, prompt)
    print(response)
    # display(Markdown(response))


    # data = parse_json(response)

    # for item in data:
    #     start_time = item["start_time"]
    #     end_time = item["end_time"]
    #     description = item["description"]

    #     display(Markdown(f"**{start_time} - {end_time}:**\t\t" + description))

    #     start_time = time_to_seconds(start_time)
    #     end_time = time_to_seconds(end_time)
    #     current_frames = []
    #     for frame, timestamp in zip(frames, timestamps):
    #         if timestamp[0] > start_time and timestamp[1] < end_time:
    #             current_frames.append(frame)
        
    #     current_frames = np.array(current_frames)
    #     current_image_grid = create_image_grid(current_frames, num_columns=8)

    #     display(current_image_grid.resize((480, (int(len(current_frames) / 8) + 1) * 60)))