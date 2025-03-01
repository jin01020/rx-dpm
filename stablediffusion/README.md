# RX-DDIM (Stable Diffusion V2)
This is the codebase for RX-DDIM with Stable Diffusion V2.
This repository is based on [Stability-AI/stablediffusion](https://github.com/Stability-AI/stablediffusion).

## Dependencies
Please refer to the instructions of [original repo](https://github.com/Stability-AI/stablediffusion).

## Pre-trained models

We use the pre-trained models downloaded from [here](https://huggingface.co/stabilityai/stable-diffusion-2-1-base).

## Text-to-Image Sampling
We can sample by RX-DDIM with k=2 which performs extrapolates every 2 steps with an option ```--ext 2```.
```
python scripts/txt2img.py --ckpt [path_to_model_ckpt] --config [path_to_config.yaml] --device cuda --n_samples [batch_size] -n_iter [iterations] --outdir [save_dir_root] --steps [steps] --ext 2 PROMPT_OPTION
```

For DDIM (baseline) sampling, set the extrapolation option to ```--ext 1```.

### Using a single prompt
To sample using a prompt option, use ```--prompt [text_prompt]``` for PROMPT_OPTION.

### Using a file with prompts separated by newlines
To sample using a file with prompts separated by newlines, use ```--from_file [path_to_file]``` for PROMPT_OPTION.

### Using a json file 
To sample using a json file with the format below, use ```--from_json [path_to_json_file]``` for PROMPT_OPTION.
This option is added for the evaluation on [COCO2014 dataset](https://cocodataset.org).
```
{
    "sample": [
        {
            "image_id": XXXXXX,
            "id": YYYYYY,
            "caption": "A caption of image_id XXXXXX.",
            "file_name": "file_name.jpg"
        },
        ...
    ]
}
```
