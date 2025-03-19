import os
import time
import torch
import random
import gc
from pathlib import Path
import pprint
# Import WAN modules
try:
    import wan
    from wan.configs import MAX_AREA_CONFIGS, WAN_CONFIGS
    from wan.utils.utils import cache_video
    from mmgp import offload
    from wan.modules.attention import get_attention_modes
except ImportError as e:
    print(f"Warning: Could not import WAN modules: {e}")
    print("Make sure the WAN modules are installed and in your Python path")

# Default data directory
DATA_DIR = "ckpts"

def get_attention_mode(attention="auto"):
    """
    Decide which attention mode to use: either the user choice or auto fallback.
    """
    installed_modes = get_attention_modes()
    
    if attention == "auto":
        for candidate in ["sage2", "sage", "sdpa"]:
            if candidate in installed_modes:
                return candidate
        return "sdpa"  # last fallback
    elif attention in installed_modes:
        return attention
    else:
        raise ValueError(
            f"Requested attention mode '{attention}' not installed. "
            f"Installed modes: {installed_modes}"
        )

def load_t2v_model(model_filename, text_encoder_filename, is_720p=False):
    """
    Load the text-to-video model with a specific size config and text encoder.
    """
    if is_720p:
        print("Loading 14B-720p t2v model ...")
        cfg = WAN_CONFIGS['t2v-14B']
        wan_model = wan.WanT2V(
            config=cfg,
            checkpoint_dir=DATA_DIR,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            i2v720p=True,
            model_filename=model_filename,
            text_encoder_filename=text_encoder_filename
        )
    else:
        print("Loading 14B-480p t2v model ...")
        cfg = WAN_CONFIGS['t2v-14B']
        wan_model = wan.WanT2V(
            config=cfg,
            checkpoint_dir=DATA_DIR,
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            model_filename=model_filename,
            text_encoder_filename=text_encoder_filename
        )
    # Pipe structure
    pipe = {
        "transformer": wan_model.model,
        "text_encoder": wan_model.text_encoder.model,
        "vae": wan_model.vae.model
    }
    return wan_model, pipe

def download_models_if_needed(transformer_filename, text_encoder_filename, local_folder=DATA_DIR):
    """
    Checks if all required WAN 2.1 files exist locally under 'ckpts/'.
    If not, downloads them from a Hugging Face Hub repo.
    """
    import os
    from pathlib import Path

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required for automatic model download. "
            "Please install it via `pip install huggingface_hub`."
        ) from e

    # Identify just the filename portion for each path
    def basename(path_str):
        return os.path.basename(path_str)

    repo_id = "DeepBeepMeep/Wan2.1"
    target_root = local_folder

    # Required files
    needed_files = [
        "Wan2.1_VAE.pth",
        "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        basename(text_encoder_filename),
        basename(transformer_filename),
    ]

    # The xlm-roberta-large folder
    subfolder_name = "xlm-roberta-large"
    if not Path(os.path.join(target_root, subfolder_name)).exists():
        snapshot_download(repo_id=repo_id, allow_patterns=subfolder_name + "/*", local_dir=target_root)

    for filename in needed_files:
        local_path = os.path.join(target_root, filename)
        if not os.path.isfile(local_path):
            print(f"File '{filename}' not found locally. Downloading from {repo_id} ...")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=target_root
            )
        else:
            # Already present
            pass

    print("All required files are present.")

async def text_to_video(
    prompt,
    output_file=None,
    negative_prompt="",
    num_frames=65,
    width=832,
    height=480,
    steps=30,
    guidance_scale=5.0,
    seed=-1,
    flow_shift=3.0,
    enable_riflex=False,
    is_720p=False,
    profile_no=4,
    attention="auto",
    quantize_transformer=False,
    compile=False,
    transformer_file=None,
    text_encoder_file=None,
    vae_tile_size=None,
    callback=None
):
    """
    Generate a video from a text prompt using WAN text-to-video model.
    
    Args:
        prompt (str): Text prompt for video generation
        output_file (str, optional): Path to save the output video. If None, a random filename is generated.
        negative_prompt (str, optional): Negative prompt for generation. Defaults to "".
        num_frames (int, optional): Number of frames to generate. Defaults to 65.
        width (int, optional): Width of the output video. Defaults to 832.
        height (int, optional): Height of the output video. Defaults to 480.
        steps (int, optional): Number of sampling steps. Defaults to 30.
        guidance_scale (float, optional): Classifier-free guidance scale. Defaults to 5.0.
        seed (int, optional): Random seed. -1 means random each time. Defaults to -1.
        flow_shift (float, optional): Flow shift parameter. Defaults to 3.0.
        enable_riflex (bool, optional): Enable RIFLEx for longer videos. Defaults to False.
        is_720p (bool, optional): Whether to use the 720p model. Defaults to False.
        profile_no (int, optional): Memory usage profile number [1..5]. Defaults to 4.
        attention (str, optional): Which attention to use: auto, sdpa, sage, sage2, flash. Defaults to "auto".
        quantize_transformer (bool, optional): Use on-the-fly transformer quantization. Defaults to False.
        compile (bool, optional): Enable PyTorch 2.0 compile for the transformer. Defaults to False.
        transformer_file (str, optional): Path to transformer model file. If None, uses default.
        text_encoder_file (str, optional): Path to text encoder file. If None, uses default.
        vae_tile_size (int, optional): VAE tile size. If None, automatically determined.
        callback (function, optional): Callback function to be called with progress information.
    Returns:
        dict: A dictionary containing the status of the generation and the path to the output file.
    """
    try:
        start_time = time.time()
        
        # Set up output file
        if output_file is None:
            output_dir = Path("./output")
            output_dir.mkdir(exist_ok=True)
            output_file = output_dir / f"text2video_{int(time.time())}.mp4"
        
        # Set default model files if not provided
        if transformer_file is None:
            # transformer_file = f"{DATA_DIR}/wan2.1_text2video_14B_bf16.safetensors"
            transformer_file = f"{DATA_DIR}/wan2.1_text2video_14B_quanto_int8.safetensors"
            # if is_720p:
            #     transformer_file = f"{DATA_DIR}/wan2.1_text2video_14B_bf16.safetensors"
            # else:
            #     transformer_file = f"{DATA_DIR}/t2v-14B-480p.safetensors"
        
        if text_encoder_file is None:
            text_encoder_file = f"{DATA_DIR}/models_t5_umt5-xxl-enc-quanto_int8.safetensors"
        
        # Setup environment
        offload.default_verboseLevel = 1
        
        # Set attention mode
        chosen_attention = get_attention_mode(attention)
        offload.shared_state["_attention"] = chosen_attention
        
        # Make sure we have the needed models locally
        download_models_if_needed(transformer_file, text_encoder_file)
        
        # Load model
        wan_model, pipe = load_t2v_model(
            model_filename=transformer_file,
            text_encoder_filename=text_encoder_file,
            is_720p=is_720p
        )
        wan_model._interrupt = False
        
        # Set up offloading
        kwargs = {}
        if profile_no == 2 or profile_no == 4:
            budgets = {"transformer": 100, "text_encoder": 100, "*": 1000}
            kwargs["budgets"] = budgets
        elif profile_no == 3:
            kwargs["budgets"] = {"*": "70%"}
        
        compile_choice = "transformer" if compile else ""
        
        # Create the offload object
        offloadobj = offload.profile(
            pipe,
            profile_no=profile_no,
            compile=compile_choice,
            quantizeTransformer=quantize_transformer,
            **kwargs
        )
        
        # Handle random seed
        if seed < 0:
            seed = random.randint(0, 999999999)
        print(f"Using seed={seed}")
        
        # Ensure frame count is 4n+1
        frame_count = (num_frames // 4) * 4 + 1
        
        # Determine VAE tile size based on available VRAM if not specified
        if vae_tile_size is None:
            device_mem_capacity = torch.cuda.get_device_properties(0).total_memory / 1048576
            if device_mem_capacity >= 28000:  # 81 frames 720p requires about 28 GB VRAM
                vae_tile_size = 0  # No tiling
            elif device_mem_capacity >= 8000:
                vae_tile_size = 256
            else:
                vae_tile_size = 128
        trans = wan_model.model
        trans.enable_teacache = False

        print(f"Using VAE tile size of {vae_tile_size}")
        
        pprint.pprint(wan_model);
        
        # Generate the video
        print(f"Starting text-to-video generation for prompt: {prompt}")
        sample_frames = await wan_model.generate(
            prompt,
            frame_num=frame_count,
            shift=flow_shift,
            sampling_steps=steps,
            guide_scale=guidance_scale,
            n_prompt=negative_prompt,
            seed=seed,
            offload_model=False,
            enable_RIFLEx=enable_riflex,
            VAE_tile_size=vae_tile_size,
            callback=callback,
        )
        
        # Clean up
        offloadobj.unload_all()
        gc.collect()
        torch.cuda.empty_cache()
        
        if sample_frames is None:
            return {
                "status": "error",
                "message": "No frames were returned (maybe generation was aborted or failed)."
            }
        
        # Save the video
        sample_frames = sample_frames.cpu()
        os.makedirs(os.path.dirname(str(output_file)) or ".", exist_ok=True)
        
        cache_video(
            tensor=sample_frames[None],  # shape => [1, c, T, H, W]
            save_file=str(output_file),
            fps=16,
            nrow=1,
            normalize=True,
            value_range=(-1, 1)
        )
        
        end_time = time.time()
        elapsed_s = end_time - start_time
        
        print(f"Done! Output written to {output_file}. Generation time: {elapsed_s:.1f} seconds.")
        
        return {
            "status": "success",
            "output_file": str(output_file),
            "elapsed_time": elapsed_s,
            "seed": seed,
            "prompt": prompt
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        
        return {
            "status": "error",
            "message": str(e)
        }
