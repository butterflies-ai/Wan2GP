import asyncio
import os
import signal
import argparse
import json
import logging
import uuid
import socket
import requests
import subprocess
from pathlib import Path

from google.cloud import storage
from t2v_utils import text_to_video
from i2v_utils import image_to_video
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrConnectionClosed, ErrTimeout, ErrNoServers

def maybe_load_env():
    # Try to import dotenv for environment variable loading
    try:
        from dotenv import load_dotenv
        # Check if .env file exists and load it
        env_path = Path('.') / '.env'
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            # We'll use print here since logging is not set up yet
            print(f"Loaded environment variables from {env_path.absolute()}")
        else:
            print("No .env file found in current directory")
    except ImportError:
        print("python-dotenv package not installed. Environment variables will not be loaded from .env file.")
        print("Install with: pip install python-dotenv")

# Set up logging
def setup_logging(log_level, log_file=None):
    """
    Set up logging with the specified log level and optional log file.
    
    Args:
        log_level (str): Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file (str, optional): Path to log file. If None, logs only to console.
    """
    # Convert string log level to logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")
    
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(numeric_level)
    
    # Clear any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log file is specified
    if log_file:
        # Create directory for log file if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def generate_default_worker_id():
    """
    Generate a default worker ID based on hostname and a UUID.
    
    Returns:
        str: A worker ID in the format 'hostname-uuid'
    """
    hostname = socket.gethostname()
    unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID for brevity
    return f"{hostname}-{unique_id}"

async def connect_to_nats(nats_server):
    # Create a NATS client
    nc = NATS()
    
    # Get auth token from environment variable
    auth_token = os.environ.get('NATS_AUTH_TOKEN')
    
    # Connect options
    connect_options = {
        'servers': [nats_server],
        'reconnect_time_wait': 5,
        'max_reconnect_attempts': 60,
        'name': "python-worker"
    }
    
    # Add token if available
    if auth_token:
        connect_options['token'] = auth_token
        logging.info("Using authorization token from NATS_AUTH_TOKEN environment variable")
    else:
        logging.warning("No NATS_AUTH_TOKEN environment variable found, connecting without token")
    
    # Connect to NATS server
    try:
        await nc.connect(**connect_options)
        logging.info(f"Connected to NATS server at {nc.connected_url.netloc}")
        return nc
    except ErrNoServers as e:
        logging.error(f"Could not connect to NATS server: {e}")
        return None

async def process_text_to_video_request(request_data, callback):
    """
    Process a text-to-video generation request.
    
    Args:
        request_data (dict): Request data containing parameters for text-to-video generation
        
    Returns:
        dict: Response data with status and result
    """
    try:
        # Extract parameters from request
        prompt = request_data.get("prompt", "")
        negative_prompt = request_data.get("negativePrompt", "")
        num_frames = int(request_data.get("frameNum", 65))
        seed = int(request_data.get("seed", -1))
        guidance_scale = float(request_data.get("guidance_scale", 5.0))
        steps = int(request_data.get("steps", 30))
        width = int(request_data.get("maxAreaWidth", 832))
        height = int(request_data.get("maxAreaHeight", 480))
        
        # Generate unique output filename
        output_dir = Path("./output")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"t2v_{uuid.uuid4()}.mp4"
        
        logging.info(f"Processing text-to-video request with prompt: '{prompt}'")
        logging.debug(f"T2V parameters: frames={num_frames}, steps={steps}, guidance={guidance_scale}, seed={seed}")
        
        # Call text_to_video function
        result = await text_to_video(
            prompt=prompt,
            output_file=str(output_file),
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            callback=callback
        )
        
        if result["status"] == "success":
            logging.info(f"Text-to-video generation successful: {result['output_file']}")
        else:
            logging.error(f"Text-to-video generation failed: {result.get('message', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        logging.exception(f"Error processing text-to-video request: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

async def process_image_to_video_request(request_data, callback):
    """
    Process an image-to-video generation request.
    
    Args:
        request_data (dict): Request data containing parameters for image-to-video generation
        
    Returns:
        dict: Response data with status and result
    """
    try:
        # Extract parameters from request
        prompt = request_data.get("prompt", "")
        negative_prompt = request_data.get("negativePrompt", "")
        num_frames = int(request_data.get("frameNum", 65))
        seed = int(request_data.get("seed", -1))
        guidance_scale = float(request_data.get("guidance_scale", 5.0))
        steps = int(request_data.get("steps", 30))
        width = int(request_data.get("maxAreaWidth", 832))
        height = int(request_data.get("maxAreaHeight", 480))
        
        image_url = request_data.get("imageUrl", "")
        if not image_url:
            logging.error("No image URL provided in request")
            return {
                "status": "error",
                "message": "No image URL provided"
            }
        
        # Download image from URL
        try:
            response = requests.get(image_url)
            response.raise_for_status()
            image_bytes = response.content
        except Exception as e:
            logging.error(f"Failed to download image from URL: {e}")
            return {
                "status": "error",
                "message": f"Failed to download image: {e}"
            }
        
        # Save image to temporary file
        temp_image_path = Path("./output") / f"temp_image_{uuid.uuid4()}.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(image_bytes)
        
        # Generate unique output filename
        output_dir = Path("./output")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"i2v_{uuid.uuid4()}.mp4"
        
        logging.info(f"Processing image-to-video request with prompt: '{prompt}'")
        logging.debug(f"I2V parameters: frames={num_frames}, steps={steps}, guidance={guidance_scale}, seed={seed}")
        
        # Call image_to_video function
        result = await image_to_video(
            image_path=str(temp_image_path),
            prompt=prompt,
            output_file=str(output_file),
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            callback=callback
        )
        
        # Clean up temporary image file
        try:
            os.remove(temp_image_path)
        except Exception as e:
            logging.warning(f"Failed to remove temporary image file: {e}")
        
        if result["status"] == "success":
            logging.info(f"Image-to-video generation successful: {result['output_file']}")
        else:
            logging.error(f"Image-to-video generation failed: {result.get('message', 'Unknown error')}")
        
        return result
        
    except Exception as e:
        logging.exception(f"Error processing image-to-video request: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

def extract_frame_from_video(video_path, frame_number=0, output_dir=None):
    """
    Extract a specific frame from a video file using ffmpeg.
    
    Args:
        video_path (str): Path to the video file
        frame_number (int): The frame number to extract (0-indexed)
        output_dir (str, optional): Directory to save the frame. If None, uses the same directory as the video.
    
    Returns:
        str: Path to the extracted frame image file, or None if extraction failed
    """
    try:
        video_path = Path(video_path)
        
        # Ensure video exists
        if not video_path.exists():
            logging.error(f"Video file not found: {video_path}")
            return None
        
        # Create output directory if specified
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)
        else:
            output_dir = video_path.parent
        
        # Generate output filename
        frame_filename = f"{video_path.stem}_frame_{frame_number}.jpg"
        frame_path = output_dir / frame_filename
        
        # Build ffmpeg command
        # -ss specifies the timestamp, -vframes 1 means extract one frame
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-vf", f"select=eq(n\\,{frame_number})",  # Select the specific frame by number
            "-vframes", "1",
            "-q:v", "2",  # Quality setting (2 is high quality)
            str(frame_path),
            "-y"  # Overwrite if exists
        ]
        
        logging.info(f"Extracting frame {frame_number} from {video_path}")
        logging.debug(f"Running command: {' '.join(cmd)}")
        
        # Run ffmpeg command
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Check if successful
        if result.returncode != 0:
            logging.error(f"ffmpeg failed with return code {result.returncode}")
            logging.error(f"ffmpeg error: {result.stderr}")
            return None
        
        # Check if file was created
        if not frame_path.exists():
            logging.error(f"Frame extraction failed: output file not created")
            return None
            
        logging.info(f"Successfully extracted frame to {frame_path}")
        return str(frame_path)
        
    except Exception as e:
        logging.exception(f"Error extracting frame from video: {e}")
        return None

def upscale_video(video_path):
    """
    Upscale a video using video2x in Docker.
    
    Args:
        video_path (str): Path to the input video file
    
    Returns:
        str: Path to the upscaled video file, or None if upscaling failed
    """
    logging.info(f"Upscaling video: {video_path}")
    try:
        video_path = Path(video_path)
        
        # Ensure video exists
        if not video_path.exists():
            logging.error(f"Video file not found: {video_path}")
            return None
        
        # Use the same directory as input for output
        output_dir = video_path.parent
        
        # Generate output filename
        output_filename = f"{video_path.stem}_upscaled.mp4"
        output_path = output_dir / output_filename
        
        # Get absolute paths for Docker volume mounting
        input_dir_abs = str(video_path.parent.absolute())
        
        # Build Docker command
        cmd = [
            "docker", "run",
            "--gpus", "all",
            "--rm",
            "-v", f"{input_dir_abs}:/host",
            "ghcr.io/k4yt3x/video2x:6.3.0",
            "-i", f"{video_path.name}",
            "-o", f"{output_filename}",
            "-p", "realesrgan",
            "-s", "4",
            "--realesrgan-model", "realesr-generalv3"
        ]
        
        logging.info(f"Upscaling video: {video_path}")
        logging.debug(f"Running command: {' '.join(cmd)}")
        
        # Run Docker command
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Check if successful
        if result.returncode != 0:
            logging.error(f"Video upscaling failed with return code {result.returncode}")
            logging.error(f"Docker error: {result.stderr}")
            return None
        
        # Check if file was created
        if not output_path.exists():
            logging.error(f"Video upscaling failed: output file not created")
            return None
            
        logging.info(f"Successfully upscaled video to {output_path}")
        return str(output_path)
        
    except Exception as e:
        logging.exception(f"Error upscaling video: {e}")
        return None

async def process_request(request_data, callback):
    """
    Process a request based on its type.
    
    Args:
        request_data (dict): Request data containing type and parameters
        
    Returns:
        dict: Response data with status and result
    """
    if (request_data.get("imageUrl", "") != ""):
        request_type = "image_to_video"
    elif (request_data.get("prompt", "") != ""):
        request_type = "text_to_video"
    else:
        logging.error(f"Unknown request type: {request_data}")
        return {
            "status": "error",
            "message": "Unknown request type"
        }
    
    if request_type == "text_to_video":
        return await process_text_to_video_request(request_data, callback)
    else:
        return await process_image_to_video_request(request_data, callback)

def upload_to_gcs(local_file_path, gcs_bucket_name, gcs_path):
    """
    Upload a file to Google Cloud Storage.
    
    Args:
        local_file_path (str): Path to the local file to upload
        gcs_bucket_name (str): Name of the GCS bucket
        gcs_path (str): Path within the bucket to upload to
        
    Returns:
        str: Public URL of the uploaded file or None if upload failed
    """
    try:
        # Create a storage client
        storage_client = storage.Client()
        
        # Get the bucket
        bucket = storage_client.bucket(gcs_bucket_name)
        
        # Generate a destination blob name
        file_name = os.path.basename(local_file_path)
        if gcs_path:
            # Remove leading/trailing slashes from gcs_path
            gcs_path = gcs_path.strip('/')
            if gcs_path:
                blob_name = f"{gcs_path}/{file_name}"
            else:
                blob_name = file_name
        else:
            blob_name = file_name
            
        # Create a blob
        blob = bucket.blob(blob_name)
        
        # Upload the file
        logging.info(f"Uploading {local_file_path} to gs://{gcs_bucket_name}/{blob_name}")
        blob.upload_from_filename(local_file_path)
        
        # Make the blob publicly readable if possible
        try:
            blob.make_public()
            public_url = blob.public_url
            logging.info(f"File uploaded successfully. Public URL: {public_url}")
            return public_url
        except Exception as e:
            logging.warning(f"Could not make blob public: {e}")
            # Return a GCS URI instead
            gcs_uri = f"gs://{gcs_bucket_name}/{blob_name}"
            logging.info(f"File uploaded successfully. GCS URI: {gcs_uri}")
            return gcs_uri
            
    except Exception as e:
        logging.exception(f"Error uploading file to GCS: {e}")
        return None

async def post_result(nc, result_subject, data):
    """
    Post the result to the result subject using request/response pattern.
    
    Args:
        nc: NATS client
        result_subject (str): Subject to post results to
        data (dict): Data to post
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Extract task_id for logging
        task_id = data.get("taskId", "unknown")
        
        # Convert data to JSON and then to bytes
        result_payload = json.dumps(data).encode()
        
        logging.info(f"Posting result for task {task_id} to '{result_subject}'")
        logging.debug(f"Result data: {data}")
        
        response = None
        for i in range(3):
            try:
                # Send request with timeout
                response = await nc.request(
                    result_subject, 
                    result_payload, 
                    timeout=5.0
                )
                break  # Break the loop if request is successful
            except Exception as e:
                logging.warning(f"Error posting result for task {task_id}, attempt {i+1} of 3: {str(e)}")
                if i < 2:  # Only sleep if we're going to retry
                    await asyncio.sleep(1.0)
                else:
                    # If all retries failed, re-raise the exception to be caught by the outer try/except
                    raise
        
        # If we didn't get a response after all retries, return False
        if response is None:
            logging.error(f"Failed to post result for task {task_id} after 3 attempts")
            return False
            
        # Process response
        response_data = response.data.decode()
        try:
            response_json = json.loads(response_data)
            logging.info(f"Received response from result posting: {response_json}")
            return True
        except json.JSONDecodeError:
            logging.warning(f"Received non-JSON response from result posting: {response_data}")
            return True  # Still consider it successful
        
    except ErrTimeout:
        logging.error(f"Timeout posting result for task {task_id}")
        return False
    except Exception as e:
        logging.exception(f"Error posting result: {e}")
        return False

async def run(nats_server, request_subject, result_subject, polling_interval, worker_id, gcs_bucket=None, gcs_path=None):
    # Connect to NATS
    nc = await connect_to_nats(nats_server)
    if not nc:
        return
    
    logging.info(f"Using polling interval of {polling_interval} seconds")
    logging.info(f"Worker ID: {worker_id}")
    
    if gcs_bucket:
        logging.info(f"Will upload videos to GCS bucket: {gcs_bucket}")
        if gcs_path:
            logging.info(f"GCS path: {gcs_path}")
    
    current_task_id = "unknown"
        
    async def progress_callback(p, _):
        logging.debug(f"Received progress update: {p}")
        try:
            await nc.publish(result_subject, json.dumps({
                "status": "processing",
                "progress": p,
                "taskId": current_task_id,
                "workerId": worker_id
            }).encode())
            await nc.flush()
        except Exception as e:
            logging.warning(f"Failed to queue progress update: {e}")

    try:
        # Loop to poll for requests
        while True:
            try:
                logging.debug(f"Polling '{request_subject}' for new requests")
                
                # Send request with timeout, including worker_id
                poll_payload = json.dumps({
                    "action": "poll",
                    "workerId": worker_id
                }).encode()
                
                response = await nc.request(
                    request_subject, 
                    poll_payload, 
                    timeout=2.0
                )
                
                # Process response
                response_data = response.data.decode()
                try:
                    request_data = json.loads(response_data)
                    
                    # Check if there's a request to process
                    if request_data.get("status") == "no_request":
                        logging.debug("No pending requests")
                        # Only sleep if there are no requests to process
                        await asyncio.sleep(polling_interval)
                    else:
                        # Extract request ID
                        task_id = request_data.get("taskId", "unknown")
                        current_task_id = task_id
                        logging.info(f"Received request {task_id} from polling '{request_subject}'")
                        logging.debug(f"Request data: {request_data}")
                        
                        # Process request synchronously
                        logging.info(f"Processing request {task_id} synchronously")
                        
                        # Process the request
                        result = await process_request(request_data.get("request"), progress_callback)
                        
                        # Check if video generation was successful and output file exists
                        if result.get("status") == "success" and result.get("output_file") and os.path.exists(result["output_file"]):
                            # Extract a thumbnail from the middle of the video (usually frame 32 for 65-frame videos)
                            frame_number = int(request_data.get("request", {}).get("frameNum", 65)) // 2
                            thumbnail_path = extract_frame_from_video(result["output_file"], frame_number)
                            
                            if thumbnail_path:
                                result["thumbnail_path"] = thumbnail_path
                                logging.info(f"Generated thumbnail at {thumbnail_path}")
                                
                                # Upload thumbnail to GCS if GCS bucket is specified
                                if gcs_bucket:
                                    thumbnail_gcs_url = upload_to_gcs(thumbnail_path, gcs_bucket, gcs_path)
                                    if thumbnail_gcs_url:
                                        result["thumbnail_url"] = thumbnail_gcs_url
                                        
                            if request_data.get("upscaleTo720p", False):
                                upscaled_video_path = upscale_video(result["output_file"])
                                if upscaled_video_path:
                                    result["output_file"] = upscaled_video_path
                        
                        # Upload the video to GCS if bucket is specified
                        if gcs_bucket and result.get("status") == "success" and result.get("output_file"):
                            gcs_url = upload_to_gcs(result["output_file"], gcs_bucket, gcs_path)
                            if gcs_url:
                                result["gcs_url"] = gcs_url
                        
                        # Post the result
                        result_data = {
                            'status': 'success', 
                            'taskId': task_id, 
                            'workerId': worker_id, 
                            'result': result,
                            'progress': 100
                        }
                        
                        # Include video URL directly in the top level for compatibility
                        if result.get("gcs_url"):
                            result_data["videoUrl"] = result["gcs_url"]
                            
                        # Include thumbnail URL directly in the top level for easy access
                        if result.get("thumbnail_url"):
                            result_data["thumbnailUrl"] = result["thumbnail_url"]
                        
                        success = await post_result(nc, result_subject, result_data)
                        if success:
                            logging.info(f"Successfully posted result for request {task_id}")
                        else:
                            logging.error(f"Failed to post result for request {task_id}")
                        
                        # Continue immediately to poll for the next request
                        # No sleep here
                        
                except json.JSONDecodeError:
                    logging.warning(f"Received non-JSON response from polling: {response_data}")
                    await asyncio.sleep(polling_interval)
                
            except ErrTimeout:
                logging.debug("Polling request timed out, will retry")
                await asyncio.sleep(polling_interval)
            except Exception as e:
                logging.exception(f"Error processing request: {e}")
                
                # If we were processing a request when the error occurred, try to send an error response
                if 'task_id' in locals():
                    try:
                        error_result = {
                            "status": "error",
                            "message": f"Internal server error: {str(e)}",
                            "taskId": task_id,
                            "workerId": worker_id
                        }
                        await post_result(nc, result_subject, error_result)
                    except Exception as e2:
                        logging.error(f"Failed to post error result: {e2}")
                
                # Sleep after an error to avoid tight error loops
                await asyncio.sleep(polling_interval)
            
    except Exception as e:
        logging.exception(f"Error in run loop: {e}")
    finally:
        # Cancel the progress update task
        if 'progress_task' in locals():
            progress_task.cancel()
            
        # Close the connection
        await nc.close()
        logging.info("Connection to NATS closed")

async def main_async(nats_server, request_subject, result_subject, polling_interval, worker_id, gcs_bucket=None, gcs_path=None):
    # Handle graceful shutdown
    loop = asyncio.get_running_loop()
    
    # Handle signals for graceful shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(loop)))
    
    await run(nats_server, request_subject, result_subject, polling_interval, worker_id, gcs_bucket, gcs_path)

async def shutdown(loop):
    logging.info("Shutting down...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    
    for task in tasks:
        task.cancel()
    
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Video Generation Worker')
    parser.add_argument('--nats-server', 
                        type=str, 
                        default='nats://localhost:4222',
                        help='NATS server address (default: nats://localhost:4222)')
    parser.add_argument('--request-subject', 
                        type=str, 
                        default='boba.video.request',
                        help='NATS subject to poll for requests (default: boba.video.request)')
    parser.add_argument('--result-subject', 
                        type=str, 
                        default='boba.video.reply',
                        help='NATS subject to post results to (default: boba.video.reply)')
    parser.add_argument('--polling-interval',
                        type=float,
                        default=1.0,
                        help='Interval in seconds between polling requests when idle (default: 1.0)')
    parser.add_argument('--worker-id',
                        type=str,
                        help='Unique ID for this worker (default: auto-generated)')
    parser.add_argument('--log-level',
                        type=str,
                        default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')
    parser.add_argument('--log-file',
                        type=str,
                        help='Log file path (default: None, logs to console only)')
    # Add GCS arguments
    parser.add_argument('--gcs-bucket',
                        type=str,
                        help='Google Cloud Storage bucket to upload videos to')
    parser.add_argument('--gcs-path',
                        type=str,
                        default='',
                        help='Path within the GCS bucket to upload videos to')
    
    args = parser.parse_args()
    
    # Validate polling interval
    if args.polling_interval < 0.1:
        print(f"Warning: Polling interval {args.polling_interval}s is too small, using 0.1s")
        args.polling_interval = 0.1
    
    # Generate worker ID if not provided
    if not args.worker_id:
        args.worker_id = generate_default_worker_id()
    
    return args

def main():
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging(args.log_level, args.log_file)
    
    logging.info(f"Starting Video Generation Worker")
    logging.info(f"Worker ID: {args.worker_id}")
    
    maybe_load_env()
    
    # Create output directory for generated videos
    os.makedirs("./output", exist_ok=True)
        
    logging.info(f"Connecting to NATS server: {args.nats_server}")
    logging.info(f"Polling request subject: {args.request_subject}")
    logging.info(f"Posting results to subject: {args.result_subject}")
    logging.info(f"Polling interval: {args.polling_interval} seconds")
    logging.info(f"Log level: {args.log_level}")
    if args.log_file:
        logging.info(f"Logging to file: {args.log_file}")
    
    # Check for GCS bucket
    if args.gcs_bucket:
        logging.info(f"Will upload videos to GCS bucket: {args.gcs_bucket}")
        if args.gcs_path:
            logging.info(f"GCS path: {args.gcs_path}")
            
    asyncio.run(main_async(args.nats_server, args.request_subject, args.result_subject, 
                          args.polling_interval, args.worker_id, args.gcs_bucket, args.gcs_path))

if __name__ == "__main__":
    main()
