import asyncio
import os
import signal
import argparse
import json
import time
import logging
import uuid
import base64
from pathlib import Path

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

from nats.aio.client import Client as NATS
from nats.aio.errors import ErrConnectionClosed, ErrTimeout, ErrNoServers

# Try to import video generation functions
try:
    from t2v_utils import text_to_video
    from i2v_utils import image_to_video
except ImportError as e:
    print(f"Warning: Could not import video generation modules: {e}")
    print("Make sure the video generation modules are installed and in your Python path")

# Create output directory for generated videos
os.makedirs("./output", exist_ok=True)

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

async def process_text_to_video_request(request_data):
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
        negative_prompt = request_data.get("negative_prompt", "")
        num_frames = int(request_data.get("num_frames", 65))
        seed = int(request_data.get("seed", -1))
        guidance_scale = float(request_data.get("guidance_scale", 5.0))
        steps = int(request_data.get("steps", 30))
        width = int(request_data.get("width", 832))
        height = int(request_data.get("height", 480))
        
        # Generate unique output filename
        output_dir = Path("./output")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / f"t2v_{uuid.uuid4()}.mp4"
        
        logging.info(f"Processing text-to-video request with prompt: '{prompt}'")
        logging.debug(f"T2V parameters: frames={num_frames}, steps={steps}, guidance={guidance_scale}, seed={seed}")
        
        # Call text_to_video function
        result = text_to_video(
            prompt=prompt,
            output_file=str(output_file),
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed
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

async def process_image_to_video_request(request_data):
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
        negative_prompt = request_data.get("negative_prompt", "")
        num_frames = int(request_data.get("num_frames", 65))
        seed = int(request_data.get("seed", -1))
        guidance_scale = float(request_data.get("guidance_scale", 5.0))
        steps = int(request_data.get("steps", 30))
        width = int(request_data.get("width", 832))
        height = int(request_data.get("height", 480))
        
        # Get base64 image data
        image_data = request_data.get("image_data", "")
        if not image_data:
            logging.error("No image data provided in request")
            return {
                "status": "error",
                "message": "No image data provided"
            }
        
        # Decode base64 image data
        try:
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            logging.error(f"Failed to decode base64 image data: {e}")
            return {
                "status": "error",
                "message": f"Invalid base64 image data: {e}"
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
        result = image_to_video(
            image_path=str(temp_image_path),
            prompt=prompt,
            output_file=str(output_file),
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed
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

async def process_request(request_data):
    """
    Process a request based on its type.
    
    Args:
        request_data (dict): Request data containing type and parameters
        
    Returns:
        dict: Response data with status and result
    """
    request_type = request_data.get("type", "")
    
    if request_type == "text_to_video":
        return await process_text_to_video_request(request_data)
    elif request_type == "image_to_video":
        return await process_image_to_video_request(request_data)
    else:
        logging.error(f"Unknown request type: {request_type}")
        return {
            "status": "error",
            "message": f"Unknown request type: {request_type}"
        }

async def post_result(nc, result_subject, request_id, result_data):
    """
    Post the result to the result subject using request/response pattern.
    
    Args:
        nc: NATS client
        result_subject (str): Subject to post results to
        request_id (str): ID of the original request
        result_data (dict): Result data to post
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Add request_id to result data
        result_data["request_id"] = request_id
        
        # Convert data to JSON and then to bytes
        result_payload = json.dumps(result_data).encode()
        
        logging.info(f"Posting result for request {request_id} to '{result_subject}'")
        logging.debug(f"Result data: {result_data}")
        
        # Send request with timeout
        response = await nc.request(
            result_subject, 
            result_payload, 
            timeout=5.0
        )
        
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
        logging.error(f"Timeout posting result for request {request_id}")
        return False
    except Exception as e:
        logging.exception(f"Error posting result: {e}")
        return False

async def run(nats_server, request_subject, result_subject):
    # Connect to NATS
    nc = await connect_to_nats(nats_server)
    if not nc:
        return
    
    try:
        # Processing flag to avoid overlapping processing
        is_processing = False
        
        # Loop to poll for requests every second
        while True:
            if not is_processing:
                try:
                    logging.debug(f"Polling '{request_subject}' for new requests")
                    
                    # Send request with timeout
                    poll_payload = json.dumps({"action": "poll"}).encode()
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
                        else:
                            # Extract request ID
                            request_id = request_data.get("request_id", "unknown")
                            logging.info(f"Received request {request_id} from polling '{request_subject}'")
                            logging.debug(f"Request data: {request_data}")
                            
                            # Set processing flag
                            is_processing = True
                            
                            # Process request asynchronously
                            asyncio.create_task(
                                process_and_post_result(nc, result_subject, request_id, request_data, 
                                                       lambda: setattr(is_processing, "_", False))
                            )
                            
                    except json.JSONDecodeError:
                        logging.warning(f"Received non-JSON response from polling: {response_data}")
                    
                except ErrTimeout:
                    logging.debug("Polling request timed out, will retry")
                except Exception as e:
                    logging.exception(f"Error polling for requests: {e}")
            
            # Wait for 1 second before polling again
            await asyncio.sleep(1)
            
    except Exception as e:
        logging.exception(f"Error in run loop: {e}")
    finally:
        # Close the connection
        await nc.close()
        logging.info("Connection to NATS closed")

async def process_and_post_result(nc, result_subject, request_id, request_data, done_callback):
    """
    Process a request and post the result.
    
    Args:
        nc: NATS client
        result_subject (str): Subject to post results to
        request_id (str): ID of the request
        request_data (dict): Request data to process
        done_callback (callable): Callback to call when done
    """
    try:
        # Process the request
        logging.info(f"Processing request {request_id}")
        result = await process_request(request_data)
        
        # Add request_id to result
        result["request_id"] = request_id
        
        # Post the result
        success = await post_result(nc, result_subject, request_id, result)
        if success:
            logging.info(f"Successfully posted result for request {request_id}")
        else:
            logging.error(f"Failed to post result for request {request_id}")
        
    except Exception as e:
        logging.exception(f"Error processing request {request_id}: {e}")
        
        # Try to post error result
        error_result = {
            "status": "error",
            "message": f"Internal server error: {str(e)}",
            "request_id": request_id
        }
        try:
            await post_result(nc, result_subject, request_id, error_result)
        except Exception as e2:
            logging.error(f"Failed to post error result: {e2}")
    
    finally:
        # Reset processing flag
        if done_callback:
            done_callback()

async def main_async(nats_server, request_subject, result_subject):
    # Handle graceful shutdown
    loop = asyncio.get_running_loop()
    
    # Handle signals for graceful shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(loop)))
    
    await run(nats_server, request_subject, result_subject)

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
                        default='video.request',
                        help='NATS subject to poll for requests (default: video.request)')
    parser.add_argument('--result-subject', 
                        type=str, 
                        default='video.result',
                        help='NATS subject to post results to (default: video.result)')
    parser.add_argument('--log-level',
                        type=str,
                        default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level (default: INFO)')
    parser.add_argument('--log-file',
                        type=str,
                        help='Log file path (default: None, logs to console only)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Set up logging
    logger = setup_logging(args.log_level, args.log_file)
    
    logging.info(f"Starting Video Generation Worker")
    
    # Log .env file status now that logging is set up
    env_path = Path('.') / '.env'
    if env_path.exists():
        logging.info(f"Loaded environment variables from {env_path.absolute()}")
    else:
        logging.warning("No .env file found in current directory")
    
    logging.info(f"Connecting to NATS server: {args.nats_server}")
    logging.info(f"Polling request subject: {args.request_subject}")
    logging.info(f"Posting results to subject: {args.result_subject}")
    logging.info(f"Log level: {args.log_level}")
    if args.log_file:
        logging.info(f"Logging to file: {args.log_file}")
    
    asyncio.run(main_async(args.nats_server, args.request_subject, args.result_subject))

if __name__ == "__main__":
    main()
