import asyncio
import os
import signal
import argparse
import json
import time
from nats.aio.client import Client as NATS
from nats.aio.errors import ErrConnectionClosed, ErrTimeout, ErrNoServers

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
        print("Using authorization token from NATS_AUTH_TOKEN environment variable")
    else:
        print("No NATS_AUTH_TOKEN environment variable found, connecting without token")
    
    # Connect to NATS server
    try:
        await nc.connect(**connect_options)
        print(f"Connected to NATS server at {nc.connected_url.netloc}")
        return nc
    except ErrNoServers as e:
        print(f"Could not connect to NATS server: {e}")
        return None

async def run(nats_server, request_subject):
    # Connect to NATS
    nc = await connect_to_nats(nats_server)
    if not nc:
        return
    
    try:
        # Request counter
        request_count = 0
        
        # Loop to send requests every second
        while True:
            try:
                request_count += 1
                request_data = {
                    "request_id": request_count,
                    "timestamp": time.time(),
                    "message": f"Request #{request_count}"
                }
                
                # Convert data to JSON and then to bytes
                request_payload = json.dumps(request_data).encode()
                
                print(f"Sending request #{request_count} to '{request_subject}'")
                
                # Send request with timeout
                response = await nc.request(
                    request_subject, 
                    request_payload, 
                    timeout=2.0
                )
                
                # Process response
                response_data = response.data.decode()
                try:
                    response_json = json.loads(response_data)
                    print(f"Received response: {response_json}")
                except json.JSONDecodeError:
                    print(f"Received non-JSON response: {response_data}")
                
            except ErrTimeout:
                print(f"Request #{request_count} timed out")
            except Exception as e:
                print(f"Error sending request: {e}")
            
            # Wait for 1 second before sending the next request
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Close the connection
        await nc.close()
        print("Connection to NATS closed")

async def main_async(nats_server, request_subject):
    # Handle graceful shutdown
    loop = asyncio.get_running_loop()
    
    # Handle signals for graceful shutdown
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda: asyncio.create_task(shutdown(loop)))
    
    await run(nats_server, request_subject)

async def shutdown(loop):
    print("Shutting down...")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    
    for task in tasks:
        task.cancel()
    
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()

def parse_arguments():
    parser = argparse.ArgumentParser(description='NATS Worker')
    parser.add_argument('--nats-server', 
                        type=str, 
                        default='nats://localhost:4222',
                        help='NATS server address (default: nats://localhost:4222)')
    parser.add_argument('--subject', 
                        type=str, 
                        default='example.request',
                        help='NATS request subject (default: example.request)')
    return parser.parse_args()

def main():
    args = parse_arguments()
    print(f"Connecting to NATS server: {args.nats_server}")
    print(f"Using request subject: {args.subject}")
    asyncio.run(main_async(args.nats_server, args.subject))

if __name__ == "__main__":
    main()
