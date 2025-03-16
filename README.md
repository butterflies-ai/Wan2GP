# NATS Request-Response Example

This project demonstrates how to use NATS for request-response communication patterns in Python.

## Requirements

- Python 3.7+
- NATS server running (locally or remotely)
- Python packages:
  - `nats-py`

## Installation

Install the required Python package:

```bash
pip install nats-py
```

## Running a NATS Server

If you don't have a NATS server running, you can:

1. Install NATS server: https://docs.nats.io/running-a-nats-service/introduction/installation
2. Run it with default settings: `nats-server`
3. Or use Docker: `docker run -p 4222:4222 -p 8222:8222 nats`

## Authentication

The scripts support token-based authentication with NATS. To use this feature:

1. Set the `NATS_AUTH_TOKEN` environment variable with your token:

```bash
# On Linux/macOS
export NATS_AUTH_TOKEN=your_secret_token

# On Windows
set NATS_AUTH_TOKEN=your_secret_token
```

2. Configure your NATS server to use token authentication:

```bash
# Example NATS server command with token auth
nats-server --auth your_secret_token
```

If the `NATS_AUTH_TOKEN` environment variable is not set, the scripts will attempt to connect without authentication.

## Scripts

### 1. Worker (Requester)

The `worker.py` script sends requests to a NATS topic every second and processes the responses.

```bash
# Run with default NATS server (localhost:4222) and default subject (example.request)
python worker.py

# Run with custom subject
python worker.py --subject my.custom.subject

# Run with custom NATS server
python worker.py --nats-server nats://custom-server:4222

# Run with both custom subject and server
python worker.py --subject my.custom.subject --nats-server nats://custom-server:4222
```

### 2. Responder

The `responder.py` script listens for requests on the NATS topic and responds to them.

```bash
# Run with default NATS server (localhost:4222) and default subject (example.request)
python responder.py

# Run with custom subject
python responder.py --subject my.custom.subject

# Run with custom NATS server
python responder.py --nats-server nats://custom-server:4222

# Run with both custom subject and server
python responder.py --subject my.custom.subject --nats-server nats://custom-server:4222
```

### Helper Scripts

The repository includes helper scripts to run the worker and responder with authentication:

```bash
# Run worker with default subject
./run_with_token.sh

# Run worker with custom subject
./run_with_token.sh my.custom.subject

# Run responder with default subject
./run_responder_with_token.sh

# Run responder with custom subject
./run_responder_with_token.sh my.custom.subject
```

## Testing the Request-Response Pattern

1. Start a NATS server
2. Run the responder in one terminal: `python responder.py --subject test.subject`
3. Run the worker in another terminal: `python worker.py --subject test.subject`

You should see the worker sending requests and receiving responses from the responder.

## Graceful Shutdown

Both scripts handle graceful shutdown. Press Ctrl+C to stop them.

## Customization

You can modify the request and response handling logic in the scripts to fit your specific use case. 