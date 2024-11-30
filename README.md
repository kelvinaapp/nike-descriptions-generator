# Nike Product Description Generator

A FastAPI-based web application that generates product descriptions for Nike shoes using GPT-2 model. The application is containerized and can be deployed to Google Cloud Run or Kubernetes.

## Features

- Generate creative product descriptions for Nike shoes
- Web interface for easy interaction
- Containerized application using Docker
- Cloud-ready with Google Cloud Run and Kubernetes support
- Pre-trained model stored in Google Cloud Storage

## Prerequisites

- Python 3.8+
- Docker (for containerization)
- Google Cloud SDK (for deployment)
- Access to Google Cloud Storage

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/nike-descriptions-generator.git
cd nike-descriptions-generator
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Local Development

Run the application locally:
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

## Docker Build

Build the Docker image:
```bash
docker build -t nike-description-generator .
```

## Deployment

### Google Cloud Run
The application can be deployed to Google Cloud Run using the included `cloudbuild.yaml` configuration.

### Kubernetes
Kubernetes deployment files are available in the `kubernetes` directory.

## Project Structure

- `app/` - Application source code
  - `main.py` - FastAPI application entry point
  - `generate.py` - Product description generation logic
- `kubernetes/` - Kubernetes deployment configurations
- `model/` - Model storage directory
- `dataset/` - Training data directory
- `Dockerfile` - Container configuration
- `requirements.txt` - Python dependencies

## License

This project is licensed under the MIT License - see the LICENSE file for details.
