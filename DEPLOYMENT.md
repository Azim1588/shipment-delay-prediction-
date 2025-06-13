# Deployment Guide

This guide covers different deployment options for the Shipment Delay Prediction System.

## Local Development Deployment

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/shipment-delay-prediction.git
cd shipment-delay-prediction

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download dataset
# Place DataCoSupplyChainDataset.csv in the data/ directory

# Run the application
streamlit run main.py
```

## Docker Deployment

### Build Docker Image

```bash
# Build the image
docker build -t shipment-delay-prediction .

# Run the container
docker run -p 8501:8501 shipment-delay-prediction
```

### Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## Cloud Deployment

### Heroku Deployment

1. **Create Heroku App**

```bash
# Install Heroku CLI
# Create new app
heroku create your-app-name

# Add buildpack
heroku buildpacks:set heroku/python
```

2. **Create Procfile**

```
web: streamlit run main.py --server.port=$PORT --server.address=0.0.0.0
```

3. **Deploy**

```bash
git add .
git commit -m "Deploy to Heroku"
git push heroku main
```

### AWS Deployment

#### EC2 Instance

1. Launch EC2 instance with Ubuntu
2. Install Python and dependencies
3. Clone repository
4. Run with systemd service

#### ECS/Fargate

1. Create Docker image
2. Push to ECR
3. Create ECS service
4. Configure load balancer

### Google Cloud Platform

#### App Engine

1. Create `app.yaml`:

```yaml
runtime: python39
entrypoint: streamlit run main.py --server.port=$PORT --server.address=0.0.0.0

env_variables:
  STREAMLIT_SERVER_PORT: 8080
```

2. Deploy:

```bash
gcloud app deploy
```

#### Cloud Run

1. Build and push Docker image
2. Deploy to Cloud Run
3. Configure environment variables

## Production Considerations

### Environment Variables

```bash
# Production settings
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_ENABLE_CORS=false
export STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=true
```

### Security

1. **HTTPS**: Use SSL/TLS certificates
2. **Authentication**: Implement user authentication
3. **Rate Limiting**: Add rate limiting for API endpoints
4. **Input Validation**: Validate all user inputs
5. **Secrets Management**: Use environment variables for secrets

### Performance

1. **Caching**: Implement caching for model predictions
2. **Load Balancing**: Use load balancers for high traffic
3. **Database**: Use production database for data storage
4. **Monitoring**: Set up application monitoring
5. **Logging**: Configure proper logging

### Monitoring and Logging

#### Application Monitoring

```python
# Add monitoring to main.py
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Log application events
logger.info(f"Application started at {datetime.now()}")
```

#### Health Checks

```python
# Add health check endpoint
@app.route('/health')
def health_check():
    return {'status': 'healthy', 'timestamp': datetime.now()}
```

## CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Deploy to production
        run: |
          # Add deployment steps here
          echo "Deploying to production..."
```

### Environment-Specific Configurations

#### Development

```bash
# .env.development
STREAMLIT_SERVER_PORT=8501
DEBUG=true
LOG_LEVEL=DEBUG
```

#### Staging

```bash
# .env.staging
STREAMLIT_SERVER_PORT=8501
DEBUG=false
LOG_LEVEL=INFO
```

#### Production

```bash
# .env.production
STREAMLIT_SERVER_PORT=8501
DEBUG=false
LOG_LEVEL=WARNING
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**

```bash
# Find process using port
lsof -i :8501
# Kill process
kill -9 <PID>
```

2. **Memory Issues**

```bash
# Increase memory limit
export STREAMLIT_SERVER_MAX_UPLOAD_SIZE=200
```

3. **Model Loading Errors**

```bash
# Check model file exists
ls -la models/
# Verify model file integrity
python -c "import joblib; joblib.load('models/random_forest_model.pkl')"
```

### Performance Optimization

1. **Model Optimization**

   - Use smaller model files
   - Implement model caching
   - Use model quantization

2. **Data Optimization**

   - Use data compression
   - Implement data caching
   - Optimize data loading

3. **Application Optimization**
   - Use async operations
   - Implement connection pooling
   - Optimize database queries

## Backup and Recovery

### Data Backup

```bash
# Backup data files
tar -czf backup_$(date +%Y%m%d).tar.gz data/ models/

# Backup database (if using)
pg_dump database_name > backup.sql
```

### Disaster Recovery

1. Document recovery procedures
2. Test recovery processes regularly
3. Maintain backup schedules
4. Store backups in multiple locations

## Scaling

### Horizontal Scaling

1. Use load balancers
2. Deploy multiple instances
3. Use container orchestration (Kubernetes)

### Vertical Scaling

1. Increase server resources
2. Optimize application performance
3. Use caching strategies

## Security Checklist

- [ ] HTTPS enabled
- [ ] Authentication implemented
- [ ] Input validation
- [ ] Rate limiting
- [ ] Secrets management
- [ ] Regular security updates
- [ ] Access logging
- [ ] Backup encryption
- [ ] Network security
- [ ] Application monitoring
