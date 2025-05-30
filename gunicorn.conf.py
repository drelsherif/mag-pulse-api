# gunicorn.conf.py
import multiprocessing

# Server socket
bind = "0.0.0.0:10000"
backlog = 2048

# Worker processes
workers = 1  # Use single worker to avoid model loading issues
worker_class = "sync"
worker_connections = 1000
timeout = 120  # Increased timeout for LSTM predictions
keepalive = 2

# Memory and resource management
max_requests = 100  # Restart worker after 100 requests to prevent memory leaks
max_requests_jitter = 10
preload_app = True  # Load models once and share across workers

# Logging
accesslog = "-"
errorlog = "-"
loglevel = "info"
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = "magpulse_api"

# Memory optimization
worker_tmp_dir = "/dev/shm"  # Use shared memory for temporary files