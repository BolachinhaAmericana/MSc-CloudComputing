# main.py (multi-user version)

from flask import Flask, request, render_template_string, jsonify, session
import os
import logging
import threadin
import subprocess
import json
import tempfile
import time
import uuid
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-change-this-in-production')

# Configure logging for Flask app
log_dir = "/app/logs"
os.makedirs(log_dir, exist_ok=True)
flask_log_path = os.path.join(log_dir, 'flask_app_log.txt')
logging.basicConfig(filename=flask_log_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

print("Logging configured...")

# Global job tracking - stores all active jobs
active_jobs = {}
job_lock = threading.Lock()

# Simple HTML form with auto-refresh
HTML_FORM = """
<!DOCTYPE html>
<html>
<head>
    <title>Spark Pipeline Runner</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f4f4f4; color: #333; }
        .container { background-color: #fff; padding: 20px; border-radius: 8px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
        h1 { color: #333; }
        .session-info { background-color: #e7f3ff; padding: 10px; border-radius: 4px; margin-bottom: 20px; font-size: 14px; }
        label { display: block; margin-top: 10px; }
        input[type="number"], input[type="submit"] {
            width: calc(100% - 22px); padding: 10px; margin-top: 5px; border-radius: 4px; border: 1px solid #ddd;
        }
        input[type="submit"] { background-color: #5cb85c; color: white; cursor: pointer; font-size: 16px; }
        input[type="submit"]:hover { background-color: #4cae4c; }
        input[type="submit"]:disabled { background-color: #ccc; cursor: not-allowed; }
        .message { margin-top: 20px; padding: 10px; border-radius: 4px; }
        .success { background-color: #dff0d8; color: #3c763d; border: 1px solid #d6e9c6; }
        .error { background-color: #f2dede; color: #a94442; border: 1px solid #ebccd1; }
        .running { background-color: #d9edf7; color: #31708f; border: 1px solid #bce8f1; }
        .progress-container { 
            margin-top: 10px; 
            background-color: #f0f0f0; 
            border-radius: 4px; 
            overflow: hidden;
        }
        .progress-bar { 
            height: 20px; 
            background-color: #5cb85c; 
            transition: width 0.3s ease; 
            text-align: center; 
            line-height: 20px; 
            color: white; 
            font-size: 12px;
        }
        #live-status { 
            margin-top: 10px; 
            padding: 10px; 
            background-color: #f9f9f9; 
            border-radius: 4px; 
            border-left: 4px solid #5cb85c;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Run Spark X-Ray Analysis Pipeline</h1>
        
        <div class="session-info">
            <strong>Session ID:</strong> {{ session_id }}<br>
            <strong>Current Time:</strong> {{ current_time }}<br>
            <strong>Active Jobs:</strong> {{ active_jobs_count }}
        </div>
        
        {% if message %}
            <div class="message {{ message_class }}">
                <strong>{{ status|capitalize }}:</strong> {{ message }}
            </div>
        {% endif %}
        
        <!-- Live status updates will appear here -->
        <div id="live-status" style="display: none;">
            <div id="status-message">Checking status...</div>
            <div class="progress-container" id="progress-container" style="display: none;">
                <div class="progress-bar" id="progress-bar" style="width: 0%;">0%</div>
            </div>
            <div id="detailed-info"></div>
        </div>
        
        <form method="POST" action="/run_pipeline">
            <label for="max_images">Max Images (1-100):</label>
            <input type="number" id="max_images" name="max_images" min="1" max="100" value="{{ max_images_val or 10 }}" required>
           
            <label for="batch_size">Batch Size (2-20):</label>
            <input type="number" id="batch_size" name="batch_size" min="2" max="20" value="{{ batch_size_val or 5 }}" required>
           
            <input type="submit" value="Run Pipeline" id="submit-btn">
        </form>
    </div>

    <script>
        let statusInterval;
        
        function updateStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    const liveStatus = document.getElementById('live-status');
                    const statusMessage = document.getElementById('status-message');
                    const progressContainer = document.getElementById('progress-container');
                    const progressBar = document.getElementById('progress-bar');
                    const detailedInfo = document.getElementById('detailed-info');
                    const submitBtn = document.getElementById('submit-btn');
                    
                    // Show live status section
                    liveStatus.style.display = 'block';
                    
                    // Update status message
                    statusMessage.textContent = data.message || 'No status available';
                    
                    // Update progress bar if progress info is available
                    if (data.progress !== undefined) {
                        progressContainer.style.display = 'block';
                        const progressPercent = Math.round(data.progress);
                        progressBar.style.width = progressPercent + '%';
                        progressBar.textContent = progressPercent + '%';
                    }
                    
                    // Update detailed info
                    if (data.processed !== undefined && data.total !== undefined) {
                        detailedInfo.innerHTML = `
                            <small>Job ID: ${data.job_id || 'N/A'}</small><br>
                            <small>Processed: ${data.processed}/${data.total} images</small><br>
                            <small>Current batch: ${data.current_batch || 'N/A'}</small><br>
                            <small>Last updated: ${new Date().toLocaleTimeString()}</small>
                        `;
                    }
                    
                    // Handle different statuses
                    if (data.status === 'running') {
                        submitBtn.disabled = true;
                        submitBtn.value = 'Pipeline Running...';
                    } else if (data.status === 'completed' || data.status === 'error' || data.status === 'idle') {
                        submitBtn.disabled = false;
                        submitBtn.value = 'Run Pipeline';
                        if (data.status === 'completed' || data.status === 'error') {
                            clearInterval(statusInterval); // Stop polling when done
                        }
                    }
                })
                .catch(error => {
                    console.log('Status check failed:', error);
                    document.getElementById('status-message').textContent = 'Status check failed';
                });
        }
        
        // Start polling when page loads
        window.onload = function() {
            updateStatus();
            statusInterval = setInterval(updateStatus, 2000);
        };
        
        // Stop polling when page is about to unload
        window.onbeforeunload = function() {
            if (statusInterval) {
                clearInterval(statusInterval);
            }
        };
    </script>
</body>
</html>
"""

print("HTML template initialized...")

def get_session_job_status():
    """Get job status for current session"""
    if 'job_id' not in session:
        return {"status": "idle", "message": "No job run yet."}
    
    job_id = session['job_id']
    with job_lock:
        return active_jobs.get(job_id, {"status": "idle", "message": "Job not found."})

def update_session_job_status(updates):
    """Update job status for current session"""
    if 'job_id' not in session:
        return
    
    job_id = session['job_id']
    with job_lock:
        if job_id in active_jobs:
            active_jobs[job_id].update(updates)

def cleanup_completed_jobs():
    """Remove completed jobs older than 1 hour"""
    current_time = time.time()
    with job_lock:
        jobs_to_remove = []
        for job_id, job_data in active_jobs.items():
            if (job_data.get('status') in ['completed', 'error'] and 
                current_time - job_data.get('last_update', current_time) > 3600):
                jobs_to_remove.append(job_id)
        
        for job_id in jobs_to_remove:
            del active_jobs[job_id]

def run_spark_job_async(job_id, max_images, batch_size):
    """Run Spark job asynchronously for a specific job ID"""
    initial_status = {
        "status": "running", 
        "message": f"Pipeline started with max_images={max_images}, batch_size={batch_size}.",
        "progress": 0,
        "processed": 0,
        "total": max_images,
        "current_batch": 1,
        "job_id": job_id,
        "start_time": datetime.now().isoformat(),
        "last_update": time.time()
    }
    
    with job_lock:
        active_jobs[job_id] = initial_status
    
    app.logger.info(f"Starting Spark job {job_id} with max_images={max_images}, batch_size={batch_size}")
    
    # Start a progress simulator thread
    def simulate_progress():
        num_batches = (max_images + batch_size - 1) // batch_size
        estimated_time_per_batch = 5  # seconds
        
        for batch_num in range(1, num_batches + 1):
            with job_lock:
                if active_jobs.get(job_id, {}).get("status") != "running":
                    break
            
            time.sleep(estimated_time_per_batch)
            
            processed = min(batch_num * batch_size, max_images)
            progress = (processed / max_images) * 100
            
            updates = {
                "current_batch": batch_num,
                "processed": processed,
                "progress": progress,
                "message": f"Processing batch {batch_num} of {num_batches} ({processed}/{max_images} images)",
                "last_update": time.time()
            }
            
            with job_lock:
                if job_id in active_jobs:
                    active_jobs[job_id].update(updates)
    
    # Start progress simulation
    progress_thread = threading.Thread(target=simulate_progress)
    progress_thread.daemon = True
    progress_thread.start()
    
    try:
        # Run spark-submit with the standalone script
        spark_submit_cmd = [
            'spark-submit',
            '--jars', '/opt/spark/jars/gcs-connector-hadoop3-latest.jar',
            'spark_runner.py',
            '--max_images', str(max_images),
            '--batch_size', str(batch_size),
            '--job_id', job_id
        ]
        
        app.logger.info(f"Running command for job {job_id}: {' '.join(spark_submit_cmd)}")
        
        # Run the command with streaming output
        process = subprocess.Popen(
            spark_submit_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Read output line by line
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line:
                app.logger.info(f"Spark {job_id}: {line}")
            
            # Parse progress from Spark output
            updates = {"last_update": time.time()}
            if "PROGRESS:" in line:
                updates["message"] = line.replace("PROGRESS:", "").strip()
            elif "Starting pipeline" in line:
                updates["message"] = "Initializing Spark pipeline..."
            elif "Pipeline completed successfully" in line:
                updates.update({
                    "progress": 100,
                    "processed": max_images,
                    "message": "Pipeline completed successfully!"
                })
            
            with job_lock:
                if job_id in active_jobs:
                    active_jobs[job_id].update(updates)
        
        # Wait for process to complete
        process.wait()
        
        final_status = {
            "last_update": time.time(),
            "end_time": datetime.now().isoformat()
        }
        
        if process.returncode == 0:
            app.logger.info(f"Spark job {job_id} completed successfully")
            final_status.update({
                "status": "completed", 
                "message": f"Pipeline completed successfully! Processed {max_images} images in batches of {batch_size}.",
                "progress": 100,
                "processed": max_images,
                "current_batch": (max_images + batch_size - 1) // batch_size
            })
        else:
            error_msg = f"Spark job failed with return code {process.returncode}"
            app.logger.error(f"Job {job_id}: {error_msg}")
            final_status.update({
                "status": "error", 
                "message": error_msg
            })
        
        with job_lock:
            if job_id in active_jobs:
                active_jobs[job_id].update(final_status)
            
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        app.logger.error(f"Job {job_id}: {error_msg}", exc_info=True)
        
        with job_lock:
            if job_id in active_jobs:
                active_jobs[job_id].update({
                    "status": "error", 
                    "message": error_msg,
                    "last_update": time.time(),
                    "end_time": datetime.now().isoformat()
                })

print("Flask routes configured...")

@app.route('/', methods=['GET'])
def index():
    # Ensure session has an ID
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())[:8]
    
    # Cleanup old jobs
    cleanup_completed_jobs()
    
    job_status = get_session_job_status()
    
    # Determine message class for styling
    message_class = 'success' if job_status.get('status') == 'completed' else \
                   'error' if job_status.get('status') == 'error' else \
                   'running' if job_status.get('status') == 'running' else ''
    
    app.logger.info(f"Index page requested for session {session['session_id']}")
    
    return render_template_string(
        HTML_FORM, 
        status=job_status.get('status'), 
        message=job_status.get('message'),
        message_class=message_class,
        session_id=session['session_id'],
        current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        active_jobs_count=len(active_jobs)
    )

@app.route('/run_pipeline', methods=['POST'])
def run_pipeline_endpoint():
    # Ensure session has an ID
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())[:8]
    
    # Check if user already has a running job
    current_job_status = get_session_job_status()
    if current_job_status.get("status") == "running":
        app.logger.warning(f"Session {session['session_id']} attempted to run pipeline while another is running.")
        return render_template_string(
            HTML_FORM, 
            status="error", 
            message="You already have a pipeline running. Please wait for it to complete.",
            message_class="error",
            max_images_val=request.form.get('max_images'), 
            batch_size_val=request.form.get('batch_size'),
            session_id=session['session_id'],
            current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            active_jobs_count=len(active_jobs)
        )

    try:
        max_images = int(request.form['max_images'])
        batch_size = int(request.form['batch_size'])
        app.logger.info(f"Session {session['session_id']} requested pipeline: max_images={max_images}, batch_size={batch_size}")

        # Validate parameters
        if not (0 < max_images <= 100):
            app.logger.error(f"Session {session['session_id']}: max_images ({max_images}) out of range.")
            return render_template_string(
                HTML_FORM, 
                status="error", 
                message="Max Images must be between 1 and 100.",
                message_class="error",
                max_images_val=max_images, 
                batch_size_val=batch_size,
                session_id=session['session_id'],
                current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                active_jobs_count=len(active_jobs)
            ), 400
       
        if not (1 < batch_size <= 20):
            app.logger.error(f"Session {session['session_id']}: batch_size ({batch_size}) out of range.")
            return render_template_string(
                HTML_FORM, 
                status="error", 
                message="Batch Size must be between 2 and 20.",
                message_class="error",
                max_images_val=max_images, 
                batch_size_val=batch_size,
                session_id=session['session_id'],
                current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                active_jobs_count=len(active_jobs)
            ), 400
       
        # Create new job ID and store in session
        job_id = f"{session['session_id']}-{uuid.uuid4().hex[:8]}"
        session['job_id'] = job_id
        
        # Run the Spark job in a separate thread
        thread = threading.Thread(target=run_spark_job_async, args=(job_id, max_images, batch_size))
        thread.daemon = True
        thread.start()
       
        initial_message = f"Pipeline run initiated with max_images={max_images}, batch_size={batch_size}. Job ID: {job_id[:12]}..."
        app.logger.info(f"Session {session['session_id']}: {initial_message}")
        
        return render_template_string(
            HTML_FORM, 
            status="running", 
            message=initial_message,
            message_class="running",
            max_images_val=max_images, 
            batch_size_val=batch_size,
            session_id=session['session_id'],
            current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            active_jobs_count=len(active_jobs)
        )

    except ValueError:
        app.logger.error(f"Session {session['session_id']}: Invalid input for parameters.")
        return render_template_string(
            HTML_FORM, 
            status="error", 
            message="Invalid input. Please enter numbers for parameters.",
            message_class="error",
            max_images_val=request.form.get('max_images'), 
            batch_size_val=request.form.get('batch_size'),
            session_id=session['session_id'],
            current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            active_jobs_count=len(active_jobs)
        ), 400
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        app.logger.error(f"Session {session['session_id']}: {error_msg}", exc_info=True)
        return render_template_string(
            HTML_FORM, 
            status="error", 
            message=error_msg,
            message_class="error",
            max_images_val=request.form.get('max_images'), 
            batch_size_val=request.form.get('batch_size'),
            session_id=session['session_id'],
            current_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            active_jobs_count=len(active_jobs)
        ), 500

@app.route('/status', methods=['GET'])
def get_status():
    """Endpoint to check the status of the current session's job."""
    # Ensure session has an ID
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())[:8]
    
    job_status = get_session_job_status()
    app.logger.info(f"Status requested for session {session['session_id']}: {job_status.get('status', 'unknown')}")
    return jsonify(job_status)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for startup probes and monitoring."""
    try:
        cleanup_completed_jobs()
        
        health_status = {
            "status": "healthy",
            "timestamp": time.time(),
            "service": "spark-pipeline-runner",
            "version": "1.0.0",
            "active_jobs": len(active_jobs),
            "sessions_supported": True
        }
        
        app.logger.info("Health check requested - service is healthy")
        return jsonify(health_status), 200
        
    except Exception as e:
        error_status = {
            "status": "unhealthy",
            "timestamp": time.time(),
            "service": "spark-pipeline-runner",
            "error": str(e)
        }
        app.logger.error(f"Health check failed: {str(e)}")
        return jsonify(error_status), 503

@app.route('/api/jobs', methods=['GET'])
def list_jobs():
    """Admin endpoint to list all active jobs."""
    with job_lock:
        jobs_summary = {
            job_id: {
                "status": job_data.get("status"),
                "progress": job_data.get("progress", 0),
                "start_time": job_data.get("start_time"),
                "last_update": job_data.get("last_update")
            }
            for job_id, job_data in active_jobs.items()
        }
    
    return jsonify({
        "total_jobs": len(jobs_summary),
        "jobs": jobs_summary
    })

if __name__ == '__main__':
    print("Starting Flask server with multi-user support...")
    app.run(host='0.0.0.0', port=5000, debug=False)
    print("Flask server stopped.")
