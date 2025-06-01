# main.py (modified version)

from flask import Flask, request, render_template_string, jsonify
import os
import logging
import threading
import subprocess
import json
import tempfile
import threading
import time

app = Flask(__name__)

# Configure logging for Flask app
log_dir = "/app/logs"
os.makedirs(log_dir, exist_ok=True)
flask_log_path = os.path.join(log_dir, 'flask_app_log.txt')
logging.basicConfig(filename=flask_log_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

print("Logging configured...")

# Simple HTML form
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
        label { display: block; margin-top: 10px; }
        input[type="number"], input[type="submit"] {
            width: calc(100% - 22px); padding: 10px; margin-top: 5px; border-radius: 4px; border: 1px solid #ddd;
        }
        input[type="submit"] { background-color: #5cb85c; color: white; cursor: pointer; font-size: 16px; }
        input[type="submit"]:hover { background-color: #4cae4c; }
        .message { margin-top: 20px; padding: 10px; border-radius: 4px; }
        .success { background-color: #dff0d8; color: #3c763d; border: 1px solid #d6e9c6; }
        .error { background-color: #f2dede; color: #a94442; border: 1px solid #ebccd1; }
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
        
        {% if message %}
            <div class="message {{ 'success' if 'success' in status else 'error' }}">
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
            <label for="max_images">Max Images (1-101):</label>
            <input type="number" id="max_images" name="max_images" min="1" max="101" value="{{ max_images_val or 10 }}" required>
           
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
                            <small>Processed: ${data.processed}/${data.total} images</small><br>
                            <small>Current batch: ${data.current_batch || 'N/A'}</small><br>
                            <small>Last updated: ${new Date().toLocaleTimeString()}</small>
                        `;
                    }
                    
                    // Handle different statuses
                    if (data.status === 'running') {
                        submitBtn.disabled = true;
                        submitBtn.value = 'Pipeline Running...';
                    } else if (data.status === 'completed' || data.status === 'error') {
                        submitBtn.disabled = false;
                        submitBtn.value = 'Run Pipeline';
                        clearInterval(statusInterval); // Stop polling when done
                    }
                })
                .catch(error => {
                    console.log('Status check failed:', error);
                    document.getElementById('status-message').textContent = 'Status check failed';
                });
        }
        
        // Start polling when page loads
        window.onload = function() {
            // Check if there might be a running job
            updateStatus();
            
            // Poll every 2 seconds
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

job_status = {"status": "idle", "message": "No job run yet."}

print("HTML template and job status initialized...")



""" New pipeline  """

def run_spark_job_async(max_images, batch_size):
    global job_status
    job_status = {
        "status": "running", 
        "message": f"Pipeline started with max_images={max_images}, batch_size={batch_size}.",
        "progress": 0,
        "processed": 0,
        "total": max_images,
        "current_batch": 1
    }
    app.logger.info(f"Starting Spark job with max_images={max_images}, batch_size={batch_size}")
    
    # Start a progress simulator thread
    def simulate_progress():
        num_batches = (max_images + batch_size - 1) // batch_size
        estimated_time_per_batch = 5  # seconds - adjust based on your actual processing time
        
        for batch_num in range(1, num_batches + 1):
            if job_status.get("status") != "running":
                break
                
            time.sleep(estimated_time_per_batch)
            
            processed = min(batch_num * batch_size, max_images)
            progress = (processed / max_images) * 100
            
            job_status.update({
                "current_batch": batch_num,
                "processed": processed,
                "progress": progress,
                "message": f"Processing batch {batch_num} of {num_batches} ({processed}/{max_images} images)"
            })
    
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
            '--batch_size', str(batch_size)
        ]
        
        app.logger.info(f"Running command: {' '.join(spark_submit_cmd)}")
        
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
            if line:  # Only log non-empty lines
                app.logger.info(f"Spark: {line}")
            
            # Parse any actual progress from Spark output
            if "PROGRESS:" in line:
                job_status["message"] = line.replace("PROGRESS:", "").strip()
            elif "Starting pipeline" in line:
                job_status["message"] = "Initializing Spark pipeline..."
            elif "Pipeline completed successfully" in line:
                job_status.update({
                    "progress": 100,
                    "processed": max_images,
                    "message": "Pipeline completed successfully!"
                })
        
        # Wait for process to complete
        process.wait()
        
        if process.returncode == 0:
            app.logger.info(f"Spark job completed successfully")
            job_status = {
                "status": "success", 
                "message": f"Pipeline completed successfully! Processed {max_images} images in batches of {batch_size}.",
                "progress": 100,
                "processed": max_images,
                "total": max_images,
                "current_batch": (max_images + batch_size - 1) // batch_size
            }
        else:
            error_msg = f"Spark job failed with return code {process.returncode}"
            app.logger.error(error_msg)
            job_status = {"status": "error", "message": error_msg}
            
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        app.logger.error(error_msg, exc_info=True)
        job_status = {"status": "error", "message": error_msg}

print("Flask routes configured...")

@app.route('/', methods=['GET'])
def index():
    app.logger.info("Index page requested.")
    return render_template_string(HTML_FORM, status=job_status.get('status'), message=job_status.get('message'))

@app.route('/run_pipeline', methods=['POST'])
def run_pipeline_endpoint():
    global job_status
    if job_status.get("status") == "running":
        app.logger.warning("Attempted to run pipeline while another is already running.")
        return render_template_string(HTML_FORM, status="error", message="A pipeline is already running. Please wait for it to complete.",
                                      max_images_val=request.form.get('max_images'), batch_size_val=request.form.get('batch_size'))

    try:
        max_images = int(request.form['max_images'])
        batch_size = int(request.form['batch_size'])
        app.logger.info(f"Received request to run pipeline: max_images={max_images}, batch_size={batch_size}")

        # Validate parameters
        if not (0 < max_images <= 101):
            app.logger.error(f"Validation error: max_images ({max_images}) out of range (1-101).")
            return render_template_string(HTML_FORM, status="error", message="Max Images must be between 1 and 101.",
                                          max_images_val=max_images, batch_size_val=batch_size), 400
       
        if not (1 < batch_size <= 20):
            app.logger.error(f"Validation error: batch_size ({batch_size}) out of range (2-20).")
            return render_template_string(HTML_FORM, status="error", message="Batch Size must be between 2 and 20.",
                                          max_images_val=max_images, batch_size_val=batch_size), 400
       
        # Run the Spark job in a separate thread
        thread = threading.Thread(target=run_spark_job_async, args=(max_images, batch_size))
        thread.start()
       
        initial_message = f"Pipeline run initiated with max_images={max_images}, batch_size={batch_size}. Check status on the main page or /status."
        app.logger.info(initial_message)
        return render_template_string(HTML_FORM, status="running", message=initial_message,
                                      max_images_val=max_images, batch_size_val=batch_size)

    except ValueError:
        app.logger.error("ValueError: Invalid input for max_images or batch_size.")
        return render_template_string(HTML_FORM, status="error", message="Invalid input. Please enter numbers for parameters.",
                                      max_images_val=request.form.get('max_images'), batch_size_val=request.form.get('batch_size')), 400
    except Exception as e:
        error_msg = f"An unexpected error occurred: {str(e)}"
        app.logger.error(error_msg, exc_info=True)
        return render_template_string(HTML_FORM, status="error", message=error_msg,
                                      max_images_val=request.form.get('max_images'), batch_size_val=request.form.get('batch_size')), 500





@app.route('/status', methods=['GET'])
def get_status():
    """Endpoint to check the status of the currently running or last run job."""
    app.logger.info(f"Status requested: {job_status}")
    return jsonify(job_status)

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000, debug=False)
    print("Flask server stopped.")

""" Test Flask """
# app = Flask(__name__)

# @app.route('/')
# def hello():
#     return '<h1>Hello World! Flask is working!</h1>'

# if __name__ == '__main__':
#     print("Starting Flask test app...")
#     app.run(host='0.0.0.0', port=5000, debug=True)
#     print("Flask app should be running now.")
