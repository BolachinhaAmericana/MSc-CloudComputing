""" TESTAR ESTE SCRIPT. FAZER COM QUE USE O MAIN DO REPORT.PY"""


# main.py
from flask import Flask, request, render_template_string, jsonify
import report # Import the refactored report.py
import os
import logging
import threading # To run Spark job in background

app = Flask(__name__)

# Configure logging for Flask app
log_dir = "/app/logs" # Should match report.py if logs are shared, or use a different file
os.makedirs(log_dir, exist_ok=True)
flask_log_path = os.path.join(log_dir, 'flask_app_log.txt')
logging.basicConfig(filename=flask_log_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Simple HTML form
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
        <form method="POST" action="/run_pipeline">
            <label for="max_images">Max Images (1-101):</label>
            <input type="number" id="max_images" name="max_images" min="1" max="101" value="{{ max_images_val or 10 }}" required>
            
            <label for="batch_size">Batch Size (2-20):</label>
            <input type="number" id="batch_size" name="batch_size" min="2" max="20" value="{{ batch_size_val or 5 }}" required>
            
            <input type="submit" value="Run Pipeline">
        </form>
    </div>
</body>
</html>
"""

job_status = {"status": "idle", "message": "No job run yet."}

def run_spark_job_async(max_images, batch_size):
    global job_status
    job_status = {"status": "running", "message": f"Pipeline started with max_images={max_images}, batch_size={batch_size}."}
    app.logger.info(f"Calling report.run_pipeline_logic with max_images={max_images}, batch_size={batch_size}")
    try:
        result = report.run_pipeline_logic(max_images_param=max_images, batch_size_param=batch_size)
        app.logger.info(f"Pipeline execution result: {result}")
        job_status = result # Store the actual result
    except Exception as e:
        error_msg = f"An unexpected error occurred in the Spark job: {str(e)}"
        app.logger.error(error_msg, exc_info=True)
        job_status = {"status": "error", "message": error_msg}


@app.route('/', methods=['GET'])
def index():
    app.logger.info("Index page requested.")
    # Pass current job status to the template
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

        # Validate parameters (as per asserts in report.py)
        if not (0 < max_images <= 101):
            app.logger.error(f"Validation error: max_images ({max_images}) out of range (1-101).")
            return render_template_string(HTML_FORM, status="error", message="Max Images must be between 1 and 101.",
                                          max_images_val=max_images, batch_size_val=batch_size), 400
        
        if not (1 < batch_size <= 20): # Your assert was `1 < batch_size <= 20`
            app.logger.error(f"Validation error: batch_size ({batch_size}) out of range (2-20).")
            return render_template_string(HTML_FORM, status="error", message="Batch Size must be between 2 and 20.",
                                          max_images_val=max_images, batch_size_val=batch_size), 400
        
        # Run the Spark job in a separate thread to avoid blocking the HTTP request
        thread = threading.Thread(target=run_spark_job_async, args=(max_images, batch_size))
        thread.start()
        
        # Immediately respond to the user
        initial_message = f"Pipeline run initiated with max_images={max_images}, batch_size={batch_size}. Check status on the main page or /status."
        app.logger.info(initial_message)
        # Redirect to main page which will show the "running" status
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
    # Make sure to run on 0.0.0.0 to be accessible within the Docker container from outside
    app.run(host='0.0.0.0', port=5000, debug=True) # debug=False for production
