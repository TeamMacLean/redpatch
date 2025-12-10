"""
Simple Flask web application for Redpatch image processing.
Provides a form to upload leaf images, process them, and display results.
"""

import os
import tempfile
import uuid

from flask import (
    Flask,
    redirect,
    render_template_string,
    request,
    send_from_directory,
    url_for,
)

import redpatch as rp

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = tempfile.mkdtemp()
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max

# HTML Templates
INDEX_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Redpatch - Leaf Lesion Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        h1 { color: #2c5530; }
        .upload-form { background: #f5f5f5; padding: 20px; border-radius: 8px; }
        .upload-form input[type="file"] { margin: 10px 0; }
        .upload-form button { background: #2c5530; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
        .upload-form button:hover { background: #3d7a45; }
        .error { color: red; margin-top: 10px; }
        .processing { color: #666; font-style: italic; }
    </style>
</head>
<body>
    <h1>Redpatch - Leaf Lesion Analysis</h1>
    <div class="upload-form">
        <form method="POST" action="/upload" enctype="multipart/form-data" id="upload-form">
            <h2>Upload Leaf Image</h2>
            <p>Select an image of plant leaves to analyze for disease lesions.</p>
            <input type="file" name="image" id="image-input" accept="image/*" required>
            <br><br>
            <button type="submit" id="submit-btn">Process Image</button>
        </form>
        {% if error %}
        <p class="error">{{ error }}</p>
        {% endif %}
    </div>
</body>
</html>
"""

PROCESSING_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Processing - Redpatch</title>
    <meta http-equiv="refresh" content="1;url={{ redirect_url }}">
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; text-align: center; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #2c5530; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 20px auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <h1>Processing Image...</h1>
    <div class="spinner"></div>
    <p id="processing-status">Analyzing leaf image for lesions...</p>
</body>
</html>
"""

RESULTS_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Results - Redpatch</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 1000px; margin: 50px auto; padding: 20px; }
        h1 { color: #2c5530; }
        .results-container { display: flex; gap: 20px; flex-wrap: wrap; }
        .image-section { flex: 1; min-width: 300px; }
        .image-section img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
        .stats-section { background: #f5f5f5; padding: 20px; border-radius: 8px; margin-top: 20px; }
        .stats-section h3 { margin-top: 0; }
        .stat-item { margin: 10px 0; }
        .stat-label { font-weight: bold; color: #555; }
        .back-link { display: inline-block; margin-top: 20px; color: #2c5530; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #2c5530; color: white; }
        .legend { margin-top: 10px; }
        .legend-item { display: inline-block; margin-right: 15px; }
        .legend-color { display: inline-block; width: 15px; height: 15px; margin-right: 5px; vertical-align: middle; }
        .healthy-color { background: rgba(127, 191, 63, 0.7); }
        .lesion-color { background: rgba(243, 80, 21, 0.7); }
        .centre-color { background: rgba(248, 252, 17, 0.8); }
    </style>
</head>
<body>
    <h1>Analysis Results</h1>

    <div class="results-container">
        <div class="image-section">
            <h3>Original Image</h3>
            <img src="{{ original_url }}" alt="Original leaf image" id="original-image">
        </div>
        <div class="image-section">
            <h3>Annotated Result</h3>
            <img src="{{ annotated_url }}" alt="Annotated result" id="annotated-image">
            <div class="legend">
                <span class="legend-item"><span class="legend-color healthy-color"></span>Healthy</span>
                <span class="legend-item"><span class="legend-color lesion-color"></span>Lesion</span>
                <span class="legend-item"><span class="legend-color centre-color"></span>Lesion Centre</span>
            </div>
        </div>
    </div>

    <div class="stats-section" id="results-summary">
        <h3>Analysis Summary</h3>
        <div class="stat-item">
            <span class="stat-label">Healthy Regions:</span> {{ stats.healthy_count }}
        </div>
        <div class="stat-item">
            <span class="stat-label">Lesion Regions:</span> {{ stats.lesion_count }}
        </div>
        <div class="stat-item">
            <span class="stat-label">Lesion Centres:</span> {{ stats.centre_count }}
        </div>
        <div class="stat-item">
            <span class="stat-label">Total Healthy Area (pixels):</span> {{ stats.healthy_area }}
        </div>
        <div class="stat-item">
            <span class="stat-label">Total Lesion Area (pixels):</span> {{ stats.lesion_area }}
        </div>

        {% if stats.lesion_count > 0 %}
        <h3>Lesion Details</h3>
        <table id="lesion-table">
            <thead>
                <tr>
                    <th>Label</th>
                    <th>Area (pixels)</th>
                    <th>Passed Filter</th>
                </tr>
            </thead>
            <tbody>
                {% for lesion in lesions %}
                <tr>
                    <td>{{ lesion.label }}</td>
                    <td>{{ lesion.area }}</td>
                    <td>{{ 'Yes' if lesion.passed else 'No' }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
        {% endif %}
    </div>

    <a href="/" class="back-link">&larr; Analyze Another Image</a>
</body>
</html>
"""

ERROR_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Error - Redpatch</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; }
        .error-box { background: #ffe6e6; border: 1px solid #ff9999; padding: 20px; border-radius: 8px; }
        .error-box h2 { color: #cc0000; margin-top: 0; }
        .back-link { display: inline-block; margin-top: 20px; color: #2c5530; }
    </style>
</head>
<body>
    <div class="error-box">
        <h2>Processing Error</h2>
        <p id="error-message">{{ error }}</p>
    </div>
    <a href="/" class="back-link">&larr; Try Again</a>
</body>
</html>
"""


def get_default_filter_settings():
    """Create default filter settings for leaf analysis."""
    fs = rp.FilterSettings()
    # Default settings that work for many leaf images
    fs.add_setting("leaf_area", h=[0.15, 0.45], s=[0.1, 1.0], v=[0.1, 0.9])
    fs.add_setting("healthy_area", h=[0.2, 0.4], s=[0.2, 0.8], v=[0.2, 0.8])
    fs.add_setting("lesion_area", h=[0.0, 0.15], s=[0.1, 1.0], v=[0.2, 0.9])
    return fs


@app.route("/")
def index():
    """Display the upload form."""
    return render_template_string(INDEX_TEMPLATE)


@app.route("/upload", methods=["POST"])
def upload():
    """Handle image upload and redirect to processing."""
    if "image" not in request.files:
        return render_template_string(INDEX_TEMPLATE, error="No file selected")

    file = request.files["image"]
    if file.filename == "":
        return render_template_string(INDEX_TEMPLATE, error="No file selected")

    # Save uploaded file
    job_id = str(uuid.uuid4())
    job_folder = os.path.join(app.config["UPLOAD_FOLDER"], job_id)
    os.makedirs(job_folder, exist_ok=True)

    filename = "uploaded_image.jpg"
    filepath = os.path.join(job_folder, filename)
    file.save(filepath)

    # Redirect to process
    return redirect(url_for("process", job_id=job_id))


@app.route("/process/<job_id>")
def process(job_id):
    """Process the uploaded image."""
    job_folder = os.path.join(app.config["UPLOAD_FOLDER"], job_id)
    filepath = os.path.join(job_folder, "uploaded_image.jpg")

    if not os.path.exists(filepath):
        return render_template_string(ERROR_TEMPLATE, error="Image not found")

    try:
        # Get filter settings
        fs = get_default_filter_settings()

        # Process image
        sub_images = rp.get_sub_images(
            filepath,
            file_settings=fs,
            dest_folder=job_folder,
            min_lesion_area=10,
            max_lc_ratio=3,
            min_lc_size=1,
            lc_prop_across_parent=0.05,
        )

        if not sub_images:
            return render_template_string(
                ERROR_TEMPLATE,
                error="No leaf regions detected in the image. Please upload a clearer image with visible leaves.",
            )

        # Use first sub-image for results
        si = sub_images[0]
        si.write_sub_image()
        si.write_annotated_sub_image()

        # Calculate statistics
        stats = {
            "healthy_count": len(si.healthy_obj_props),
            "lesion_count": len(si.lesion_area_props),
            "centre_count": len(si.lesion_centre_props),
            "healthy_area": sum(h.area for h in si.healthy_obj_props),
            "lesion_area": sum(l.area for l in si.lesion_area_props),
        }

        lesions = [
            {"label": l.label, "area": l.area, "passed": l.passed}
            for l in si.lesion_area_props
        ]

        return render_template_string(
            RESULTS_TEMPLATE,
            original_url=url_for(
                "serve_image", job_id=job_id, filename=os.path.basename(si.imtag)
            ),
            annotated_url=url_for(
                "serve_image", job_id=job_id, filename=os.path.basename(si.annot_imtag)
            ),
            stats=stats,
            lesions=lesions,
        )

    except Exception as e:
        return render_template_string(ERROR_TEMPLATE, error=str(e))


@app.route("/images/<job_id>/<filename>")
def serve_image(job_id, filename):
    """Serve processed images."""
    job_folder = os.path.join(app.config["UPLOAD_FOLDER"], job_id)
    return send_from_directory(job_folder, filename)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
