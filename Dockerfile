# Smart Feedback System - container image
#
# This follows the structure taught in the Docker practical (Experiment 5)
# and the ML deployment practical (Experiment 9): a slim Python base, a
# working directory, requirements installed first, the application copied in,
# the port exposed, and the Flask app as the start command.
#
# The image is a SELF-CONTAINED DEPLOYMENT ARTIFACT. It holds the application
# code AND the exact model version that was approved by the quality gate.
# Runtime state - the SQLite database - deliberately stays outside, in a
# mounted volume, so recreating the container never destroys collected
# feedback.

# 1. Use an official Python runtime as a parent image.
#
#    DIFFERENCE FROM THE PRACTICAL: the practicals use python:3.9-slim.
#    This project pins 3.12 to match the machine that trains the model.
#    model_v1.pkl is produced by joblib on Python 3.12 with
#    scikit-learn 1.5.1; loading that pickle under a different Python or
#    scikit-learn version raises InconsistentVersionWarning and can fail
#    outright. The base image and requirements.txt are pinned together so the
#    artifact always loads.
FROM python:3.12-slim

# 2. Set the working directory in the container.
WORKDIR /app

# 3. Copy requirements first, then install.
#    Doing this before copying the code means Docker can reuse the cached
#    dependency layer whenever only application code changes, which makes
#    Jenkins rebuilds much faster.
COPY requirements.txt .

# 4. Install the pinned dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the application code AND the staged production model.
#    .dockerignore keeps the database, the build/ scratch space and the
#    caches out; models/model_vN.pkl is intentionally included.
COPY . .

# 6. Point the application at the mounted data volume.
#
#    ADDITION BEYOND THE PRACTICAL: the practicals keep model.pkl beside the
#    code and have no database. This project separates the two:
#
#      /app/models   baked into the image  - the approved production model
#      /data         mounted from the host - the SQLite database
#
#    config.py reads MLOPS_DATA_DIR, so setting it here is all that is needed
#    to redirect the database out of the container's writable layer.
ENV MLOPS_DATA_DIR=/data

#    Without this, Python buffers stdout and Flask's startup banner and log
#    lines do not appear in `docker logs` until the buffer flushes, which
#    makes a failed Jenkins deployment much harder to diagnose.
ENV PYTHONUNBUFFERED=1

#    Create the mount point so the container still starts (with an empty
#    database it creates itself) even when no volume is supplied.
RUN mkdir -p /data/models

# 7. Make port 5000 available outside the container.
EXPOSE 5000

# 8. Run the Flask application when the container launches.
#    app.py binds to 0.0.0.0, which is what makes the service reachable
#    through the published port.
CMD ["python", "app.py"]
