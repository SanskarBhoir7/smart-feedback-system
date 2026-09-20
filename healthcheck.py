"""
Poll the deployed application's /health endpoint until it answers.

WHY THIS EXISTS
---------------
`docker run -d` returns as soon as the container is CREATED, not when Flask
is ready to serve. Without a wait, the Jenkins pipeline would race ahead to
the Selenium stage and fail against a server that simply had not finished
starting - an intermittent red build with no real defect behind it.

Written in Python rather than curl because Python is already a verified
dependency of this pipeline, whereas curl's presence on the Jenkins PATH is
not guaranteed. Uses only the standard library.

Exit codes, for the Jenkins `bat` step:

    0   the application is up (and, unless --require-model is passed,
        that is all that is checked)
    1   the application did not become healthy within the timeout

Usage:
    python healthcheck.py
    python healthcheck.py --url http://localhost:5000/health --retries 30
    python healthcheck.py --require-model     also insist a model is loaded
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request


def poll(url, retries, delay, require_model):
    """Try the endpoint until it answers healthily or the attempts run out."""
    print("Waiting for {}".format(url))
    print("  up to {} attempts, {}s apart".format(retries, delay))

    last_problem = "no attempt was made"

    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                body = response.read().decode("utf-8")
                payload = json.loads(body)

                if response.status != 200:
                    last_problem = "HTTP {}".format(response.status)
                elif payload.get("status") != "ok":
                    last_problem = "status={}".format(payload.get("status"))
                elif require_model and not payload.get("model_loaded"):
                    # The container is up but has no model. Worth failing on
                    # after a deployment, because serving without a model is
                    # not a successful deployment.
                    last_problem = "model not loaded: {}".format(
                        payload.get("error"))
                else:
                    print("")
                    print("  HEALTHY after {} attempt(s)".format(attempt))
                    print("    status        : {}".format(payload.get("status")))
                    print("    model_loaded  : {}".format(payload.get("model_loaded")))
                    print("    model_version : {}".format(payload.get("model_version")))
                    print("    database      : {}".format(payload.get("database")))
                    print("    total_feedback: {}".format(payload.get("total_feedback")))
                    return 0

        except urllib.error.URLError as error:
            last_problem = "connection refused ({})".format(error.reason)
        except (ValueError, OSError) as error:
            last_problem = str(error)

        print("  attempt {}/{}: not ready - {}".format(attempt, retries, last_problem))
        if attempt < retries:
            time.sleep(delay)

    print("")
    print("  FAILED: not healthy after {} attempts.".format(retries))
    print("  Last problem: {}".format(last_problem))
    print("")
    print("  Diagnose with:")
    print("    docker ps -a --filter name=smart_feedback_app")
    print("    docker logs smart_feedback_app")
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Wait for the deployed application to become healthy.")
    parser.add_argument("--url", default="http://localhost:5000/health",
                        help="health endpoint to poll")
    parser.add_argument("--retries", type=int, default=30,
                        help="how many attempts before giving up")
    parser.add_argument("--delay", type=float, default=2.0,
                        help="seconds between attempts")
    parser.add_argument("--require-model", action="store_true",
                        help="also require that a production model is loaded")
    args = parser.parse_args()

    return poll(args.url, args.retries, args.delay, args.require_model)


if __name__ == "__main__":
    sys.exit(main())
