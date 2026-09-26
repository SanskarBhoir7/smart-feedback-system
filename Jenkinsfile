// ===========================================================================
//  Smart Feedback System - CI/CD + Auto-Retraining Pipeline
// ===========================================================================
//
//  Declarative pipeline, following the CI/CD practical (PS4 - CI/CD Pipeline
//  / ML CI-CD Practical) but extended from "train, test, build" into the full
//  MLOps lifecycle:
//
//      COLLECT FEEDBACK -> CHECK THRESHOLD -> RETRAIN -> EVALUATE
//      -> QUALITY GATE -> STAGE MODEL -> DOCKERISE -> DEPLOY -> TEST
//
//  TWO THINGS MAKE THIS PIPELINE DIFFERENT FROM AN ORDINARY ONE
//  ------------------------------------------------------------
//
//  1. IT IS TRIGGERED BY DATA, NOT ONLY BY CODE.
//     Feedback is collected into SQLite, not into Git, so a Poll SCM trigger
//     would never notice it. The cron trigger below wakes the pipeline up
//     every 5 minutes to ask the database whether enough new feedback has
//     arrived to justify retraining.
//
//  2. MOST OF IT USUALLY DOES NOT RUN.
//     If the retraining threshold has not been reached, the pipeline does
//     NOT train, does NOT build an image and does NOT redeploy. It reports
//     "RETRAINING NOT REQUIRED" and finishes green. Rebuilding and
//     redeploying an unchanged model every five minutes would be pointless
//     churn and would restart the live service for no reason.
//
//  HOW THE STAGES COMMUNICATE
//  --------------------------
//  Each stage is a separate process, so Python cannot set a Jenkins variable
//  directly. retrain.py signals its decision through its EXIT CODE, which
//  `bat(returnStatus: true)` captures reliably on Windows:
//
//      python retrain.py --check    exit 0  -> retraining NOT required
//                                   exit 10 -> retraining IS required
//                                   exit 1  -> error
//
//      python retrain.py --gate     exit 0  -> candidate ACCEPTED
//                                   exit 1  -> candidate REJECTED, stop
//
//  Exit code 10 rather than 1 means a genuine Python crash can never be
//  mistaken for "retraining required".
//
//  WINDOWS
//  -------
//  Every step uses `bat`, not `sh`. The practicals show `sh` because they
//  assume a Linux agent; this Jenkins runs on Windows 11, where `sh` steps
//  fail. Experiment 3 uses "Execute Windows batch command" for exactly this
//  reason.
//
//  The account Jenkins runs as must be able to use python, git and the
//  Docker named pipe. On the development machine that account is
//  LocalSystem, and it can. Run jenkins_env_check.bat first to confirm it
//  on any other machine.
// ===========================================================================

pipeline {
    agent any

    triggers {
        // Data-driven trigger. H spreads the load so every job on the
        // controller does not fire on the same second.
        cron('H/5 * * * *')
    }

    options {
        buildDiscarder(logRotator(numToKeepStr: '20'))
        // 45 minutes: the FIRST build downloads CPU PyTorch into the Docker
        // image. Later builds reuse Docker's cached dependency layer.
        timeout(time: 45, unit: 'MINUTES')
        disableConcurrentBuilds()   // two pipelines must never retrain at once
    }

    environment {
        // The persistent store: database and every model version. Lives
        // outside the workspace so a workspace clean cannot destroy it.
        MLOPS_DATA_DIR = 'C:\\mlops-data'

        IMAGE_NAME     = 'smart-feedback'
        CONTAINER_NAME = 'smart_feedback_app'
        APP_PORT       = '5000'
        HEALTH_URL     = 'http://localhost:5000/health'
    }

    stages {

        // -------------------------------------------------------------------
        stage('Checkout') {
            steps {
                script {
                    // Works for a "Pipeline script from SCM" job. For a job
                    // with the script pasted inline there is no SCM to check
                    // out, and the existing workspace is used instead.
                    try {
                        checkout scm
                        echo 'Source checked out from SCM.'
                    } catch (ignored) {
                        echo 'No SCM configured - using the existing workspace.'
                    }
                }
                bat 'echo Workspace: %CD% && git --version && python --version && docker --version'
            }
        }

        // -------------------------------------------------------------------
        stage('Install Dependencies') {
            steps {
                bat 'python -m pip install --disable-pip-version-check -r requirements.txt'
            }
        }

        // -------------------------------------------------------------------
        stage('Bootstrap Encoder and Model') {
            steps {
                // Both steps are idempotent and do nothing when their
                // artifact already exists, so this stage only does real work
                // on a brand new machine.
                //
                // prepare_encoder.py puts the pinned, frozen sentence encoder
                // into C:\mlops-data\encoders once. It is the only step that
                // may need the internet, and only if the encoder is not
                // already in the local Hugging Face cache.
                bat 'python prepare_encoder.py'
                bat 'python train.py'
            }
        }

        // -------------------------------------------------------------------
        stage('Run Tests') {
            steps {
                // The first hard gate. If the unit and API tests fail the
                // pipeline stops here and nothing is built or deployed.
                bat 'python -m pytest tests -q'
            }
        }

        // -------------------------------------------------------------------
        stage('Check Retraining Requirement') {
            steps {
                script {
                    def code = bat(returnStatus: true,
                                   script: 'python retrain.py --check')

                    if (code == 10) {
                        env.RETRAIN_REQUIRED = 'true'
                        echo 'Threshold reached - the retraining stages will run.'
                    } else if (code == 0) {
                        env.RETRAIN_REQUIRED = 'false'
                        echo 'Below threshold - retraining, build and deploy will be skipped.'
                    } else {
                        error("retrain.py --check failed with exit code ${code}. " +
                              'Expected 0 (not required) or 10 (required).')
                    }

                    currentBuild.description = (env.RETRAIN_REQUIRED == 'true')
                        ? 'Retraining triggered'
                        : 'No retraining needed'
                }
            }
        }

        // ===================================================================
        //  Everything below runs ONLY when the threshold has been reached.
        // ===================================================================

        // -------------------------------------------------------------------
        stage('Train Candidate') {
            when { expression { env.RETRAIN_REQUIRED == 'true' } }
            steps {
                // Produces build/candidate_model.pkl. The candidate is not a
                // model version yet and is not registered anywhere.
                bat 'python retrain.py --train'
            }
        }

        // -------------------------------------------------------------------
        stage('Evaluate Candidate') {
            when { expression { env.RETRAIN_REQUIRED == 'true' } }
            steps {
                // Scores the candidate on the FIXED holdout set - the same 72
                // rows every model version is measured against - and writes
                // build/candidate_metrics.json for the quality gate.
                bat 'python evaluate.py --candidate --detailed'
            }
        }

        // -------------------------------------------------------------------
        stage('Quality Gate') {
            when { expression { env.RETRAIN_REQUIRED == 'true' } }
            steps {
                // THE SECOND HARD GATE, and the point of the whole project.
                //
                //   exit 0 -> accepted, promoted to production, carry on
                //   exit 1 -> REJECTED. The stage turns red, the pipeline
                //             stops, and the Docker build and deployment
                //             below never happen. The previous production
                //             model stays live and untouched.
                //
                // A plain `bat` is used deliberately: a non-zero exit fails
                // the stage, which is exactly the behaviour required.
                bat 'python retrain.py --gate'
            }
        }

        // -------------------------------------------------------------------
        stage('Stage Production Model') {
            when { expression { env.RETRAIN_REQUIRED == 'true' } }
            steps {
                // Copies the newly accepted model from the persistent store
                // (C:\mlops-data\models) into the workspace models/ folder,
                // AND the frozen sentence encoder it was trained against
                // (C:\mlops-data\encoders) into encoders/, so the Docker build
                // context contains both and the image bakes both in. The
                // encoder is only re-copied when its fingerprint changes.
                bat 'python stage_model.py --clean'
            }
        }

        // -------------------------------------------------------------------
        stage('Build Docker Image') {
            when { expression { env.RETRAIN_REQUIRED == 'true' } }
            steps {
                // Tagged with the build number so every deployment is
                // traceable back to the exact Jenkins run that produced it,
                // and :latest for convenience.
                bat 'docker build -t %IMAGE_NAME%:v%BUILD_NUMBER% -t %IMAGE_NAME%:latest .'
                bat 'docker images %IMAGE_NAME%'
            }
        }

        // -------------------------------------------------------------------
        stage('Deploy') {
            when { expression { env.RETRAIN_REQUIRED == 'true' } }
            steps {
                script {
                    // Remove any previous container first, or the port would
                    // already be taken. returnStatus swallows the non-zero
                    // exit when no such container exists - the Windows
                    // equivalent of the practical's `docker rm -f ... || true`.
                    bat(returnStatus: true, script: 'docker rm -f %CONTAINER_NAME%')
                }

                // -v mounts the persistent store, so the database survives
                // this container being destroyed and recreated.
                bat 'docker run -d -p %APP_PORT%:5000 -v "C:/mlops-data:/data" ' +
                    '--name %CONTAINER_NAME% %IMAGE_NAME%:v%BUILD_NUMBER%'

                bat 'docker ps --filter name=%CONTAINER_NAME%'

                // Wait for Flask to actually answer before declaring success.
                // --require-model also insists the baked model loaded, since
                // a container serving no model is not a successful deploy.
                // Up to 2 minutes: startup now imports PyTorch and loads the
                // sentence encoder before the first request is served.
                bat 'python healthcheck.py --url %HEALTH_URL% --retries 60 --delay 2 --require-model'
            }
        }

        // -------------------------------------------------------------------
        stage('Deployment Integration Tests') {
            when { expression { env.RETRAIN_REQUIRED == 'true' } }
            steps {
                // Real HTTP requests against the container that was just
                // deployed - not the Flask test client. They confirm the
                // DEPLOYED artifact works end to end: /health reports the new
                // model loaded with its encoder, /predict answers, the pages
                // render, and a submitted feedback row really lands in the
                // mounted database. Anything the tests submit is removed
                // again afterwards.
                withEnv(["BASE_URL=http://localhost:${env.APP_PORT}"]) {
                    bat 'python -m pytest integration_tests -q'
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    post {
        always {
            // Keep the evidence from every build: the retraining decision,
            // the candidate's metrics and its training summary - including
            // which feedback rows were excluded as inconsistent.
            archiveArtifacts artifacts: 'build/*.json',
                             allowEmptyArchive: true, fingerprint: true
        }
        success {
            script {
                if (env.RETRAIN_REQUIRED == 'true') {
                    echo '''
==========================================================
 PIPELINE SUCCESS - new model trained, approved, deployed
==========================================================
  The candidate passed the quality gate, was promoted to
  production, baked into a new image and deployed.
=========================================================='''
                } else {
                    echo '''
==========================================================
 PIPELINE SUCCESS - no retraining was required
==========================================================
  Not enough new feedback to justify retraining, so the
  training, build and deployment stages were skipped.
  The existing production model keeps serving.
=========================================================='''
                }
            }
        }
        failure {
            echo '''
==========================================================
 PIPELINE FAILED
==========================================================
  Check which stage went red:

   Bootstrap      the sentence encoder is missing or damaged.
                  Run python prepare_encoder.py once with
                  internet access.

   Run Tests      the test suite failed. Nothing was built.

   Quality Gate   THIS IS NOT A BUG. The candidate model was
                  rejected for being below the required
                  accuracy or worse than the current
                  production model. Deployment was blocked
                  ON PURPOSE and the previous model is still
                  serving. Open the Model page to see the
                  rejected version and the reason.

   Build/Deploy   a Docker problem. Confirm Docker Desktop
                  is running and that the Jenkins account can
                  reach the Docker engine:
                      docker ps
                  or run jenkins_env_check.bat.

   Deployment     the new container is up but misbehaves.
   Integration    The failing test names the endpoint.
   Tests
=========================================================='''
        }
    }
}
