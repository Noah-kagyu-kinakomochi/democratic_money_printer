import os
import json
import subprocess
import sys
import time

# Configuration
JOB_NAME = "MoneyPrinter_Serverless_Train"
Config_File = "jobs/job_config.json"
Workspace_Dir = "/Shared/MoneyPrinter"
Remote_Script_Path = f"{Workspace_Dir}/jobs/train_remote.py"
Local_Script_Path = "jobs/train_remote.py"

def run_cmd(args, check=True):
    """Run command and return stdout string. content robust to warnings."""
    print(f"Running: {' '.join(args)}")
    result = subprocess.run(args, capture_output=True, text=True)
    
    # Print stderr for debugging (warnings/errors)
    if result.stderr:
        print(f"STDERR: {result.stderr}")
        
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        if check:
            raise RuntimeError(f"Command failed with code {result.returncode}")
    
    return result.stdout

def parse_json_from_output(output):
    """Find the first '{' and parse as JSON."""
    try:
        start = output.find('{')
        if start == -1:
            return None
        json_str = output[start:]
        return json.loads(json_str)
    except json.JSONDecodeError:
        return None

def main():
    print(f"🚀 Deploying {JOB_NAME} to Databricks Workspace...")
    
    # 1. Upload Script to Workspace
    # Ensure directory exists
    print(f"Creating directory: {Workspace_Dir}/jobs")
    run_cmd(["databricks", "workspace", "mkdirs", f"{Workspace_Dir}/jobs"], check=False)
    
    print(f"Uploading {Local_Script_Path} to {Remote_Script_Path}...")
    run_cmd(["databricks", "workspace", "import", Local_Script_Path, Remote_Script_Path, "--language", "PYTHON", "--overwrite"])
    
    # 2. Generate Serverless Config (Workspace Source)
    # We create a new config on the fly to avoid messing up the git-based one if user wants to revert
    serverless_config = {
        "name": JOB_NAME,
        "email_notifications": { "no_alert_for_skipped_runs": False },
        "timeout_seconds": 0,
        "max_concurrent_runs": 1,
        "tasks": [
            {
                "task_key": "train_model",
                "run_if": "ALL_SUCCESS",
                "spark_python_task": {
                    "python_file": Remote_Script_Path,
                    "source": "WORKSPACE",
                    "parameters": ["--epochs", "50", "--batch_size", "32"]
                },
                "environment_key": "Default",
                "timeout_seconds": 0
            }
        ],
        "environments": [
            {
                "environment_key": "Default",
                "spec": {
                    "dependencies": [
                        "mlflow", "torch", "scikit-learn", "pandas", "numpy", "joblib"
                    ]
                }
            }
        ],
        "format": "MULTI_TASK"
    }
    
    temp_config_file = "serverless_job_config.json"
    with open(temp_config_file, "w") as f:
        json.dump(serverless_config, f, indent=2)
        
    print(f"Generated config: {temp_config_file}")

    # 3. List Jobs
    print(f"Checking for existing job...")
    output = run_cmd(["databricks", "jobs", "list", "--output", "json"])
    data = parse_json_from_output(output)
    
    existing_job_id = None
    if data and "jobs" in data:
        for job in data["jobs"]:
            if job.get("settings", {}).get("name") == JOB_NAME:
                existing_job_id = job["job_id"]
                break
    
    job_id = None
    
    # 4. Create or Reset
    if existing_job_id:
        print(f"Found existing Job ID: {existing_job_id}. Resetting...")
        run_cmd(["databricks", "jobs", "reset", "--job-id", str(existing_job_id), "--json-file", temp_config_file])
        job_id = existing_job_id
    else:
        print("Job not found. Creating new...")
        create_output = run_cmd(["databricks", "jobs", "create", "--json-file", temp_config_file])
        create_data = parse_json_from_output(create_output)
        if not create_data or "job_id" not in create_data:
             # Fallback parsing
             if "job_id" in create_output:
                 import re
                 m = re.search(r'"job_id":\s*(\d+)', create_output)
                 if m: job_id = m.group(1)
             
             if not job_id:
                raise RuntimeError("Failed to extract job_id from create output")
        else:
            job_id = create_data["job_id"]
        print(f"Created Job ID: {job_id}")

    # 5. Run
    print(f"Triggering run for Job ID: {job_id}...")
    run_output = run_cmd(["databricks", "jobs", "run-now", "--job-id", str(job_id)])
    run_data = parse_json_from_output(run_output)
    
    run_id = run_data.get("run_id") if run_data else "UNKNOWN"
    host = os.environ.get("DATABRICKS_HOST", "")
    
    print(f"🚀 Run started! Run ID: {run_id}")
    print(f"🔗 Track run at: {host}/?o=0#job/{job_id}/run/{run_id}")

if __name__ == "__main__":
    main()
