import os
import json
import subprocess
import sys

def run_cmd(args):
    """Run command and return stdout string. content robust to warnings."""
    print(f"Running: {' '.join(args)}")
    result = subprocess.run(args, capture_output=True, text=True)
    
    # Print stderr for debugging (warnings/errors)
    if result.stderr:
        print(f"STDERR: {result.stderr}")
        
    if result.returncode != 0:
        print(f"STDOUT: {result.stdout}")
        raise RuntimeError(f"Command failed with code {result.returncode}")

    return result.stdout

def parse_json_from_output(output):
    """Find the first '{' and parse as JSON."""
    try:
        start = output.find('{')
        if start == -1:
            return None
        # Naive but usually sufficient: parse from start
        # If there is trailing garbage, json.loads might fail or ignore it depending on strictness
        # For 'databricks jobs create', it creates a single dict.
        json_str = output[start:]
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Raw Output: {output}")
        raise

def main():
    job_name = "MoneyPrinter_Production_Train"
    config_file = "jobs/job_config.json"
    
    # 1. List Jobs
    print(f"Checking for job: {job_name}...")
    output = run_cmd(["databricks", "jobs", "list", "--output", "json"])
    
    data = parse_json_from_output(output)
    existing_job_id = None
    
    if data and "jobs" in data:
        for job in data["jobs"]:
            if job.get("settings", {}).get("name") == job_name:
                existing_job_id = job["job_id"]
                break
    
    job_id = None
    
    # 2. Create or Reset
    if existing_job_id:
        print(f"Found existing Job ID: {existing_job_id}. Resetting...")
        run_cmd(["databricks", "jobs", "reset", "--job-id", str(existing_job_id), "--json-file", config_file])
        job_id = existing_job_id
    else:
        print("Job not found. Creating new...")
        create_output = run_cmd(["databricks", "jobs", "create", "--json-file", config_file])
        create_data = parse_json_from_output(create_output)
        if not create_data or "job_id" not in create_data:
            raise RuntimeError("Failed to extract job_id from create output")
        job_id = create_data["job_id"]
        print(f"Created Job ID: {job_id}")

    # 3. Run
    print(f"Triggering run for Job ID: {job_id}...")
    run_output = run_cmd(["databricks", "jobs", "run-now", "--job-id", str(job_id), "--output", "json"])
    run_data = parse_json_from_output(run_output)
    
    run_id = run_data.get("run_id") if run_data else "UNKNOWN"
    host = os.environ.get("DATABRICKS_HOST", "")
    
    print(f"🚀 Run started! Run ID: {run_id}")
    print(f"🔗 Track run at: {host}/?o=0#job/{job_id}/run/{run_id}")

if __name__ == "__main__":
    main()
