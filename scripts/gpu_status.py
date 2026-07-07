#!/usr/bin/env python3
import subprocess
import re
import os
import pwd
import json
import sys

def get_process_cmdline(pid):
    try:
        with open(f"/proc/{pid}/cmdline", "r") as f:
            cmd = f.read().replace("\x00", " ").strip()
            return cmd if cmd else "[no cmdline]"
    except Exception:
        return "[unknown]"

def get_process_owner(pid):
    try:
        stat = os.stat(f"/proc/{pid}")
        uid = stat.st_uid
        return pwd.getpwuid(uid).pw_name
    except Exception:
        return "unknown"

def get_container_info(pid):
    # Read cgroups to see if containerized
    try:
        with open(f"/proc/{pid}/cgroup", "r") as f:
            content = f.read()
    except Exception:
        return None
    
    # Try to find a docker container ID (64 character hex string)
    match = re.search(r'/docker/([a-f0-9]{64})', content)
    if not match:
        match = re.search(r'docker-([a-f0-9]{64})\.scope', content)
    if not match:
        match = re.search(r'([a-f0-9]{64})', content)
        
    if not match:
        return None
        
    container_id = match.group(1)[:12]
    
    # Get container details from docker inspect
    try:
        res = subprocess.run(
            ["docker", "inspect", container_id],
            capture_output=True, text=True, check=True
        )
        data = json.loads(res.stdout)[0]
        name = data.get("Name", "").lstrip("/")
        
        # Try to infer the host user who launched it
        host_user = None
        
        # Check environment and labels
        env = data.get("Config", {}).get("Env", [])
        labels = data.get("Config", {}).get("Labels", {})
        
        search_texts = list(labels.values()) + env
        
        working_dir = labels.get("com.docker.compose.project.working_dir", "")
        if working_dir:
            search_texts.append(working_dir)
            
        mounts = data.get("Mounts", [])
        for m in mounts:
            search_texts.append(m.get("Source", ""))
            
        # Search for username in paths
        for text in search_texts:
            m_user = re.search(r'/home/campus\.ncl\.ac\.uk/([^/]+)', text)
            if m_user:
                host_user = m_user.group(1)
                break
            m_user2 = re.search(r'/home/([^/]+)', text)
            if m_user2:
                u = m_user2.group(1)
                if u not in ["root", "ubuntu", "debian", "centos"]:
                    host_user = u
                    break
        
        if not host_user:
            host_user = "docker (root)"
            
        return {
            "container_id": container_id,
            "container_name": name,
            "host_user": host_user
        }
    except Exception:
        return {
            "container_id": container_id,
            "container_name": f"container ({container_id})",
            "host_user": "docker (unknown)"
        }

def main():
    # 1. Get GPU mapping: bus_id -> (index, name)
    gpu_map = {}
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,pci.bus_id,name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        for line in res.stdout.strip().split("\n"):
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                idx, bus_id, name = parts[0], parts[1].lower(), parts[2]
                gpu_map[bus_id] = (idx, name)
    except Exception as e:
        print(f"Error querying GPUs: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Get active compute processes on GPUs
    processes = []
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_bus_id,pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True
        )
        for line in res.stdout.strip().split("\n"):
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                bus_id, pid, mem = parts[0].lower(), parts[1], parts[2]
                processes.append((bus_id, pid, mem))
    except Exception as e:
        print(f"Error querying compute apps: {e}", file=sys.stderr)
        sys.exit(1)

    if not processes:
        print("No active processes running on GPUs.")
        return

    # Print Header
    header = f"{'GPU':<5} {'PID':<8} {'USER':<12} {'CONTAINER/HOST':<40} {'MEMORY':<10} {'COMMAND'}"
    print(header)
    print("-" * 120)

    for bus_id, pid, mem in processes:
        gpu_info = gpu_map.get(bus_id, ("?", "Unknown"))
        gpu_str = f"GPU {gpu_info[0]}"
        
        container_info = get_container_info(pid)
        if container_info:
            user = container_info["host_user"]
            env_str = f"docker: {container_info['container_name']}"
        else:
            user = get_process_owner(pid)
            env_str = "host"
            
        cmd = get_process_cmdline(pid)
        # Clean up double spaces
        cmd = re.sub(r'\s+', ' ', cmd)
        # Truncate command to fit nicely on terminal but show most details
        if len(cmd) > 80:
            cmd = cmd[:77] + "..."
            
        mem_str = f"{mem} MiB"
        
        print(f"{gpu_str:<5} {pid:<8} {user:<12} {env_str:<40} {mem_str:<10} {cmd}")

if __name__ == "__main__":
    main()
