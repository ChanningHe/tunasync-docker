#!/usr/bin/env python3
import requests
import hashlib
import traceback
import json
import os
import re
import shutil
import subprocess as sp
import tempfile
import argparse
import bz2
import gzip
import lzma
import time
import concurrent.futures
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

# Set BindIP
# https://stackoverflow.com/a/70772914
APTSYNC_BINDIP = os.getenv("BIND_ADDRESS", "")
if APTSYNC_BINDIP:
    import urllib3
    real_create_conn = urllib3.util.connection.create_connection

    def set_src_addr(address, timeout, *args, **kw):
        source_address = (APTSYNC_BINDIP, 0)
        return real_create_conn(
    address,
    timeout=timeout,
     source_address=source_address)

    urllib3.util.connection.create_connection = set_src_addr


# Set user agent
APTSYNC_USER_AGENT = os.getenv("APTSYNC_USER_AGENT", "APT-Mirror-Tool/1.0")
requests.utils.default_user_agent = lambda: APTSYNC_USER_AGENT

OS_TEMPLATE = {
    'ubuntu-lts': ["focal", "jammy", "noble"],
    'debian-current': ["bullseye", "bookworm"],
    'debian-latest2': ["bullseye", "bookworm"],
    'debian-latest': ["bookworm"],
}
MAX_RETRY = int(os.getenv('MAX_RETRY', '3'))
DOWNLOAD_TIMEOUT = int(os.getenv('DOWNLOAD_TIMEOUT', '7200'))
REPO_SIZE_FILE = os.getenv('REPO_SIZE_FILE', '')
# 是否显示aria2c详细调试信息
ARIA2_DEBUG = os.getenv('ARIA2_DEBUG', '').lower() in ('true', '1', 'yes', 'y')
# 是否显示索引调试信息
INDEX_DEBUG = os.getenv('INDEX_DEBUG', '').lower() in ('true', '1', 'yes', 'y')
# 并行下载任务数
PARALLEL_DOWNLOADS = int(os.getenv('PARALLEL_DOWNLOADS', '4'))

pattern_os_template = re.compile(r"@\{(.+)\}")
pattern_package_name = re.compile(r"^Filename: (.+)$", re.MULTILINE)
pattern_package_size = re.compile(r"^Size: (\d+)$", re.MULTILINE)
pattern_package_sha256 = re.compile(r"^SHA256: (\w{64})$", re.MULTILINE)
# Pattern to capture checksum blocks in Release files
pattern_checksum_block = re.compile(
    r"^(MD5Sum|SHA1|SHA256|SHA512):\s*$", re.MULTILINE)

download_cache = dict()

# Mapping from hashlib function names to attribute names
HASH_MAPPING = {
    "md5sum": hashlib.md5,
    "sha1": hashlib.sha1,
    "sha256": hashlib.sha256,
    "sha512": hashlib.sha512,
}
# Preferred hash order for verification
PREFERRED_HASH_ORDER = ["sha512", "sha256", "sha1", "md5sum"]


def check_args(prop: str, lst: List[str]):
    for s in lst:
        if len(s) == 0 or ' ' in s:
            raise ValueError(f"Invalid item in {prop}: {repr(s)}")


def replace_os_template(os_list: List[str]) -> List[str]:
    ret = []
    for i in os_list:
        matched = pattern_os_template.search(i)
        if matched:
            for os in OS_TEMPLATE[matched.group(1)]:
                ret.append(pattern_os_template.sub(os, i))
        elif i.startswith('@'):
            ret.extend(OS_TEMPLATE[i[1:]])
        else:
            ret.append(i)
    return ret


def check_and_download(
    url: str,
    dst_file: Path,
    caching=False,
     is_package_file=False) -> int:
    """
    Downloads a file from a URL. Uses aria2c for package files if available,
    otherwise uses requests. Handles caching for non-package files.
    Returns 0 on success, 1 on failure.
    """
    if is_package_file:
        # Use aria2c for package files
        try:
            # Check if aria2c exists first
            sp.run(['aria2c', '--version'], check=True,
                   capture_output=True, text=True)
        except (FileNotFoundError, sp.CalledProcessError) as e:
            # print(f"DEBUG: aria2c check failed: {e}", flush=True) # Debug
            # output
            print(
    f"WARN: 'aria2c' not found or not working. Falling back to requests for {url}",
     flush=True)
            # Fallback to requests logic (duplicated below for clarity, could
            # refactor)
            try:
                start_req = time.time()
                # Use longer timeout for data, keep connect timeout reasonable
                with requests.get(url, stream=True, timeout=(10, DOWNLOAD_TIMEOUT)) as r_req:
                    r_req.raise_for_status()
                    remote_ts_req = None
                    if 'last-modified' in r_req.headers:
                        try:
                            remote_ts_req = parsedate_to_datetime(
                                r_req.headers['last-modified']).timestamp()
                        except Exception as ts_e:
                            if ARIA2_DEBUG:
                                print(
    f"WARN: Could not parse Last-Modified header for {url}: {ts_e}", flush=True)

                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    with dst_file.open('wb') as f_req:
                        for chunk in r_req.iter_content(chunk_size=1024**2):
                            if time.time() - start_req > DOWNLOAD_TIMEOUT:
                                raise TimeoutError(
    f"Fallback download timeout for {url}")
                            if not chunk: continue
                            f_req.write(chunk)
                    if remote_ts_req is not None:
                        try:
                            os.utime(dst_file, (remote_ts_req, remote_ts_req))
                        except FileNotFoundError:
                            if ARIA2_DEBUG:
                                print(
    f"WARN: Fallback download - File {dst_file} not found when trying to set timestamp.",
     flush=True)
                return 0
            except BaseException as e_req:
                print(
    f"Error during fallback download (requests) for {url}: {e_req}",
     flush=True)
                if dst_file.is_file(): dst_file.unlink(missing_ok=True)
                return 1
        # --- End of Fallback ---

        # Proceed with aria2c
        # print(f"Using aria2c to download {url} to {dst_file}", flush=True)
        remote_ts = None
        try:
            # Get Last-Modified timestamp first
            with requests.head(url, timeout=(5, 10)) as r_head:
                r_head.raise_for_status()
                if 'last-modified' in r_head.headers:
                    try:
                        remote_ts = parsedate_to_datetime(
    r_head.headers['last-modified']).timestamp()
                    except Exception as ts_e:
                        if ARIA2_DEBUG:
                            print(
    f"WARN: Could not parse Last-Modified header for {url}: {ts_e}",
     flush=True)
        except requests.exceptions.RequestException as e:
            if ARIA2_DEBUG:
                print(
    f"Warning: Failed to get headers for {url}: {e}",
     flush=True)
            # Continue download attempt anyway

        # Ensure parent directory exists
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        # Construct aria2c command
        aria2c_command = [
            'aria2c',
            '-x', '8',  # Max 8 connections per server
            '-s', '8',  # Split download into 8 pieces
            '-k', '1M',  # Min split size 1MB
            '--allow-overwrite=true',
            '--auto-file-renaming=false',
            '--connect-timeout=5',
            '--timeout=60',  # Timeout for data transfer segments
            '--max-tries=3',  # Let aria2 handle some retries
            '--retry-wait=5',
            '--user-agent', APTSYNC_USER_AGENT,
            '--enable-rpc=false',
            '--show-console-readout=false',
            '--summary-interval=0',
            '--download-result=hide',
            # 使用绝对路径，防止 aria2c 添加自己的基目录前缀
            # Use resolve() for absolute path
            '-d', str(dst_file.parent.resolve()),
            '-o', dst_file.name,
            url
        ]
        if APTSYNC_BINDIP:
             aria2c_command.extend(['--interface', APTSYNC_BINDIP])

        start = time.time()
        try:
            # Use a reasonable overall timeout for the download process
            if ARIA2_DEBUG:
                print(
    f"Running aria2c command: {
        ' '.join(aria2c_command)}",
         flush=True)
                print(f"Current working directory: {os.getcwd()}", flush=True)

            # Capture stderr for better error reporting
            result = sp.run(
    aria2c_command,
    check=True,
    capture_output=True,
    text=True,
     timeout=DOWNLOAD_TIMEOUT)

            # 只在调试模式下输出详细日志
            if ARIA2_DEBUG:
                # print(f"aria2c stdout:\n{result.stdout}", flush=True) # Often
                # empty or less useful
                if result.stderr:
                    print(
    f"aria2c stderr:\n{
        result.stderr}",
         flush=True)  # Log stderr

            download_duration = time.time() - start
            # Only print success if not in debug, otherwise logs are printed
            # above
            if not ARIA2_DEBUG:
                print(
    f"aria2c finished for {
        dst_file.name} in {
            download_duration:.2f} seconds.",
             flush=True)

            # Explicitly check if the file exists *after* aria2c reports success
            # 检查文件是否存在的同时列出目录内容以便调试
            if not dst_file.is_file():
                print(
    f"ERROR: aria2c reported success, but the destination file {dst_file} is missing!",
     flush=True)

                if ARIA2_DEBUG:
                    print(
    f"Directory contents of {
        dst_file.parent}:",
         flush=True)
                    try:
                        for f in dst_file.parent.glob('*'):
                            print(f"  {f}", flush=True)
                    except Exception as e:
                        print(f"Error listing directory: {e}", flush=True)
                return 1  # Indicate failure

            # Set timestamp if available
            if remote_ts is not None:
                 try:
                    os.utime(dst_file, (remote_ts, remote_ts))
                 except FileNotFoundError:
                    # This should theoretically not happen now due to the check
                    # above, but keep for robustness
                    if ARIA2_DEBUG:
                        print(
    f"WARN: File {dst_file} not found when trying to set timestamp (unexpected).",
     flush=True)
                    # Don't return failure here if file existed moments ago but vanished
                    # return 1 # Indicate failure if file doesn't exist after
                    # download
            return 0
        except FileNotFoundError:
            # This specific error was already checked, but kept for robustness
            print(
    "Error: 'aria2c' command not found, and fallback failed.",
     flush=True)
            return 1
        except sp.CalledProcessError as e:
            print(
    f"Error: aria2c failed for {url} with exit code {
        e.returncode}", flush=True)

            # 输出详细日志，即使不在调试模式下，因为这是错误情况
            # print(f"aria2c stdout:\n{e.stdout}", flush=True)
            if e.stderr:
                # Always log stderr on error
                print(f"aria2c stderr:\n{e.stderr}", flush=True)

            if dst_file.is_file(): dst_file.unlink(missing_ok=True)  # Clean up partial download
            return 1
        except sp.TimeoutExpired as e:
            print(
    f"Error: aria2c timed out after {DOWNLOAD_TIMEOUT} seconds for {url}",
     flush=True)

            # 输出详细日志，即使不在调试模式下
            # if e.stdout:
            #     print(f"aria2c stdout:\n{e.stdout}", flush=True)
            if e.stderr:
                print(f"aria2c stderr:\n{e.stderr}", flush=True)

            if dst_file.is_file(): dst_file.unlink(missing_ok=True)  # Clean up partial download
            return 1
        except Exception as e:
            print(f"Error during aria2c execution for {url}: {e}", flush=True)
            if ARIA2_DEBUG:
                traceback.print_exc()
            if dst_file.is_file(): dst_file.unlink(missing_ok=True)  # Clean up partial download
            return 1

    else:
        # Existing logic for non-package files (metadata, etc.) using requests
        try:
            if caching:
                if url in download_cache:
                    if ARIA2_DEBUG:
                        print(f"Using cached content: {url}", flush=True)
                    # Ensure parent directory exists before writing cache
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    with dst_file.open('wb') as f:
                        f.write(download_cache[url])
                    # Try to set timestamp from cache if possible (not
                    # implemented here)
                    return 0
                download_cache[url] = bytes()  # Initialize cache entry

            start = time.time()
            # Use a slightly longer connect timeout, keep response timeout
            # short for metadata
            # Increased response timeout
            with requests.get(url, stream=True, timeout=(10, 120)) as r:
                r.raise_for_status()
                remote_ts = None  # Initialize remote_ts for this scope
                content_to_write = bytes()

                if 'last-modified' in r.headers:
                    try:
                        remote_ts = parsedate_to_datetime(
                        r.headers['last-modified']).timestamp()
                    except Exception as ts_e:
                        if ARIA2_DEBUG:
                            print(
    f"WARN: Could not parse Last-Modified header for {url}: {ts_e}",
     flush=True)

                # Ensure parent directory exists
                dst_file.parent.mkdir(parents=True, exist_ok=True)

                # Read content first
                for chunk in r.iter_content(chunk_size=1024**2):
                    # Check overall download timeout
                    if time.time() - start > DOWNLOAD_TIMEOUT:
                        raise TimeoutError(
                        f"Requests download timeout for {url}")
                    if not chunk: continue  # filter out keep-alive new chunks
                content_to_write += chunk

                # Write content to file
                with dst_file.open('wb') as f:
                    f.write(content_to_write)

                # Cache content if caching is enabled
                if caching:
                    download_cache[url] = content_to_write

                # Set timestamp if available
                if remote_ts is not None:
                    try:
                        os.utime(dst_file, (remote_ts, remote_ts))
                    except FileNotFoundError:
                         if ARIA2_DEBUG:
                            print(
    f"WARN: Requests download - File {dst_file} not found when trying to set timestamp.",
     flush=True)

            return 0
        except BaseException as e:
            print(f"Error when downloading (requests) {url}: {e}", flush=True)
            # Only print traceback in debug mode
            if ARIA2_DEBUG:
                 traceback.print_exc()
            if dst_file.is_file(): dst_file.unlink(missing_ok=True)
            # Use pop with default to avoid KeyError if url wasn't added due to
            # early error
            if caching: download_cache.pop(url, None)
            return 1


def verify_file_checksum(filepath: Path,
    expected_checksums: Dict[str,
    str],
     expected_size: Optional[int] = None) -> bool:
    """
    Verify a file against expected checksums and size.
    Uses the best available checksum from PREFERRED_HASH_ORDER.
    Returns True if verification passes, False otherwise.
    """
    if not filepath.is_file():
        print(
    f"Error: File not found for verification: {filepath}",
     flush=True)
        return False

    actual_size = filepath.stat().st_size
    if expected_size is not None and actual_size != expected_size:
        print(
    f"Error: Size mismatch for {filepath}. Expected {expected_size}, got {actual_size}",
     flush=True)
        return False

    best_algo_name = None
    expected_hash = None
    hashlib_algo = None

    # Find the best available hash algorithm to use for verification
    for algo_name in PREFERRED_HASH_ORDER:
        if algo_name in expected_checksums:
            hash_func = HASH_MAPPING.get(algo_name.lower())
            if hash_func:
                best_algo_name = algo_name
                expected_hash = expected_checksums[algo_name]
                hashlib_algo = hash_func()
                break  # Use the first preferred hash found

    if not hashlib_algo:
        print(
    f"Warning: No supported checksum found for {filepath} in expected data {expected_checksums}. Skipping checksum verification.",
     flush=True)
        return True  # Cannot verify checksum, rely on size check only

    # Calculate the actual checksum
    try:
        with filepath.open("rb") as f:
            for block in iter(lambda: f.read(1024**2), b""):
                hashlib_algo.update(block)
        actual_hash = hashlib_algo.hexdigest()
    except OSError as e:
        print(
    f"Error reading file for checksum calculation {filepath}: {e}",
     flush=True)
        return False

    # Compare checksums
    if actual_hash != expected_hash:
        print(
    f"Error: Checksum mismatch for {filepath} using {best_algo_name}. Expected {expected_hash}, got {actual_hash}",
     flush=True)
        return False

    if INDEX_DEBUG or ARIA2_DEBUG:
        print(
    f"Verified {filepath} successfully using {best_algo_name}.",
     flush=True)
    return True


def mkdir_with_dot_tmp(folder: Path) -> Tuple[Path, Path]:
    tmpdir = folder / ".tmp"
    # Do not delete tmpdir if it exists, might contain files from previous failed runs
    # if tmpdir.is_dir():
    #     shutil.rmtree(str(tmpdir))
    tmpdir.mkdir(parents=True, exist_ok=True)
    return (folder, tmpdir)


def move_files_in(src: Path, dst: Path):
    """Move files from src to dst, overwriting existing files in dst."""
    empty = True
    moved_count = 0
    failed_moves = []
    if not dst.is_dir():
        dst.mkdir(parents=True, exist_ok=True)

    for file in src.glob('*'):
        empty = False
        target_path = dst / file.name
        if ARIA2_DEBUG:
            print(f"Moving {file} to {target_path}")
        try:
            # Use rename for efficiency, fallback to move if across devices
            # potentially
            # Overwrites automatically on most systems
            file.rename(target_path)
            moved_count += 1
        except OSError as e:
            # If rename fails (e.g., across filesystems), try shutil.move
            try:
                shutil.move(str(file), str(target_path))
                moved_count += 1
            except Exception as move_e:
                print(
    f"Error moving {file} to {target_path}: {e}, fallback failed: {move_e}",
     flush=True)
                failed_moves.append(file.name)

    if empty and ARIA2_DEBUG:
        print(f"{src} is empty, nothing to move.")
    elif not empty and ARIA2_DEBUG:
         print(
    f"Moved {moved_count} items from {src} to {dst}. {
        len(failed_moves)} failures.",
         flush=True)
    if failed_moves:
        print(
    f"WARNING: Failed to move the following items from {src}: {
        ', '.join(failed_moves)}",
         flush=True)


# 定义一个下载任务结构
class DownloadTask:
    def __init__(self, url: str, dest_filename: Path,
                 pkg_size: int, pkg_checksums: Dict[str, str]):
        self.url = url
        self.dest_filename = dest_filename
        self.pkg_size = pkg_size
        # Now a dict: {'sha256': '...', 'sha1': '...'}
        self.pkg_checksums = pkg_checksums
        self.success = False
        self.error = None
        self.tmp_filename = dest_filename.with_name(
            '._syncing_.' + dest_filename.name)

# 单个文件下载功能(包括重试)


def download_single_file(task: DownloadTask) -> bool:
    """下载单个文件并进行校验"""
    for retry in range(MAX_RETRY):
        if retry > 0:
             print(
                 f"Retrying download ({retry + 1}/{MAX_RETRY}) for {task.url}", flush=True)

        # Ensure tmp directory exists before download attempt
        task.tmp_filename.parent.mkdir(parents=True, exist_ok=True)

        if check_and_download(
    task.url,
    task.tmp_filename,
     is_package_file=True) != 0:
            print(
                f"Download attempt {retry + 1}/{MAX_RETRY} failed for {task.url}.", flush=True)
            if task.tmp_filename.is_file():
                task.tmp_filename.unlink(
    missing_ok=True)  # Clean up partial tmp file
            # Exponential backoff before retrying
            time.sleep(2**(retry))
            continue

        # Verify checksum and size
        if not verify_file_checksum(
    task.tmp_filename,
    task.pkg_checksums,
     task.pkg_size):
            print(
    f"Verification failed for {
        task.tmp_filename} (from {
            task.url}). Download attempt {
                retry + 1}/{MAX_RETRY}.",
                 flush=True)
            if task.tmp_filename.is_file():
                task.tmp_filename.unlink(missing_ok=True)
            # Verification failed, treat as download failure and retry
            time.sleep(2**(retry))
            continue

        # 下载成功并校验通过
        try:
            # Ensure destination directory exists before renaming
            task.dest_filename.parent.mkdir(parents=True, exist_ok=True)
            task.tmp_filename.rename(task.dest_filename)
            if ARIA2_DEBUG:
                 print(
    f"Successfully downloaded and verified {
        task.dest_filename.name}",
         flush=True)
            return True
        except OSError as e:
             print(
    f"Error renaming temp file {
        task.tmp_filename} to {
            task.dest_filename}: {e}",
             flush=True)
             # Attempt to clean up tmp file if rename failed
             if task.tmp_filename.is_file():
                 task.tmp_filename.unlink(missing_ok=True)
             # Treat rename error as a failure for this attempt
             time.sleep(2**(retry))
             continue

    print(
    f"Failed to download and verify {
        task.dest_filename} (from {
            task.url}) after {MAX_RETRY} attempts",
             flush=True)
    # Ensure tmp file is removed if all attempts failed
    if task.tmp_filename.is_file():
        task.tmp_filename.unlink(missing_ok=True)
    return False

# 并行下载多个文件


def download_files_parallel(tasks: List[DownloadTask]) -> int:
    """并行下载多个文件，返回失败的下载数量"""
    if not tasks:
        return 0  # 没有需要下载的文件

    failed_count = 0
    # Ensure PARALLEL_DOWNLOADS is at least 1
    max_workers = max(1, min(PARALLEL_DOWNLOADS, len(tasks)))

    print(
    f"Starting parallel download of {
        len(tasks)} files with {max_workers} workers",
         flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有下载任务
        future_to_task = {
    executor.submit(
        download_single_file,
         task): task for task in tasks}

        processed_count = 0
        # 处理完成的任务
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            processed_count += 1
            try:
                success = future.result()
                if not success:
                    failed_count += 1
                    # Error message already printed in download_single_file
            except Exception as e:
                print(
    f"Download task for {
        task.url} generated an unhandled exception: {e}",
         flush=True)
                if ARIA2_DEBUG:
                    traceback.print_exc()
                failed_count += 1

            # Print progress update periodically
            if processed_count % 100 == 0 or processed_count == len(tasks):
                 progress = (processed_count / len(tasks)) * 100
                 print(
    f"Download progress: {processed_count}/{
        len(tasks)} ({
            progress:.1f}%) completed. Failures: {failed_count}",
             flush=True)

    print(
    f"Parallel download complete. {
        len(tasks) - failed_count}/{
            len(tasks)} files succeeded.",
             flush=True)
    return failed_count

def parse_release_file(release_content: str) -> Dict[str, Dict[str, Any]]:
    """
    Parses the content of a Release file.
    Returns a dictionary where keys are filenames and values are dicts
    containing size and checksums (e.g., {'size': 123, 'sha256': '...', 'md5sum': '...'}).
    """
    file_metadata = {}
    current_hash_algo = None
    lines = release_content.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        match = pattern_checksum_block.match(line)
        if match:
            # Found a checksum block header (e.g., "SHA256:")
            current_hash_algo = match.group(1).lower()  # e.g., "sha256"
            # Read the checksum entries until the next block or end of file
            while i < len(lines):
                entry_line = lines[i].strip()
                if not entry_line:  # Skip empty lines
                    i += 1
                    continue
                # Check if this line is another block header
                if pattern_checksum_block.match(entry_line):
                    break  # Start of the next block, stop processing current one

                parts = entry_line.split()
                if len(parts) == 3:
                    checksum, size_str, filename = parts
                    try:
                        size = int(size_str)
                    except ValueError:
                        if INDEX_DEBUG:
                            print(
    f"Warning: Could not parse size '{size_str}' for file '{filename}' in Release file. Skipping entry.",
     flush=True)
                        i += 1
                        continue  # Skip this invalid line

                    if filename not in file_metadata:
                        file_metadata[filename] = {
                            'size': size, 'checksums': {}}
                    elif file_metadata[filename]['size'] != size:
                         # This case should ideally not happen if the Release
                         # file is consistent
                         if INDEX_DEBUG:
                             print(
    f"Warning: Inconsistent size found for '{filename}' in Release file ({
        file_metadata[filename]['size']} vs {size}). Using the latest value.",
         flush=True)
                         file_metadata[filename]['size'] = size

                    # Store the checksum, converting algo name for consistency
                    # if needed
                    if current_hash_algo == 'md5sum':  # Normalize MD5Sum to md5sum
                        file_metadata[filename]['checksums']['md5sum'] = checksum
                    else:
                        file_metadata[filename]['checksums'][current_hash_algo] = checksum

                else:
                    # Handle potential malformed lines within a checksum block
                    if INDEX_DEBUG:
                        print(
    f"Warning: Malformed line in {current_hash_algo} block: '{entry_line}'. Skipping.",
     flush=True)
                i += 1
            # Reset current hash algorithm after processing the block
            current_hash_algo = None
        else:
            # Handle other lines in the Release file if needed (e.g., Origin, Label, Suite)
            # Currently, we only care about checksum blocks.
            pass  # Ignore other lines like Origin, Label etc. for now

    return file_metadata

def apt_mirror(base_url: str, dist: str, repo: str, arch: str,
               dest_base_dir: Path, deb_set: Dict[str, int]) -> int:
    """
    Mirrors a specific component of an APT repository.

    Returns:
        0 on success for this component.
        1 on critical failure for this component.
        Stores package paths and sizes in deb_set.
    """
    if not dest_base_dir.is_dir():
        # This should not happen if called after main's mkdir
        print(
    f"ERROR: Destination base directory {dest_base_dir} does not exist!",
     flush=True)
        return 1
    print(
    f"Starting mirror process for: {base_url} - Dist: {dist}, Repo: {repo}, Arch: {arch}",
     flush=True)

    # --- 1. Download and Prepare Release Files ---
    dist_dir_rel = Path("dists") / dist  # Relative path for URLs and structure
    dist_dir, dist_tmp_dir = mkdir_with_dot_tmp(dest_base_dir / dist_dir_rel)
    release_file_content = None
    release_file_path_tmp = None
    # Track which file (InRelease or Release) was used
    release_file_used = None

    # Try InRelease first
    inrelease_url = f"{base_url}/{dist_dir_rel}/InRelease"
    inrelease_tmp_path = dist_tmp_dir / "InRelease"
    if check_and_download(
    inrelease_url,
    inrelease_tmp_path,
     caching=True) == 0:
        if inrelease_tmp_path.is_file():
            print(f"Successfully downloaded {inrelease_url}", flush=True)
            release_file_path_tmp = inrelease_tmp_path
            release_file_used = "InRelease"
            # Create Release as a link or copy for compatibility
            release_tmp_path_compat = dist_tmp_dir / "Release"
            try:
                 if release_tmp_path_compat.exists(): release_tmp_path_compat.unlink()
                 os.link(inrelease_tmp_path, release_tmp_path_compat)
            # Fallback to copy if link fails (e.g. different filesystems)
            except OSError:
                 try:
                     shutil.copy2(inrelease_tmp_path, release_tmp_path_compat)
                 except Exception as e_copy:
                     print(
    f"Warning: Failed to create compatibility Release file from InRelease: {e_copy}",
     flush=True)
                     # Continue anyway, as InRelease is primary
            # Attempt to download Release.gpg even if InRelease exists (some tools might expect it)
            # But treat its failure as non-critical if InRelease is present
            gpg_url = f"{base_url}/{dist_dir_rel}/Release.gpg"
            gpg_tmp_path = dist_tmp_dir / "Release.gpg"
            check_and_download(
    gpg_url,
    gpg_tmp_path,
     caching=True)  # Ignore result

        else:
             print(
    f"Warning: Download reported success for {inrelease_url}, but file {inrelease_tmp_path} is missing.",
     flush=True)
             release_file_used = None  # Reset as InRelease failed validation

    # If InRelease failed or wasn't found, try Release + Release.gpg
    if not release_file_used:
        release_url = f"{base_url}/{dist_dir_rel}/Release"
        release_tmp_path = dist_tmp_dir / "Release"
        if check_and_download(
    release_url,
    release_tmp_path,
     caching=True) == 0:
            if release_tmp_path.is_file():
                print(f"Successfully downloaded {release_url}", flush=True)
                release_file_path_tmp = release_tmp_path
                release_file_used = "Release"
                # Try to download Release.gpg
                gpg_url = f"{base_url}/{dist_dir_rel}/Release.gpg"
                gpg_tmp_path = dist_tmp_dir / "Release.gpg"
                if check_and_download(
    gpg_url, gpg_tmp_path, caching=True) != 0:
                    # This might be okay if the repo isn't signed, but log a
                    # warning
                    print(
    f"Warning: Failed to download Release.gpg for {dist}. Signature verification might fail.",
     flush=True)
            else:
                print(
    f"Warning: Download reported success for {release_url}, but file {release_tmp_path} is missing.",
     flush=True)
                release_file_used = None  # Reset as Release failed validation

    # Check if we have a usable Release/InRelease file
    if not release_file_used or not release_file_path_tmp or not release_file_path_tmp.is_file():
        print(
    f"ERROR: Failed to download a valid Release or InRelease file for {dist} from {base_url}/{dist_dir_rel}/",
     flush=True)
        # Check if files existed previously - if not, maybe the dist is gone
        # upstream
        if not (
    dist_dir /
    "Release").is_file() and not (
        dist_dir /
         "InRelease").is_file():
            print(
    f"Info: Neither Release nor InRelease existed locally for {dist}. Upstream might not provide this distribution anymore. Skipping component.",
     flush=True)
            # Allow skipping without returning error if the dist never existed locally.
            # However, if only *this* component sync fails but others for the same dist worked,
            # it could lead to deletion issues if the dist *does* exist.
            # A safer approach might be to return error=1 unless *all* components for this dist fail.
            # For now, let's return 0 to avoid blocking other components if
            # this one seems genuinely absent.
            return 0  # Treat as skippable if never existed
        else:
             print(
    f"Error: Previous Release/InRelease files existed locally for {dist}. Failing component sync.",
     flush=True)
             return 1  # Critical failure if files previously existed

    # Read the content of the downloaded Release/InRelease file
    try:
        with release_file_path_tmp.open('r', encoding='utf-8', errors='ignore') as f:
            release_file_content = f.read()
    except Exception as e:
        print(
    f"ERROR: Failed to read downloaded {release_file_used} file {release_file_path_tmp}: {e}",
     flush=True)
        return 1

    # --- 2. Parse Release File ---
    print(f"Parsing {release_file_used} file for {dist}...", flush=True)
    try:
        release_metadata = parse_release_file(release_file_content)
        if not release_metadata:
             print(
    f"Warning: No file metadata found in {release_file_path_tmp}. Check the file content.",
     flush=True)
             # Continue, but index download might fail
        elif INDEX_DEBUG:
             print(
    f"INDEX_DEBUG: Found metadata for {
        len(release_metadata)} files in {release_file_used}.",
         flush=True)
             # print(f"INDEX_DEBUG: Sample metadata:
             # {list(release_metadata.items())[:2]}", flush=True)

    except Exception as e:
        print(
    f"ERROR: Failed to parse {release_file_used} file {release_file_path_tmp}: {e}",
     flush=True)
        if INDEX_DEBUG:
            traceback.print_exc()
            print(
                f"INDEX_DEBUG: Release content sample:\n{release_file_content[:500]}...", flush=True)
        return 1

    # --- 3. Download and Verify Index Files ---
    comp_dir_rel = dist_dir_rel / repo  # e.g., dists/bookworm/main
    comp_dir, comp_tmp_dir = mkdir_with_dot_tmp(dest_base_dir / comp_dir_rel)
    # e.g., dists/bookworm/main/binary-amd64
    pkgidx_dir_rel = comp_dir_rel / f"binary-{arch}"
    pkgidx_dir, pkgidx_tmp_dir = mkdir_with_dot_tmp(
        dest_base_dir / pkgidx_dir_rel)

    # Find relevant index files from Release metadata
    # List of tuples: (relative_path, temp_path, expected_meta)
    packages_files_to_process = []
    # Add other index types here later (Sources, Contents, etc.)

    print(f"Identifying index files for {dist}/{repo}/{arch}...", flush=True)
    for filename_rel, meta in release_metadata.items():
        # Check if the file belongs to the current component and architecture
        # Handle Packages files (binary)
        is_packages_index = "Packages" in Path(filename_rel).name
        is_current_component = filename_rel.startswith(f"{repo}/")
        # is_current_arch handles exact match (binary-amd64) and 'binary-all'
        # TODO: Refine arch matching logic if necessary (e.g., Contents-<arch>)
        is_current_arch_specific = f"binary-{arch}" in filename_rel
        is_arch_all = "binary-all" in filename_rel
        # Check for Contents files (optional metadata)
        is_contents_index = filename_rel.startswith(f"{repo}/Contents-{arch}") or \
                            filename_rel.startswith(
    f"{repo}/Contents-all")  # Check component specific first
        # Check dist level (less common in component structure)
        is_contents_dist_level = filename_rel.startswith(f"Contents-{arch}")

        if is_packages_index and is_current_component and (
            is_current_arch_specific or is_arch_all):
            print(
    f"Found package index in Release: {filename_rel}",
     flush=True)
            # Relative path within dists/dist dir:
            # main/binary-amd64/Packages.gz
            index_path_in_dist = Path(filename_rel)
            # Full temporary path for download: /path/to/workdir/dists/bookworm/.tmp/main/binary-amd64/Packages.gz
            # Store in component's tmp dir to keep related files together
            tmp_download_path = comp_tmp_dir / index_path_in_dist

            pkglist_url = f"{base_url}/{dist_dir_rel}/{filename_rel}"
            print(
    f"Downloading index: {pkglist_url} to {tmp_download_path}",
     flush=True)
            if check_and_download(pkglist_url, tmp_download_path) == 0:
                # Verify downloaded index file
                if verify_file_checksum(
    tmp_download_path,
    meta['checksums'],
     meta['size']):
                     packages_files_to_process.append(
    (filename_rel, tmp_download_path, meta))
                else:
                     print(
    f"ERROR: Verification failed for index file {tmp_download_path}. Skipping.",
     flush=True)
                     if tmp_download_path.is_file(): tmp_download_path.unlink(missing_ok=True)
            else:
                 print(
    f"ERROR: Failed to download index file: {pkglist_url}",
     flush=True)

        # TODO: Add similar logic for Sources, Contents, Translations etc.
        # Example for Contents (adjust paths as needed):
        # elif is_contents_index or is_contents_dist_level:
        #     print(f"Found Contents index: {filename_rel}", flush=True)
        #     tmp_download_path = comp_tmp_dir / Path(filename_rel) # Adjust target dir if needed
        #     index_url = f"{base_url}/{dist_dir_rel}/{filename_rel}"
        #     if check_and_download(index_url, tmp_download_path) == 0:
        #         if verify_file_checksum(tmp_download_path, meta['checksums'], meta['size']):
        #             # Mark for moving later, maybe store path in a different list
        #             pass
        #         else:
        #              print(f"ERROR: Verification failed for Contents file {tmp_download_path}. Skipping.", flush=True)
        #              if tmp_download_path.is_file(): tmp_download_path.unlink(missing_ok=True)
        #     else:
        # print(f"ERROR: Failed to download Contents file: {index_url}",
        # flush=True)

    # Check if we found any valid Packages files to process
    if not packages_files_to_process:
        print(
    f"ERROR: No valid Packages index files found or downloaded for {dist}/{repo}/{arch}.",
     flush=True)
        # Check if they existed locally before
        if not list(pkgidx_dir.glob('Packages*')):
            print(
    f"Info: No Packages files existed locally for {dist}/{repo}/{arch}. Upstream might not provide this combination. Skipping component.",
     flush=True)
            # Clean up potentially empty tmp dirs created
            try:
                pkgidx_tmp_dir.rmdir()
                comp_tmp_dir.rmdir()
                 # Don't remove dist_tmp_dir yet, other components might use it
            except OSError: pass  # Ignore errors if not empty
            return 0  # Not a critical error if it never existed
        else:
             print(
    f"Error: Packages files previously existed locally for {dist}/{repo}/{arch}. Failing component sync.",
     flush=True)
             return 1  # Critical failure

    # --- 4. Process Packages Index Files ---
    print(
        f"Processing {len(packages_files_to_process)} downloaded package index files for {dist}/{repo}/{arch}...",
        flush=True)
    pkgidx_content_combined = ""
    processed_files_count = 0
    final_index_paths = {}  # Track final destination paths for index files

    for filename_rel, tmp_pkgidx_file, _ in packages_files_to_process:
        if INDEX_DEBUG:
            print(
                f"INDEX_DEBUG: Processing index file {tmp_pkgidx_file} ({filename_rel})",
                flush=True)

        try:
            with tmp_pkgidx_file.open('rb') as t:
                content = t.read()

            suffix = tmp_pkgidx_file.suffix
            if suffix == '.xz':
                decoded_content = lzma.decompress(
                    content).decode('utf-8', errors='replace')
            elif suffix == '.bz2':
                decoded_content = bz2.decompress(
                    content).decode('utf-8', errors='replace')
            elif suffix == '.gz':
                decoded_content = gzip.decompress(
                    content).decode('utf-8', errors='replace')
            elif suffix == '':  # Uncompressed Packages file
                decoded_content = content.decode('utf-8', errors='replace')
            else:
                print(
                    f"Warning: Unsupported compression format for {tmp_pkgidx_file.name}, skipping.")
                continue  # Skip this file

            if INDEX_DEBUG:
                packages_count = decoded_content.count("\nFilename: ")
                print(
                    f"INDEX_DEBUG: Index {tmp_pkgidx_file.name} contains approximately {packages_count} packages.",
                    flush=True)
                # print(f"INDEX_DEBUG: First 500 bytes of decoded
                # content:\n{decoded_content[:500]}...", flush=True)

            pkgidx_content_combined += decoded_content + "\n\n"  # Ensure separation
            processed_files_count += 1

            # Determine final path for this index file (in the non-tmp dir)
            # Relative path within dists/dist dir:
            # main/binary-amd64/Packages.gz
            index_path_in_dist = Path(filename_rel)
            # Final destination:
            # /path/to/workdir/dists/bookworm/main/binary-amd64/Packages.gz
            final_dest_path = dest_base_dir / dist_dir_rel / index_path_in_dist
            final_index_paths[tmp_pkgidx_file] = final_dest_path

        except Exception as e:
            print(
                f"ERROR: Failed to decompress or parse {tmp_pkgidx_file.name}: {e}",
                flush=True
            )
            if INDEX_DEBUG:
                traceback.print_exc()
                # Try to read first few bytes for debugging even if decompression failed
                try:
                    with tmp_pkgidx_file.open('rb') as t_err:
                        content_sample = t_err.read(200)
                    print(
                        f"INDEX_DEBUG: Raw content sample (first 200 bytes): {content_sample}",
                        flush=True
                    )
                except Exception:
                    pass  # Ignore errors reading the problematic file again
            continue  # Skip this file

    if processed_files_count == 0:
        print("ERROR: Failed to process any downloaded Packages files.", flush=True)
        return 1  # Critical error if downloaded files couldn't be processed

    print(
        f"Successfully processed {processed_files_count} package index files for {dist}/{repo}/{arch}.",
        flush=True)

    # --- 5. Download Packages ---
    err = 0
    deb_count = 0
    deb_size = 0
    download_tasks = []  # Collect DownloadTask objects
    component_pkg_paths = set()  # Track relative paths within this component

    print(f"Parsing combined package index for package download list...", flush=True)
    package_entries = pkgidx_content_combined.split('\n\n')
    print(
    f"Found approximately {
        len(package_entries)} package entries to analyze.",
         flush=True)

    processed_entry_count = 0
    missing_info_count = 0
    for pkg_entry in package_entries:
        processed_entry_count += 1
        if len(pkg_entry.strip()) < 10:  # ignore blanks or very short entries
            continue

        # Simple parsing using regex (could be improved for robustness)
        pkg_filename_match = pattern_package_name.search(pkg_entry)
        pkg_size_match = pattern_package_size.search(pkg_entry)
        pkg_checksum_match = pattern_package_sha256.search(
            pkg_entry)  # TODO: Support other hashes?

        if not (pkg_filename_match and pkg_size_match and pkg_checksum_match):
            missing_info_count += 1
            if INDEX_DEBUG and missing_info_count < 10:  # Log first few misses
                 print(
                     f"INDEX_DEBUG: Skipping entry due to missing info (Filename/Size/SHA256): {pkg_entry[:100]}...", flush=True)
            continue

        try:
            # Filename is relative to the repository root, e.g., pool/main/a/adduser/adduser_3.118_all.deb
            pkg_filename = pkg_filename_match.group(1)
            pkg_size = int(pkg_size_match.group(1))
            pkg_checksum = pkg_checksum_match.group(1) # Currently only extracts SHA256

            # Create a checksums dict - for now only SHA256
            # TODO: Enhance parsing to get SHA1, MD5Sum etc. if present in Packages file
            pkg_checksums = {'sha256': pkg_checksum}
            
        except Exception as e:
            print(f"ERROR: Failed to parse package description: {e}", flush=True)
            if INDEX_DEBUG:
                print(f"INDEX_DEBUG: Problematic entry: {pkg_entry[:200]}...", flush=True)
                traceback.print_exc()
            err = 1 # Mark as error, but continue processing other packages
            continue
            
        deb_count += 1
        deb_size += pkg_size

        # Destination path is absolute
        dest_filename = dest_base_dir / pkg_filename
        # Ensure parent directory exists *before* adding download task
        # dest_filename.parent.mkdir(parents=True, exist_ok=True) # Done in download_single_file now
            
        # Relative path for tracking and deletion logic
        rel_path = pkg_filename # Already relative
        component_pkg_paths.add(rel_path)
        
        # Add to global set if it's a .deb file (used for deletion)
        # TODO: Should cleanup handle non-.deb files too?
        if dest_filename.suffix == '.deb':
            deb_set[rel_path] = pkg_size
        
        # Check if file exists locally and size matches (basic check)
        if dest_filename.is_file() and dest_filename.stat().st_size == pkg_size:
            # TODO: Add checksum verification for existing files? (Could be slow)
            if ARIA2_DEBUG:
                 print(f"Skipping existing file (size match): {pkg_filename}")
            continue
        
        # Add package to download queue
        pkg_url = f"{base_url}/{pkg_filename}"
        download_tasks.append(DownloadTask(pkg_url, dest_filename, pkg_size, pkg_checksums))

    if missing_info_count > 0:
         print(f"Warning: Skipped {missing_info_count} package entries due to missing Filename/Size/SHA256.", flush=True)
    print(f"Identified {deb_count} packages ({len(download_tasks)} need download) totaling {deb_size / (1024*1024):.2f} MB for {dist}/{repo}/{arch}.", flush=True)

    # Perform parallel download
    if download_tasks:
        print(f"Starting download of {len(download_tasks)} packages...", flush=True)
        failed_downloads = download_files_parallel(download_tasks)
        if failed_downloads > 0:
            print(f"Warning: {failed_downloads} package downloads failed for {dist}/{repo}/{arch}.", flush=True)
            err = 1 # Mark component sync as problematic if downloads failed
    else:
        print("No new packages need downloading for this component.", flush=True)


    # --- 6. Finalize: Move Files from Temp Dirs ---
    print(f"Finalizing component {dist}/{repo}/{arch}...", flush=True)
    move_error = False
    try:
        # Move downloaded and verified index files to their final location
        # Move from comp_tmp_dir/... to dest_base_dir/dists/dist/repo/binary-arch/...
        if final_index_paths:
             print(f"Moving {len(final_index_paths)} processed index files to final destination...", flush=True)
        for tmp_path, final_path in final_index_paths.items():
            if tmp_path.is_file():
                 if ARIA2_DEBUG: print(f"Moving index {tmp_path.name} to {final_path}")
                 final_path.parent.mkdir(parents=True, exist_ok=True)
                 try:
                     tmp_path.rename(final_path)
                 except OSError as e:
                      print(f"ERROR moving index file {tmp_path} to {final_path}: {e}. Trying copy.", flush=True)
                      try:
                          shutil.copy2(str(tmp_path), str(final_path))
                          tmp_path.unlink() # Remove original after successful copy
                      except Exception as copy_e:
                           print(f"ERROR: Copy fallback also failed for {tmp_path.name}: {copy_e}", flush=True)
                           move_error = True
            elif ARIA2_DEBUG:
                 print(f"Skipping move for non-existent temp index file: {tmp_path}")


        # Move Release/InRelease/Release.gpg files from dist_tmp_dir to dist_dir
        print("Moving Release files...", flush=True)
        move_files_in(dist_tmp_dir, dist_dir)

        # Clean up temporary directories
        print("Cleaning up temporary directories...", flush=True)
        try:
            # 递归删除所有临时目录
            if pkgidx_tmp_dir.exists():
                recursively_remove_dir(pkgidx_tmp_dir)
            if comp_tmp_dir.exists():
                recursively_remove_dir(comp_tmp_dir)
            # 只在不会被其他组件使用时删除dist tmp目录
            if dist_tmp_dir.exists() and not any(
                path for path in dist_tmp_dir.iterdir() 
                if path.name != "main" or not path.is_dir()
            ):
                recursively_remove_dir(dist_tmp_dir)
        except Exception as e:
            print(f"警告: 清理临时目录时出错: {e}", flush=True)
            if ARIA2_DEBUG:
                traceback.print_exc()

    except Exception as final_e:
        print(f"ERROR during finalization/cleanup for {dist}/{repo}/{arch}: {final_e}", flush=True)
        traceback.print_exc()
        err = 1 # Mark as error if finalization fails

    if move_error:
        err = 1 # Mark as error if moving index files failed

    # --- 7. Summary ---
    if err == 0:
        print(f"Successfully completed mirror process for {dist}/{repo}/{arch}.", flush=True)
    else:
        print(f"Mirror process for {dist}/{repo}/{arch} completed with errors.", flush=True)
    return err

def recursively_remove_dir(dir_path: Path):
    """递归删除目录及其所有内容"""
    if not dir_path.exists():
        return
    
    # 递归删除所有子目录和文件
    for item in dir_path.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            recursively_remove_dir(item)
    
    # 删除现在应该为空的目录
    try:
        dir_path.rmdir()
    except OSError as e:
        print(f"警告: 无法删除目录 {dir_path}: {e}", flush=True)
        # 如果仍然无法删除，输出目录内容以帮助调试
        try:
            list_dir_content(dir_path)
        except Exception as debug_e:
            print(f"尝试列出目录内容时出错: {debug_e}", flush=True)

def list_dir_content(path, indent=""):
    """递归列出目录内容"""
    print(f"列出无法删除的目录内容: {path}", flush=True)
    for item in path.iterdir():
        if item.is_file():
            try:
                print(f"{indent}文件: {item.name} ({item.stat().st_size} 字节)", flush=True)
            except OSError:
                print(f"{indent}文件: {item.name} (无法获取大小)", flush=True)
        elif item.is_dir():
            subdir_items = list(item.iterdir())
            print(f"{indent}目录: {item.name}/ (包含 {len(subdir_items)} 个项目)", flush=True)
            # 递归列出子目录内容，增加缩进
            list_dir_content(item, indent + "  ")

def apt_delete_old_debs(dest_base_dir: Path, remote_set: Dict[str, int], dry_run: bool, current_components=None):
    """
    删除不在远程索引中的包文件。
    
    Args:
        dest_base_dir: 目标基础目录
        remote_set: 远程索引中包路径和大小的映射 (当前只含.deb)
                      TODO: Extend this set to include all expected files.
        dry_run: 是否只显示将要删除的文件而不实际删除
        current_components: 当前正在同步的组件路径列表 (e.g., ['dists/bookworm/main', 'dists/bookworm/contrib'])
                           如果提供则只清理这些组件路径下的文件。
                           TODO: Adapt this to clean files based on expected paths, not just globs.
    """
    print("\nStarting cleanup process...", flush=True)
    expected_files = set(remote_set.keys()) # Currently only .deb files

    # TODO: This function needs significant rework to be accurate.
    # It should compare *all* files within the managed directories against
    # a comprehensive set of *all* expected files (indices, packages, sources, etc.)
    # derived from the *successful* sync operations.
    # The current approach only considers .deb files found by globbing and compares
    # against a remote_set that also only contains .deb paths.

    print("WARNING: Cleanup logic is currently INCOMPLETE and only considers .deb files.", flush=True)
    print("Comprehensive cleanup for index files, source packages, etc. is NOT YET IMPLEMENTED.", flush=True)

    # --- Temporary .deb cleanup logic (based on original code) ---
    on_disk_debs = set()
    search_paths = []

    if current_components:
        if INDEX_DEBUG:
            print(f"INDEX_DEBUG: Restricting .deb cleanup scan to components: {current_components}", flush=True)
        # Define search paths based on components (pool directory structure assumed)
        # This assumption might be wrong depending on the repo layout.
        for comp_rel_path in current_components: # e.g., dists/bookworm/main
            # Guess pool path structure - this is unreliable!
            # A better way is needed, based on actual 'Filename' fields.
            parts = Path(comp_rel_path).parts
            if len(parts) == 3: # Expects dists/dist_name/comp_name
                pool_comp_path = Path("pool") / parts[2] # e.g., pool/main
                search_paths.append(dest_base_dir / pool_comp_path)
            else:
                 print(f"Warning: Cannot determine pool path from component path: {comp_rel_path}", flush=True)
        # Also search directly within the component's dists path? Unlikely for .debs
        # search_paths.extend([dest_base_dir / p for p in current_components])

        if not search_paths:
             print("Warning: Could not determine specific pool paths to scan for cleanup. Scanning all pools.", flush=True)
             search_paths.append(dest_base_dir / "pool") # Fallback to scanning all of pool
    else:
        # If no specific components, scan the entire pool directory
        search_paths.append(dest_base_dir / "pool")
        # Also scan dists? Usually .debs are not directly in dists folders.
        # search_paths.append(dest_base_dir / "dists")

    print(f"Scanning for .deb files in: {', '.join(map(str, search_paths))}", flush=True)
    for search_path in search_paths:
        if search_path.is_dir():
             print(f"Searching: {search_path}/**/*.deb")
             found = list(search_path.glob('**/*.deb'))
             print(f"Found {len(found)} .deb files in {search_path}")
             on_disk_debs.update(
                 str(p.relative_to(dest_base_dir)) for p in found if p.is_file()
             )
        elif ARIA2_DEBUG:
            print(f"Cleanup scan path does not exist or is not a directory: {search_path}")

    # The 'expected_files' set currently only contains .deb paths collected during sync
    print(f"Cleanup: Found {len(expected_files)} packages in index (deb_set) and {len(on_disk_debs)} .deb packages on disk within scanned paths.", flush=True)

    # Calculate files to delete (on disk but not in index)
    deleting = on_disk_debs - expected_files

    # --- Safety Checks (adapted from original logic, applied to .deb files only) ---
    if len(deleting) > 0:
        print(f"Analyzing {len(deleting)} potentially deletable .deb files...", flush=True)
        # Simple safety check: if deleting more than 50% of discovered debs, abort.
        # This is a very basic check and might prevent valid cleanup.
        if len(on_disk_debs) > 10 and len(deleting) / len(on_disk_debs) > 0.5:
            print(f"CRITICAL WARNING: Attempting to delete {len(deleting)} out of {len(on_disk_debs)} discovered .deb files (> 50%).", flush=True)
            print("Aborting deletion as a safety measure. Check index integrity and component configuration.", flush=True)
        if INDEX_DEBUG:
                 print("INDEX_DEBUG: Sample files to be deleted:", flush=True)
                 for i, f in enumerate(sorted(list(deleting))):
                    if i >= 10: break
                    print(f"  - {f}", flush=True)
        return # Abort deletion

    # Perform deletion (or dry run)
    print(f"Preparing to delete {len(deleting)} orphaned .deb packages{' (dry run)' if dry_run else ''}", flush=True)
    deleted_count = 0
    failed_delete_count = 0
    for file_rel_path in sorted(list(deleting)):
        file_abs_path = dest_base_dir / file_rel_path
        if dry_run:
            if deleted_count < 20 or INDEX_DEBUG: # Print first few or all in debug
                print(f"Dry run: Would delete {file_rel_path}")
            deleted_count += 1
        else:
            if deleted_count < 20 or ARIA2_DEBUG: # Print first few or all in debug
                print(f"Deleting: {file_rel_path}")
            try:
                file_abs_path.unlink()
                deleted_count += 1
            except OSError as e:
                print(f"Error deleting {file_abs_path}: {e}", flush=True)
                failed_delete_count += 1

    total_deleted = deleted_count if dry_run else deleted_count - failed_delete_count
    print(f"Cleanup finished. {'Would have deleted' if dry_run else 'Deleted'} {total_deleted} orphaned .deb files.", flush=True)
    if not dry_run and failed_delete_count > 0:
        print(f"Warning: Failed to delete {failed_delete_count} files.", flush=True)

def main():

    parser = argparse.ArgumentParser(description="Mirror APT repositories using requests and aria2c.")
    parser.add_argument("base_url", type=str, help="Base URL of the APT repository (e.g., http://deb.debian.org/debian)")
    parser.add_argument("os_version", type=str, help="Comma-separated list of OS versions/codenames or templates (e.g., bookworm,@ubuntu-lts)")
    parser.add_argument("component", type=str, help="Comma-separated list of components (e.g., main,contrib,non-free)")
    parser.add_argument("arch", type=str, help="Comma-separated list of architectures (e.g., amd64,i386,arm64)")
    parser.add_argument("working_dir", type=Path, help="Working directory to store the mirror")
    parser.add_argument("--delete", action='store_true',
                        help='Delete unreferenced package files (currently .deb only, EXPERIMENTAL)')
    parser.add_argument("--delete-dry-run", action='store_true',
                        help='Print package files that would be deleted only')
    # TODO: Add arguments for source mirroring, specific index types, etc.
    args = parser.parse_args()

    # Validate and expand arguments
    try:
        os_list_raw = args.os_version.split(',')
        check_args("os_version", os_list_raw)
        component_list = args.component.split(',')
        check_args("component", component_list)
        arch_list = args.arch.split(',')
        check_args("arch", arch_list)
    except ValueError as e:
        parser.error(str(e))

    os_list = replace_os_template(os_list_raw)
    print(f"Mirroring for OS: {os_list}, Components: {component_list}, Archs: {arch_list}")
    print(f"Base URL: {args.base_url}")
    print(f"Working Directory: {args.working_dir}")

    # Ensure working directory exists
    try:
        args.working_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating working directory {args.working_dir}: {e}", flush=True)
        exit(1)

    # --- Main Mirroring Loop ---
    start_time = time.time()
    failed_components = []
    successful_components_details = [] # Store (os, comp, arch) for successful ones
    deb_set = {}  # Global package set (path -> size) for cleanup (currently .deb only)

    total_combinations = len(os_list) * len(component_list) * len(arch_list)
    print(f"\nStarting sync for {total_combinations} combinations...", flush=True)
    combination_count = 0

    for os_ver in os_list:
        for comp in component_list:
            for arch in arch_list:
                combination_count += 1
                print(f"\n--- [{combination_count}/{total_combinations}] Syncing: OS={os_ver}, Comp={comp}, Arch={arch} ---", flush=True)
                # Run the mirroring process for this specific combination
                component_start_time = time.time()
                result = apt_mirror(args.base_url, os_ver, comp, arch, args.working_dir, deb_set=deb_set)
                component_duration = time.time() - component_start_time
                print(f"--- Finished {os_ver}/{comp}/{arch} in {component_duration:.2f} seconds. Result: {'Success' if result == 0 else 'Failed'} ---\n", flush=True)

                if result != 0:
                    failed_components.append((os_ver, comp, arch))
                else:
                    successful_components_details.append((os_ver, comp, arch))

    print(f"\n--- Sync Phase Complete ---", flush=True)
    total_duration = time.time() - start_time
    print(f"Total sync time: {total_duration:.2f} seconds.")
    print(f"Successfully synced {len(successful_components_details)} components.")
    if failed_components:
        print(f"Failed to sync {len(failed_components)} components:")
        for os_ver, comp, arch in failed_components:
            print(f"  - {os_ver}/{comp}/{arch}")
    else:
        print("All components synced successfully.")

    # Safety check: If all components failed, do not proceed to delete.
    if len(failed_components) == total_combinations:
        print("\nCRITICAL: All component syncs failed. Skipping deletion process entirely for safety.", flush=True)
        exit(len(failed_components))

    # --- Cleanup Phase ---
    if args.delete or args.delete_dry_run:
        print(f"\n--- Cleanup Phase ({'Dry Run' if args.delete_dry_run else 'Actual Deletion'}) ---", flush=True)

        if not deb_set:
             # This can happen if only non-.deb files were synced or if all syncs failed to populate deb_set
             print("Warning: No package information collected during sync (deb_set is empty). Skipping deletion.", flush=True)
        else:
            # Build the list of paths corresponding to *successfully* synced components
            # to limit the scope of the cleanup scan.
            successful_component_paths = []
            for os_ver, comp, arch in successful_components_details:
                 # This path is used by apt_delete_old_debs to guess pool locations etc.
                 # Needs refinement based on how cleanup logic evolves.
                 comp_path_str = str(Path("dists") / os_ver / comp)
                 if comp_path_str not in successful_component_paths:
                     successful_component_paths.append(comp_path_str)
        
        if INDEX_DEBUG:
            print(f"INDEX_DEBUG: Passing {len(successful_component_paths)} successful component base paths to cleanup function:", flush=True)
            # print(successful_component_paths)

            # Call the cleanup function
            apt_delete_old_debs(args.working_dir, deb_set, args.delete_dry_run, current_components=successful_component_paths)
    else:
        print("\nCleanup skipped (no --delete or --delete-dry-run specified).", flush=True)

    # --- Final Summary & Size Reporting ---
    print("\n--- Final Summary ---")
    if len(REPO_SIZE_FILE) > 0:
        try:
            # Calculate total size from the collected deb_set
            total_size = sum(deb_set.values())
            print(f"Total size of tracked .deb files: {total_size / (1024*1024):.2f} MB")
            # Append size delta (assuming initial size is 0 for this run)
            # A more robust approach would read the previous size if available
            with open(REPO_SIZE_FILE, "a") as fd:
                # Format as "+<size>" might be specific to another tool's input needs
                # Consider just writing the total size or a timestamped entry
                fd.write(f"\n{time.strftime('%Y-%m-%d %H:%M:%S')} {args.working_dir} size: +{total_size}")
            print(f"Repository size information updated in {REPO_SIZE_FILE}", flush=True)
        except Exception as e:
             print(f"Error writing repository size to {REPO_SIZE_FILE}: {e}", flush=True)

    final_exit_code = len(failed_components)
    print(f"\nExiting with status code: {final_exit_code}")
    exit(final_exit_code)


if __name__ == "__main__":
    main()
