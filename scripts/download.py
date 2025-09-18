import argparse
import json
import os
import shutil
from glob import glob

from bdpy.dataset.utils import download_file


def is_postprocess_completed(output_path, postproc_name, postproc_config):
    """Check if postprocess has already been completed by examining output files/directories"""
    
    if postproc_name == "merge":
        # For merge, check if the final output file exists
        return os.path.exists(output_path)
    
    elif postproc_name == "unzip":
        dest = postproc_config.get("destination")
        if not dest:
            return False
        # Check if destination directory exists and has content
        if not os.path.exists(dest):
            return False
        try:
            return len(os.listdir(dest)) > 0
        except OSError:
            return False
    
    elif postproc_name == "unzip-reorganize":
        dest = postproc_config.get("destination")
        if not dest:
            return False
        # Check if destination directory exists and has content
        if not os.path.exists(dest):
            return False
        try:
            return len(os.listdir(dest)) > 0
        except OSError:
            return False
    
    elif postproc_name == "copy":
        dest = postproc_config.get("destination")
        if not dest:
            return False
        # Check if the copied file exists in destination
        copied_file = os.path.join(dest, os.path.basename(output_path))
        return os.path.exists(copied_file)
    
    return False


def main(cfg):
    with open(cfg.filelist) as f:
        filelist = json.load(f)

    target = filelist[cfg.target]

    for fl in target["files"]:
        output = os.path.join(target["save_in"], fl["name"])
        os.makedirs(target["save_in"], exist_ok=True)

        # Downloading
        if not os.path.exists(output):
            if isinstance(fl["url"], str):
                print(f'Downloading {output} from {fl["url"]}')
                download_file(fl["url"], output, progress_bar=True, md5sum=fl["md5sum"])
            else:
                # fl['url'] and fl['md5sum'] are lists
                for i, (url, md5) in enumerate(zip(fl["url"], fl["md5sum"])):
                    output_chunk = output + f".{i:04d}"
                    if os.path.exists(output_chunk):
                        continue
                    print(f"Downloading {output_chunk} from {url}")
                    download_file(url, output_chunk, progress_bar=True, md5sum=md5)

        # Postprocessing
        if "postproc" in fl:
            for pp in fl["postproc"]:
                # Check if this postprocess has already been completed
                if is_postprocess_completed(output, pp["name"], pp):
                    print(f"Skipping {pp['name']} for {output} (already completed)")
                    continue
                
                if pp["name"] == "merge":
                    if os.path.exists(output):
                        continue
                    print(f"Merging {output}")
                    cat_files = sorted(glob(output + ".*"))
                    with open(output, "wb") as f:
                        for cf in cat_files:
                            with open(cf, "rb") as cf_f:
                                shutil.copyfileobj(cf_f, f)
                elif pp["name"] == "unzip":
                    print(f"Unzipping {output}")
                    if "destination" in pp:
                        os.makedirs(pp["destination"], exist_ok=True)
                        dest = pp["destination"]
                    else:
                        print("No destination specified for unzip")
                    shutil.unpack_archive(output, extract_dir=dest)
                elif pp["name"] == "unzip-reorganize":
                    print(f"Unzipping and reorganizing {output}")
                    if "destination" not in pp:
                        print("No destination specified for unzip-reorganize")
                        continue
                    if "source_pattern" not in pp:
                        print("No source_pattern specified for unzip-reorganize")
                        continue

                    dest = pp["destination"]
                    source_pattern = pp["source_pattern"]
                    os.makedirs(dest, exist_ok=True)

                    # Extract to temporary directory first
                    temp_dir = dest + "_temp"
                    os.makedirs(temp_dir, exist_ok=True)
                    shutil.unpack_archive(output, extract_dir=temp_dir)

                    # Reorganize directory structure
                    source_path = os.path.join(temp_dir, source_pattern)

                    if os.path.exists(source_path):
                        # Move all contents from source to target
                        for item in os.listdir(source_path):
                            source_item = os.path.join(source_path, item)
                            target_item = os.path.join(dest, item)

                            if os.path.exists(target_item):
                                if os.path.isdir(target_item):
                                    shutil.rmtree(target_item)
                                else:
                                    os.remove(target_item)

                            shutil.move(source_item, target_item)
                            print(f"Moved {source_item} to {target_item}")
                    else:
                        print(f"Warning: Source pattern {source_path} not found")

                    # Clean up temporary directory
                    shutil.rmtree(temp_dir)
                elif pp["name"] == "copy":
                    print(f"Copying {output}")
                    if "destination" in pp:
                        os.makedirs(pp["destination"], exist_ok=True)
                        dest = pp["destination"]
                    else:
                        print("No destination specified for copy")
                    shutil.copy(output, dest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filelist", default="scripts/download_files.json")
    parser.add_argument("--target", default="basic_data")

    cfg = parser.parse_args()

    main(cfg)
