# core/package_manager.py
import os
import subprocess
import sys
import uuid
import shutil
from pathlib import Path
from typing import Optional, Dict

from .smart_extractor import SmartExtractor

# Import manifest integration
try:
    from .manifest_integration import (
        integrate_manifest_with_package_manager,
        integrate_manifest_from_folder,
    )
    MANIFEST_INTEGRATION_AVAILABLE = True
except ImportError:
    MANIFEST_INTEGRATION_AVAILABLE = False
    print("[package_manager] Warning: manifest_integration not available")


class PackageManager:
    def __init__(self,
                 zip_path: Optional[Path] = None,
                 comfy_path: Path = None,
                 temp_path: Optional[Path] = None,
                 extract_minimal: bool = False,
                 force_extract: bool = False,
                 log_file: Optional[Path] = None,
                 folder_path: Optional[Path] = None):
        """
        Initialize PackageManager with smart extraction and manifest capabilities.

        Either zip_path or folder_path must be provided (not both).
        """
        if zip_path is None and folder_path is None:
            raise ValueError("Either zip_path or folder_path must be provided")
        if comfy_path is None:
            raise ValueError("comfy_path is required")

        self.comfy_path = Path(comfy_path).resolve()
        self.temp_path = Path(temp_path).resolve() if temp_path else self.comfy_path / "temp"
        self.extract_minimal = extract_minimal
        self.force_extract = force_extract
        self.log_file = log_file
        self.temp_dir = None
        self.created_temp_dir = False
        self.extractor: Optional[SmartExtractor] = None
        self.custom_nodes_extracted = False  # Tracks BOTH smart extractor AND manifest installations
        self.has_manifest = False
        self.manifest_config: Optional[Dict] = None

        if folder_path is not None:
            self.is_folder_mode = True
            self.folder_path = Path(folder_path).resolve()
            self.zip_path = None
            self.package_name = self.folder_path.name
        else:
            self.is_folder_mode = False
            self.zip_path = Path(zip_path).resolve()
            self.folder_path = None
            self.package_name = self.zip_path.stem

    def log(self, message: str):
        print(message)
        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')

    def extract_zip(self) -> Path:
        """
        Extract ZIP using manifest handler (if present) + SmartExtractor.
        
        Workflow:
        1. Create temp directory
        2. Check for manifest in ZIP
        3. If manifest exists:
           a. Process manifest (downloads models from HF/Git/URLs)
           b. Track if custom nodes were installed via manifest
           c. Run smart extractor for bundled ComfyUI/ content
        4. If no manifest:
           a. Run smart extractor only (existing behavior)
        5. Save workflows to user directory
        
        Returns path to extracted workflow.json
        """
        self.temp_dir = self.temp_path / f"temp_{uuid.uuid4().hex[:8]}"
        os.makedirs(self.temp_dir, exist_ok=True)
        self.created_temp_dir = True
        self.log(f"Extracting package to temp dir: {self.temp_dir}")

        # === MANIFEST INTEGRATION ===
        if MANIFEST_INTEGRATION_AVAILABLE:
            self.log("[package_manager] Checking for manifest...")
            
            has_manifest, manifest_config, custom_nodes_from_manifest = integrate_manifest_with_package_manager(
                zip_path=self.zip_path,
                comfy_path=self.comfy_path,
                temp_dir=self.temp_dir,
                log_file=self.log_file,
                skip_existing=not self.force_extract,  # Skip existing unless force extract
                verify_checksums=True,
                parallel_downloads=True,
                max_workers=4
            )
            
            self.has_manifest = has_manifest
            self.manifest_config = manifest_config
            
            # Track custom nodes from manifest
            if custom_nodes_from_manifest:
                self.custom_nodes_extracted = True
                self.log("[package_manager] ✅ Custom nodes installed via manifest - restart will be required")
            
            if has_manifest:
                self.log("[package_manager] ✅ Manifest processed - models downloaded")
                self.log("[package_manager] Continuing with smart extraction for bundled content...")
            else:
                self.log("[package_manager] No manifest detected - using standard extraction")
        else:
            self.log("[package_manager] Manifest integration not available - using standard extraction")

        # === SMART EXTRACTOR ===
        # Always run smart extractor to handle:
        # - Bundled ComfyUI/ content (models, inputs, custom nodes, etc.)
        # - Root JSON files (workflow.json, warmup.json, baseconfig.json)
        # - Scripts (pre.py, post.py)
        self.extractor = SmartExtractor(
            zip_path=self.zip_path,
            comfy_path=self.comfy_path,
            temp_dir=self.temp_dir,
            log_file=self.log_file,
            minimal=self.extract_minimal,
            force_extraction=self.force_extract
        )

        try:
            workflow_path = self.extractor.extract()
            self.log(f"[package_manager] Smart extraction complete: {workflow_path}")
        except Exception as e:
            self.log(f"[package_manager] Smart extraction failed: {e}")
            raise

        # Track custom nodes from smart extractor (bundled in ZIP)
        # This combines with manifest-based installations
        if self.extractor.custom_nodes_extracted:
            self.custom_nodes_extracted = True
            self.log("[package_manager] ✅ Custom nodes extracted from ZIP - restart will be required")

        # === SAVE TO USER WORKFLOWS (for GUI/history) ===
        workflow_dir = self.comfy_path / "user" / "default" / "workflows" / self.package_name
        os.makedirs(workflow_dir, exist_ok=True)
        for json_file in ["workflow.json", "warmup.json", "baseconfig.json"]:
            src = self.temp_dir / json_file
            if src.exists():
                dest = workflow_dir / json_file
                shutil.copy2(src, dest)
                self.log(f"[package_manager] Saved {json_file} to {dest}")

        return workflow_path
    
    def process_folder(self, folder_path: Optional[Path] = None) -> Path:
        """
        Process an already-extracted workflow folder (no ZIP needed).

        Reads manifest.yaml and baseconfig.json directly from the folder,
        downloads any missing models declared in the manifest, then copies
        workflow files to the ComfyUI user workflows directory.

        Returns path to workflow.json inside the folder.
        """
        if folder_path is None:
            folder_path = self.folder_path
        folder_path = Path(folder_path).resolve()

        # temp_dir points at the folder so downstream load_baseconfig() and
        # safetensors validation (which probe package_manager.temp_dir) work unchanged.
        self.temp_dir = folder_path
        self.log(f"[package_manager] Processing workflow folder: {folder_path}")

        # === MANIFEST INTEGRATION ===
        if MANIFEST_INTEGRATION_AVAILABLE:
            self.log("[package_manager] Checking for manifest in folder...")
            has_manifest, manifest_config, custom_nodes_from_manifest = integrate_manifest_from_folder(
                folder_path=folder_path,
                comfy_path=self.comfy_path,
                log_file=self.log_file,
                skip_existing=not self.force_extract,
                verify_checksums=True,
                parallel_downloads=True,
                max_workers=4,
            )
            self.has_manifest = has_manifest
            self.manifest_config = manifest_config
            if custom_nodes_from_manifest:
                self.custom_nodes_extracted = True
                self.log("[package_manager] Custom nodes installed via manifest - restart will be required")
            elif has_manifest:
                self.log("[package_manager] Manifest processed - models already present")
        else:
            self.log("[package_manager] Manifest integration not available - skipping manifest check")

        # === SAVE TO USER WORKFLOWS (for GUI/history) ===
        workflow_dir = self.comfy_path / "user" / "default" / "workflows" / self.package_name
        os.makedirs(workflow_dir, exist_ok=True)
        for json_file in ["workflow.json", "warmup.json", "baseconfig.json"]:
            src = folder_path / json_file
            if src.exists():
                dest = workflow_dir / json_file
                shutil.copy2(src, dest)
                self.log(f"[package_manager] Saved {json_file} to {dest}")

        workflow_path = folder_path / "workflow.json"
        if not workflow_path.exists():
            raise FileNotFoundError(f"workflow.json not found in {folder_path}")

        return workflow_path

    def get_manifest_config(self) -> Optional[Dict]:
        """
        Get configuration from manifest (NUM_INSTANCES, GENERATIONS).
        
        Returns:
            Dictionary with config values or None if no manifest
        """
        return self.manifest_config if self.has_manifest else None

    def cleanup(self):
        """Clean up temp directory."""
        if self.is_folder_mode:
            # temp_dir points at the source workflow folder — never delete it
            self.log("[package_manager] Folder mode - skipping cleanup (source directory preserved)")
            return
        if self.extractor:
            self.extractor.cleanup()
        elif self.temp_dir and self.temp_dir.exists() and self.created_temp_dir:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.log(f"[package_manager] Removed temporary directory: {self.temp_dir}")