"""
Status tracking service for the Auto Web Scraper.
"""
import time
import uuid
from typing import Dict, List, Optional, Any
import threading
from ..models.schemas import ProcessStage, StatusResponse
from ..utils.logger import app_logger

# Global dictionary to store process status information
# Key: request_id, Value: StatusResponse
_status_store: Dict[str, StatusResponse] = {}
_status_lock = threading.Lock()

def create_request_id() -> str:
    """Generate a unique request ID"""
    return str(uuid.uuid4())

def create_status(request_id: str) -> StatusResponse:
    """Create a new status entry for a request"""
    status = StatusResponse(
        request_id=request_id,
        status="initializing",
        current_stage=None,
        progress=0.0,
        stages=[],
        message="Request initialized",
        error=None
    )

    with _status_lock:
        _status_store[request_id] = status

    return status

def get_status(request_id: str) -> Optional[StatusResponse]:
    """Get the current status for a request"""
    with _status_lock:
        return _status_store.get(request_id)

def update_status(request_id: str, **kwargs) -> Optional[StatusResponse]:
    """Update the status for a request"""
    with _status_lock:
        status = _status_store.get(request_id)
        if not status:
            return None

        for key, value in kwargs.items():
            if hasattr(status, key):
                setattr(status, key, value)

        return status

def start_stage(request_id: str, stage_name: str, details: Optional[str] = None) -> Optional[ProcessStage]:
    """Start a new processing stage"""
    with _status_lock:
        status = _status_store.get(request_id)
        if not status:
            return None

        # Create a new stage
        stage = ProcessStage(
            name=stage_name,
            start_time=time.time(),
            status="in_progress",
            details=details
        )

        # Add to stages list
        status.stages.append(stage)

        # Update current stage
        status.current_stage = stage_name

        # Calculate progress based on predefined stages
        all_stages = ["setup", "prompt_formatting", "code_generation", "code_extraction",
                      "code_execution", "data_validation", "refinement"]

        if stage_name in all_stages:
            stage_index = all_stages.index(stage_name)
            # Each stage represents progress towards completion
            status.progress = min(95.0, (stage_index / len(all_stages)) * 100)

        return stage

def complete_stage(request_id: str, stage_name: str, details: Optional[str] = None) -> Optional[ProcessStage]:
    """Mark a processing stage as completed"""
    with _status_lock:
        status = _status_store.get(request_id)
        if not status:
            return None

        # Find the stage
        stage = next((s for s in status.stages if s.name == stage_name), None)
        if not stage:
            return None

        # Update stage
        stage.end_time = time.time()
        stage.duration = stage.end_time - stage.start_time
        stage.status = "completed"
        if details:
            stage.details = details

        # If this is the last stage, mark as completed
        all_stages = ["setup", "prompt_formatting", "code_generation", "code_extraction",
                      "code_execution", "data_validation", "refinement"]

        if stage_name == all_stages[-1] or stage_name == "data_validation":
            status.status = "completed"
            status.progress = 100.0
            status.message = "Processing completed"

        return stage

def fail_stage(request_id: str, stage_name: str, error: str) -> Optional[ProcessStage]:
    """Mark a processing stage as failed"""
    with _status_lock:
        status = _status_store.get(request_id)
        if not status:
            return None

        # Find the stage
        stage = next((s for s in status.stages if s.name == stage_name), None)
        if not stage:
            return None

        # Update stage
        stage.end_time = time.time()
        stage.duration = stage.end_time - stage.start_time
        stage.status = "failed"
        stage.details = error

        # Update overall status
        status.status = "failed"
        status.error = error
        status.message = f"Failed during {stage_name}: {error}"

        return stage

def cleanup_old_statuses(max_age_seconds: int = 300) -> int:
    """Remove status entries older than max_age_seconds (default: 5 minutes)"""
    current_time = time.time()
    to_remove = []

    with _status_lock:
        for request_id, status in _status_store.items():
            # Check if the status is old enough to remove
            if status.stages and status.stages[0].start_time < (current_time - max_age_seconds):
                to_remove.append(request_id)
            # Also remove completed statuses after a shorter time
            elif status.status in ["completed", "failed"] and status.stages and \
                 status.stages[-1].end_time and \
                 status.stages[-1].end_time < (current_time - 60):  # 1 minute for completed statuses
                to_remove.append(request_id)

        # Remove old entries
        for request_id in to_remove:
            # Use debug level to avoid cluttering logs
            app_logger.debug(f"Cleaning up old status entry for request {request_id}")
            del _status_store[request_id]

    return len(to_remove)
