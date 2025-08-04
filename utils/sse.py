import json
import time
import logging
from typing import Generator, Dict, Any, Optional
from flask import Response

logger = logging.getLogger(__name__)

def format_sse_event(event_type: str, data: Dict[str, Any], event_id: Optional[str] = None) -> str:
    """
    Format data as Server-Sent Event.
    
    Args:
        event_type: The event type
        data: The event data (will be JSON serialized)
        event_id: Optional event ID for client tracking
        
    Returns:
        Formatted SSE string
    """
    sse_data = []
    
    if event_id:
        sse_data.append(f"id: {event_id}")
    
    sse_data.append(f"event: {event_type}")
    
    # Serialize data to JSON and handle multi-line data
    json_data = json.dumps(data, default=str)
    for line in json_data.split('\n'):
        sse_data.append(f"data: {line}")
    
    # SSE format requires double newline at the end
    sse_data.append("")
    sse_data.append("")
    
    return '\n'.join(sse_data)

def create_sse_response(event_generator: Generator[str, None, None]) -> Response:
    """
    Create a Flask Response object for Server-Sent Events.
    
    Args:
        event_generator: Generator that yields SSE-formatted strings
        
    Returns:
        Flask Response with appropriate SSE headers
    """
    response = Response(
        event_generator,
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'X-Accel-Buffering': 'no',  # Disable nginx buffering
        }
    )
    
    return response

def stream_job_events(job_id: str, streaming_manager, timeout_seconds: int = 300) -> Generator[str, None, None]:
    """
    Generator that yields SSE events for a streaming job.
    
    Args:
        job_id: The job identifier
        streaming_manager: The StreamingJobManager instance
        timeout_seconds: Maximum time to wait for job completion
        
    Yields:
        SSE-formatted event strings
    """
    start_time = time.time()
    last_event_index = 0
    
    try:
        logger.info(f"📡 Starting SSE stream for job {job_id}")
        
        # Send initial keepalive
        yield format_sse_event('connected', {
            'job_id': job_id,
            'timestamp': time.time(),
            'message': 'Connected to streaming endpoint'
        })
        
        while True:
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                logger.warning(f"📡 Stream timeout for job {job_id}")
                yield format_sse_event('error', {
                    'job_id': job_id,
                    'error': 'Stream timeout',
                    'timeout_seconds': timeout_seconds
                })
                break
            
            # Get job status
            job = streaming_manager.get_job(job_id)
            if not job:
                logger.error(f"📡 Job {job_id} not found")
                yield format_sse_event('error', {
                    'job_id': job_id,
                    'error': 'Job not found'
                })
                break
            
            # Send any new events
            if len(job.events) > last_event_index:
                new_events = job.events[last_event_index:]
                for event in new_events:
                    yield format_sse_event(
                        event['event'],
                        event['data'],
                        event_id=f"{job_id}_{last_event_index}"
                    )
                    last_event_index += 1
            
            # Check if job is completed
            if job.status == 'completed':
                logger.info(f"✅ Job {job_id} completed, ending stream")
                break
            elif job.status == 'failed':
                logger.error(f"❌ Job {job_id} failed, ending stream")
                yield format_sse_event('error', {
                    'job_id': job_id,
                    'error': 'Job processing failed'
                })
                break
            
            # Send periodic heartbeat
            current_time = time.time()
            if int(current_time) % 30 == 0:  # Every 30 seconds
                yield format_sse_event('heartbeat', {
                    'job_id': job_id,
                    'timestamp': current_time,
                    'active_jobs': streaming_manager.get_active_jobs_count()
                })
            
            # Small sleep to prevent busy waiting
            time.sleep(0.5)
            
    except GeneratorExit:
        # Client disconnected
        logger.info(f"📡 Client disconnected from job {job_id}")
        streaming_manager.mark_client_disconnected(job_id)
    except Exception as e:
        logger.error(f"📡 Error in SSE stream for job {job_id}: {str(e)}")
        yield format_sse_event('error', {
            'job_id': job_id,
            'error': f'Stream error: {str(e)}'
        })

def stream_immediate_response(data: Dict[str, Any]) -> Generator[str, None, None]:
    """
    Create a simple SSE response for immediate data.
    
    Args:
        data: Data to send as SSE event
        
    Yields:
        SSE-formatted event string
    """
    yield format_sse_event('response', data)
    yield format_sse_event('complete', {'finished': True})