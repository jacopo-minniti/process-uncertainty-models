# rely/inference/sbs/clients.py

import uuid
import time
from multiprocessing import Queue
from typing import List, Dict, Any

class ValueClient:
    """Client for interacting with the Value Model server via Queues."""
    def __init__(self, task_queue: Queue, result_queue: Queue, worker_rank: int):
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.worker_rank = worker_rank

    def get_values(self, prompts: List[str], generated_texts: List[str]) -> List[float]:
        if not prompts:
            return []
        request_id = str(uuid.uuid4())
        payload = {
            "request_id": request_id, 
            "worker_rank": self.worker_rank, 
            "prompts": prompts, 
            "generated_texts": generated_texts
        }
        self.task_queue.put(payload)
        
        while True:
            response = self.result_queue.get()
            if response.get("request_id") == request_id:
                return response["values"]
            # If the message is not for us, put it back and wait.
            self.result_queue.put(response)
            time.sleep(0.01)

class UncertaintyClient:
    """Client for interacting with the Uncertainty Model server via Queues."""
    def __init__(self, task_queue: Queue, result_queue: Queue, worker_rank: int):
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.worker_rank = worker_rank

    def get_uncertainties(self, prompts: List[str]) -> List[float]:
        if not prompts:
            return []
        request_id = str(uuid.uuid4())
        payload = {
            "request_id": request_id, 
            "worker_rank": self.worker_rank, 
            "prompts": prompts
        }
        self.task_queue.put(payload)
        
        while True:
            response = self.result_queue.get()
            if response.get("request_id") == request_id:
                return response["uncertainties"]
            # If the message is not for us, put it back and wait.
            self.result_queue.put(response)
            time.sleep(0.01)