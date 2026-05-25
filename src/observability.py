import time
import json
import os

class PipelineLogger:
    def __init__(self, metrics_path: str = "data/results/pipeline_metrics.json"):
        self.metrics_path = metrics_path
        self.start_times = {}
        self.metrics = {}
        
        # Load existing metrics if present
        if os.path.exists(self.metrics_path):
            try:
                with open(self.metrics_path, "r") as f:
                    self.metrics = json.load(f)
            except Exception:
                pass

    def start_timer(self, step_name: str):
        self.start_times[step_name] = time.time()
        print(f"[{step_name.upper()}] Execution started...")

    def stop_timer(self, step_name: str, records_processed: int = 0):
        if step_name in self.start_times:
            elapsed = time.time() - self.start_times[step_name]
            self.metrics[step_name] = {
                'execution_time_seconds': float(elapsed),
                'records_processed': int(records_processed),
                'last_run_timestamp': float(time.time())
            }
            # Save metrics
            os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
            with open(self.metrics_path, "w") as f:
                json.dump(self.metrics, f, indent=4)
            print(f"[{step_name.upper()}] Execution finished in {elapsed:.2f} seconds. Rows processed: {records_processed}")

    def log_ml_metric(self, model_name: str, accuracy: float, f1: float):
        if 'ml_models' not in self.metrics:
            self.metrics['ml_models'] = {}
            
        self.metrics['ml_models'][model_name] = {
            'accuracy': float(accuracy),
            'f1_score': float(f1),
            'last_trained_timestamp': float(time.time())
        }
        
        with open(self.metrics_path, "w") as f:
            json.dump(self.metrics, f, indent=4)
        print(f"[ML_OBSERVABILITY] Logged model {model_name} (Acc: {accuracy:.4f}, F1: {f1:.4f})")
