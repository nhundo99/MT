from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Point this to your TensorBoard log directory
log_dir = "../results/runs/20260724_1431_GBM_wrong_drift_monte_carlo_regularized_most_samples"
event_acc = EventAccumulator(log_dir)
event_acc.Reload()

# Get the events for your main loss scalar
events = event_acc.Scalars("LearningRate/train")

if events:
    start_time = events[0].wall_time
    end_time = events[-1].wall_time
    duration_seconds = end_time - start_time
    
    hours = duration_seconds // 3600
    minutes = (duration_seconds % 3600) // 60
    print(f"Training duration: {hours} hours, {minutes} minutes")