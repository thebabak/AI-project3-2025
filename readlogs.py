from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Path to a specific log directory (e.g., logs/adam_fold_0/)
log_dir = "logs/adam/"

# Load the event accumulator
event_acc = EventAccumulator(log_dir)
event_acc.Reload()  # Load events from the log directory

# Get available tags
tags = event_acc.Tags()
print("Available Tags:", tags)

# Access scalar metrics (e.g., 'Loss/Train')
scalar_data = event_acc.Scalars('Loss/Train')
for entry in scalar_data:
    print(f"Step: {entry.step}, Value: {entry.value}")