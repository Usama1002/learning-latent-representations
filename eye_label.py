import numpy as np
import matplotlib.pyplot as plt
from dataloader import get_dataloaders

def eye_diagram_label(signal, time, bit_period_ps=100, samples_per_bit=100, 
                      eye_height_mv=80, eye_width_ps=35, plot_path=None):
    """
    Plots the eye diagram for a given signal and returns 'valid' or 'invalid' based on eye window criteria.

    Args:
        signal (np.ndarray): 1D array of voltage values (in V).
        time (np.ndarray): 1D array of time values (in ps).
        bit_period_ps (float): Bit period in picoseconds (default: 100).
        samples_per_bit (int): Number of samples per bit period (default: 100).
        eye_height_mv (float): Eye window height in mV (default: 80).
        eye_width_ps (float): Eye window width in ps (default: 35).
        plot_path (str or None): If provided, saves the plot to this path.

    Returns:
        label (str): 'valid' if eye opening is above threshold, else 'invalid'.
    """
    # Normalize time to start at 0
    t = time - time[0]
    bit_period = bit_period_ps  # in ps
    samples_per_period = samples_per_bit
    UI_to_plot = 2

    # Segment the signal into overlapping windows for the eye diagram
    num_bits = len(signal) // samples_per_period
    segments = []
    for i in range(num_bits - 1):
        start_idx = i * samples_per_period
        end_idx = start_idx + samples_per_period * UI_to_plot
        if end_idx <= len(signal):
            segments.append(signal[start_idx:end_idx])
    segments = np.array(segments)
    if segments.shape[0] == 0:
        raise ValueError("Signal too short for eye diagram.")

    # Time axis for 2 UI
    t_eye = np.linspace(0, UI_to_plot * bit_period, samples_per_period * UI_to_plot)

    # Focus on the central UI for window placement
    center_start = samples_per_period // 2
    center_end = center_start + samples_per_period
    x_center = t_eye[center_start:center_end]

    # Eye window: center in the middle of the UI
    eye_center_x = t_eye[center_start + samples_per_period // 2]
    eye_window_left = eye_center_x - eye_width_ps / 2
    eye_window_right = eye_center_x + eye_width_ps / 2

    # Find indices in the center UI that fall within the eye window
    window_mask = (x_center >= eye_window_left) & (x_center <= eye_window_right)
    window_indices = np.where(window_mask)[0]

    # For each time point in the window, get the distribution of amplitudes
    y_window = segments[:, center_start:center_end][:, window_indices]

    # Compute eye opening: difference between 99th and 1st percentile at each time point
    upper = np.percentile(y_window, 99, axis=0)
    lower = np.percentile(y_window, 1, axis=0)
    eye_openings = upper - lower  # in V

    # Eye height: minimum opening in the window (in V)
    min_eye_height = np.min(eye_openings)
    # Eye window height threshold (convert mV to V)
    eye_height_thresh = eye_height_mv / 1000.0

    # Labeling
    label = 'valid' if min_eye_height >= eye_height_thresh else 'invalid'

    # Plot if requested
    if plot_path is not None:
        plt.figure(figsize=(10, 6))
        for seg in segments:
            plt.plot(t_eye, seg, color='blue', alpha=0.1)
        # Draw eye window rectangle
        plt.gca().add_patch(
            plt.Rectangle(
                (eye_window_left, np.min(lower)),
                eye_width_ps,
                eye_height_thresh,
                fill=False, edgecolor='red', linewidth=2, linestyle='--', label='Eye Window'
            )
        )
        plt.title(f"Eye Diagram ({label.upper()})\nMin Eye Height: {min_eye_height*1000:.1f} mV")
        plt.xlabel("Time (ps)")
        plt.ylabel("Amplitude (V)")
        plt.grid(True, linestyle=':')
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()

    return label

if __name__ == "__main__":
    root = "data/prepared_data"
    batch_size = 1
    train_loader, _ = get_dataloaders(root, batch_size=batch_size, shuffle=True)
    batch = next(iter(train_loader))
    y = batch['y'][0].numpy()  # shape: (10000,)
    time = np.arange(len(y))
    label = eye_diagram_label(y, time)
    print(f"Eye diagram label for sample output signal: {label}")
