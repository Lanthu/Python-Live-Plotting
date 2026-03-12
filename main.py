import os
os.environ["NIDAQMX_ENABLE_WAVEFORM_SUPPORT"] = "1"

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import nidaqmx
from nidaqmx.constants import AcquisitionType

### VARIABLES ###
# -----------------------------
# Settings
# -----------------------------
SAMPLING_RATE = 100          # Hz
CHUNK_SAMPLES = 50           # samples per read (0.5 s at 100 Hz)
PLOT_WINDOW_S = 10           # show last N seconds
TOTAL_SECONDS = 10           # stop after N seconds (or Ctrl+C)
# SYRINGE_DIAMETER = 26.26   # ony for (Harvard) syringe pump relevant
dt = 1.0 / SAMPLING_RATE


def main():
    ### SCRIPT ###
    # -----------------------------
    # Real-time plot setup
    # -----------------------------
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [])
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Voltage [V]")
    ax.set_title("Real-time DAQ")
    ax.grid(True)

    # -----------------------------
    # Storage (logging)
    # -----------------------------
    sample_index = 0
    plot_time = []
    plot_data = []
    all_data = []
    all_time_s = []

    # -----------------------------
    # DAQ task
    # -----------------------------
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan("Dev2/ai0")
        task.timing.cfg_samp_clk_timing(
            SAMPLING_RATE,
            sample_mode=AcquisitionType.CONTINUOUS
        )
        task.start()

        print("Running task. Press Ctrl+C to stop.")
        t_start = time.monotonic()

        try:
            while True:
                # Stop condition
                if time.monotonic() - t_start >= TOTAL_SECONDS:
                    break

                # Read one chunk as a waveform
                wavef = task.read_waveform(CHUNK_SAMPLES)
                data_chunk = np.asarray(wavef.scaled_data, dtype=float)

                n = data_chunk.size
                if n == 0:
                    # If nothing is available yet, yield briefly
                    time.sleep(0.001)
                    continue

                # Build time axis for this chunk (seconds)
                t_chunk = (sample_index + np.arange(n)) * dt
                sample_index += n

                # ---- Log everything ----
                all_data.extend(data_chunk.tolist())
                all_time_s.extend(t_chunk.tolist())

                # ---- Update plot buffer (sliding window) ----
                plot_data.extend(data_chunk.tolist())
                plot_time.extend(t_chunk.tolist())

                # Keep only last PLOT_WINDOW_S seconds for plotting speed
                t_min = plot_time[-1] - PLOT_WINDOW_S
                # find first index within window
                k = 0
                while k < len(plot_time) and plot_time[k] < t_min:
                    k += 1
                if k > 0:
                    plot_time = plot_time[k:]
                    plot_data = plot_data[k:]

                # Update line
                line.set_data(plot_time, plot_data)
                ax.set_xlim(max(0.0, plot_time[0]), plot_time[-1])

                # Autoscale y to current window (simple, stable-ish)
                y = np.asarray(plot_data)
                ypad = 0.05 * (y.max() - y.min() + 1e-12)
                ax.set_ylim(y.min() - ypad, y.max() + ypad)

                fig.canvas.draw()
                fig.canvas.flush_events()

                # Small pause to keep GUI responsive (adjust if needed)
                plt.pause(0.001)

        except KeyboardInterrupt:
            pass
        finally:
            task.stop()

    print(f"Logged {len(all_data)} samples.")

    # -----------------------------
    # Build DataFrame
    # -----------------------------
    all_data = np.asarray(all_data, dtype=float)
    all_time_s = np.asarray(all_time_s, dtype=float)

    df = pd.DataFrame({
        "time_s": all_time_s,
        "voltage_V": all_data
    })

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"Measurement_{timestamp}.csv"

    # df.to_csv(filename, index=False)


if __name__ == "__main__":
    main()
