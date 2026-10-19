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
SAMPLING_RATE = 100          # Rate of sampling of the voltage measurements into buffer (Hz)
CHUNK_SAMPLES = 50           # Samples per read retrieved from buffer (0.5 s at 100 Hz)
PLOT_WINDOW_S = 10          # show last N seconds
TOTAL_SECONDS = 270          # stop after N seconds (or Ctrl+C)
# Pressure conversion factors for the specific sensor up to 4 bar
# PRESSURE_CONVERSION_FACTOR = 5054.974227  # Convert voltage to pressure
# PRESSURE_CONVERSION_FACTOR = 299.27  # Convert voltage to pressure in mmHg
# Pressure conversion factors for the specific sensor up to 0.5 bar
# PRESSURE_CONVERSION_FACTOR = 37.916  # Convert voltage to pressure in mmHg
# Pressure conversion factor unity
PRESSURE_CONVERSION_FACTOR = 1.0
# Pressure conversion factors for sensor 0..4bar 09.10.2026
# PRESSURE_CONVERSION_FACTOR = 4092.4  # Convert voltage to pressure in mmH2O
# VOLUME_FLOW_CONVERSION_FACTOR_SLOPE = 5100.3026  # Convert voltage to volume flow (ml/min)
# VOLUME_FLOW_CONVERSION_FACTOR_INTERCEPT = -2398.8386  # Intercept for volume flow conversion (ml/min)
# Volume flow conversion factors for the specific sensor Sonoflow V3.0 (ml/min)
# VOLUME_FLOW_CONVERSION_FACTOR_SLOPE = 1364.5  # Convert voltage to volume flow (ml/min)
# VOLUME_FLOW_CONVERSION_FACTOR_INTERCEPT = 585.6  # Intercept for volume flow conversion (ml/min)
# Volume flow conversion factors unity
VOLUME_FLOW_CONVERSION_FACTOR_SLOPE = 1.0  # Convert voltage to volume flow (ml/min)
VOLUME_FLOW_CONVERSION_FACTOR_INTERCEPT = 0.0  # Intercept for volume flow conversion (ml/min)
# LOWER_Y_LIM_FLOW = 0
# UPPER_Y_LIM_FLOW = 3000
# VOLUME_FLOW_CONVERSION_FACTOR_INTERCEPT_VOLTAGE = 0.471237  # Voltage corresponding to zero flow (ml/min)
# SYRINGE_DIAMETER = 26.26   # ony for (Harvard) syringe pump relevant
dt = 1.0 / SAMPLING_RATE


def main():
    ### SCRIPT ###
    # -----------------------------
    # Real-time plot setup
    # -----------------------------
    plt.ion()
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
    
    # Subplot 1: absolute difference of ai1 and ai2 (in pressure)
    line_diff, = ax1.plot([], [], label='|P_ai1 - P_ai2|', color='purple')
    ax1.set_xlabel("Time [s]")
    ax1.set_ylabel("Pressure [mmHg]")
    ax1.set_title("Absolute Difference of Pressure (|ai1 - ai2|)")
    ax1.grid(True)
    ax1.legend()
    
    # Subplot 2: ai3 (voltage)
    line_ch2, = ax2.plot([], [], label='ai3', color='green')
    ax2.set_xlabel("Time [s]")
    ax2.set_ylabel("Volume Flow [ml/min]")
    ax2.set_title("Channel ai3")
    ax2.grid(True)
    ax2.legend()
    
    # Subplot 3: original ai1 and ai2 (voltage, for debugging)
    line_ch0_debug, = ax3.plot([], [], label='ai1 (voltage)', color='blue')
    line_ch1_debug, = ax3.plot([], [], label='ai2 (voltage)', color='red')
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Voltage [V]")
    ax3.set_title("Original Voltages (Debug)")
    ax3.grid(True)
    ax3.legend()

    # -----------------------------
    # Storage (logging)
    # -----------------------------
    sample_index = 0
    plot_time = []
    plot_data_ch0 = []
    plot_data_ch1 = []
    plot_data_ch2 = []
    plot_data_ch0_pressure = []
    plot_data_ch1_pressure = []
    plot_data_ch2_flow = []
    plot_data_diff = []
    all_data_ch0 = []
    all_data_ch1 = []
    all_data_ch2 = []
    all_time_s = []

    # -----------------------------
    # DAQ task
    # -----------------------------
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan("Dev2/ai1")
        task.ai_channels.add_ai_voltage_chan("Dev2/ai2")
        task.ai_channels.add_ai_voltage_chan("Dev2/ai3")
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

                # Read one chunk as waveforms (list of waveforms, one per channel)
                waveforms = task.read_waveform(CHUNK_SAMPLES)
                data_chunk_ch0 = np.asarray(waveforms[0].scaled_data, dtype=float)
                data_chunk_ch1 = np.asarray(waveforms[1].scaled_data, dtype=float)
                data_chunk_ch2 = np.asarray(waveforms[2].scaled_data, dtype=float)

                n = data_chunk_ch0.size
                if n == 0:
                    # If nothing is available yet, yield briefly
                    time.sleep(0.001)
                    continue

                # Build time axis for this chunk (seconds)
                t_chunk = (sample_index + np.arange(n)) * dt
                sample_index += n

                # ---- Log everything (raw voltage) ----
                all_data_ch0.extend(data_chunk_ch0.tolist())
                all_data_ch1.extend(data_chunk_ch1.tolist())
                all_data_ch2.extend(data_chunk_ch2.tolist())
                all_time_s.extend(t_chunk.tolist())

                # ---- Convert to pressure ----
                data_chunk_ch0_pressure = data_chunk_ch0 * PRESSURE_CONVERSION_FACTOR
                data_chunk_ch1_pressure = data_chunk_ch1 * PRESSURE_CONVERSION_FACTOR

                # ---- Convert to volume flow ----
                data_chunk_ch2_flow = data_chunk_ch2 * VOLUME_FLOW_CONVERSION_FACTOR_SLOPE + VOLUME_FLOW_CONVERSION_FACTOR_INTERCEPT

                # ---- Update plot buffer (sliding window) ----
                plot_data_ch0.extend(data_chunk_ch0.tolist())
                plot_data_ch1.extend(data_chunk_ch1.tolist())
                plot_data_ch2.extend(data_chunk_ch2.tolist())
                plot_data_ch0_pressure.extend(data_chunk_ch0_pressure.tolist())
                plot_data_ch1_pressure.extend(data_chunk_ch1_pressure.tolist())
                plot_data_ch2_flow.extend(data_chunk_ch2_flow.tolist())

                # Calculate absolute difference for this chunk (in pressure)
                data_chunk_diff = np.abs(data_chunk_ch0_pressure - data_chunk_ch1_pressure)
                plot_data_diff.extend(data_chunk_diff.tolist())
                
                plot_time.extend(t_chunk.tolist())

                # Keep only last PLOT_WINDOW_S seconds for plotting speed
                t_min = plot_time[-1] - PLOT_WINDOW_S
                # find first index within window
                k = 0
                while k < len(plot_time) and plot_time[k] < t_min:
                    k += 1
                if k > 0:
                    plot_time = plot_time[k:]
                    plot_data_ch0 = plot_data_ch0[k:]
                    plot_data_ch1 = plot_data_ch1[k:]
                    plot_data_ch2_flow = plot_data_ch2_flow[k:]
                    plot_data_ch0_pressure = plot_data_ch0_pressure[k:]
                    plot_data_ch1_pressure = plot_data_ch1_pressure[k:]
                    plot_data_diff = plot_data_diff[k:]

                # Update subplot 1: absolute difference (pressure)
                line_diff.set_data(plot_time, plot_data_diff)
                ax1.set_xlim(max(0.0, plot_time[0]), plot_time[-1])
                y_diff = np.asarray(plot_data_diff)
                ypad_diff = 0.05 * (y_diff.max() - y_diff.min() + 1e-12)
                ax1.set_ylim(y_diff.min() - ypad_diff, y_diff.max() + ypad_diff)

                # Update subplot 2: ai3 (voltage)
                line_ch2.set_data(plot_time, plot_data_ch2_flow)
                ax2.set_xlim(max(0.0, plot_time[0]), plot_time[-1])

                # Comment in the following lines to set dynamic y-limits for volume flow plot
                y_ch2 = np.asarray(plot_data_ch2_flow)
                ypad_ch2 = 0.05 * (y_ch2.max() - y_ch2.min() + 1e-12)
                ax2.set_ylim(y_ch2.min() - ypad_ch2, y_ch2.max() + ypad_ch2)

                # Comment in the following line to set fixed y-limits for volume flow plot
                # ax2.set_ylim(LOWER_Y_LIM_FLOW, UPPER_Y_LIM_FLOW)

                # Update subplot 3: original ai1 and ai2 (voltage, for debugging)
                line_ch0_debug.set_data(plot_time, plot_data_ch0)
                line_ch1_debug.set_data(plot_time, plot_data_ch1)
                ax3.set_xlim(max(0.0, plot_time[0]), plot_time[-1])
                y_ch0 = np.asarray(plot_data_ch0)
                y_ch1 = np.asarray(plot_data_ch1)
                y_debug = np.concatenate([y_ch0, y_ch1])
                ypad_debug = 0.05 * (y_debug.max() - y_debug.min() + 1e-12)
                ax3.set_ylim(y_debug.min() - ypad_debug, y_debug.max() + ypad_debug)

                fig.canvas.draw()
                fig.canvas.flush_events()

                # Small pause to keep GUI responsive (adjust if needed)
                plt.pause(0.001)

        except KeyboardInterrupt:
            pass
        finally:
            task.stop()

    # print(f"Logged {len(all_data)} samples.")

    # -----------------------------
    # Build DataFrame
    # -----------------------------
    all_data_ch0 = np.asarray(all_data_ch0, dtype=float)
    all_data_ch1 = np.asarray(all_data_ch1, dtype=float)
    all_data_ch2 = np.asarray(all_data_ch2, dtype=float)
    all_time_s = np.asarray(all_time_s, dtype=float)

    df = pd.DataFrame({
        "time_s": all_time_s,
        "voltage_V_ai0": all_data_ch0,
        "voltage_V_ai1": all_data_ch1,
        "voltage_V_ai2": all_data_ch2
    })

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    filename = f"Measurement_{timestamp}.csv"

    df.to_csv(filename, index=False)


if __name__ == "__main__":
    main()
