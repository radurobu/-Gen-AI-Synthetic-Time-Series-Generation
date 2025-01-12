# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 20:17:13 2024

@author: robur
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json


# Parameters
num_waves = 5          # Number of waves to generate
num_samples = 25000     # Number of samples to generate
duration = 10.0        # Duration in seconds
sampling_rate = 30     # Samples per second

# Generate time values
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Create an empty DataFrame to store the waves
df = []


# Itterate through num samples
# for n in range(num_samples):
    
#     amplitude = np.random.uniform(0.5, 2.0)   # Random amplitude between 0.5 and 2.0
#     frequency = np.random.uniform(0.1, 3)     # Random frequency between 0.5 Hz and 5.0 Hz
    
#     # Generate and save each wave in the DataFrame
#     df_aux = []
    
#     for i in range(num_waves):
    
#         # Generate random amplitude and frequency for each wave
#         phase = np.random.uniform(1, 10)           # Phase shift in radians
#         slope = np.random.uniform(-1, 1)         # Linear slope component
    
#         # Generate the sinusoidal wave with a linear component
#         wave = amplitude * np.sin(2 * np.pi * frequency * t + phase) + slope * t
    
#         # Add the wave to the DataFrame
#         df_aux.append(list(wave))
        
#     df.append(df_aux)

for n in range(num_samples):
    
    amplitude = np.random.uniform(0.5, 2.0)   # Random amplitude between 0.5 and 2.0
    frequency = np.random.uniform(0.1, 3)     # Random frequency between 0.5 Hz and 5.0 Hz
    
    # Generate and save each wave in the DataFrame
    df_aux = []
    
    for i in range(2):
    
        # Generate random amplitude and frequency for each wave
        phase = 0 #np.random.uniform(1, 10)           # Phase shift in radians
        slope = np.random.uniform(0, 1)         # Linear slope component
    
        # Generate the sinusoidal wave with a linear component
        wave = amplitude * np.sin(2 * np.pi * frequency * t + phase) + slope * t
    
        # Add the wave to the DataFrame
        df_aux.append(list(wave))
        
    for i in range(2):
    
        # Generate random amplitude and frequency for each wave
        phase = 0 #np.random.uniform(1, 10)           # Phase shift in radians
        slope = np.random.uniform(-1, 0)         # Linear slope component
    
        # Generate the sinusoidal wave with a linear component
        wave = amplitude * np.sin(2 * np.pi * frequency * t + phase) + slope * t
    
        # Add the wave to the DataFrame
        df_aux.append(list(wave))
        
    for i in range(1):
    
        # Generate random amplitude and frequency for each wave
        phase = 0 #np.random.uniform(1, 10)           # Phase shift in radians
        slope = 0 #np.random.uniform(-1, 0)         # Linear slope component
    
        # Generate the sinusoidal wave with a linear component
        wave = amplitude * np.sin(2 * np.pi * frequency * t + phase) + slope * t
    
        # Add the wave to the DataFrame
        df_aux.append(list(wave))
        
        
    df.append(df_aux)


# Plot all waves from the DataFrame
plt.figure(figsize=(12, 6))

plt.title('Multiple Sinusoidal Waves with Random Frequency and Amplitude')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
for i in range(0, num_waves):
    plt.plot(t, df[0][i], label=f'Wave {i}')

# Configure the plot
plt.grid(True)
plt.legend()
plt.show()

# Save array
with open('C:/Radu/ML_Projects/Synthetic Time Series/tts-gan/data/sine_waves.json', 'w') as fp:
    json.dump(df, fp)
    
    

