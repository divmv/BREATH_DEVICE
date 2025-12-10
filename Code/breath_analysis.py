import os
import time
import machine

# --- CONFIGURATION ---
TARGET_ROWS    = 30000          
# The header MUST match what your Python script expects
DEFAULT_HDR    = "Samples,Xpos,Ypos,Pow,NebMode\n"
PROGRESS_EVERY = 5000           
BUFFER_SIZE    = 100            

# XIAO ESP32C6 Pin A1 (GPIO 1)
ADC_PIN_NUM = 1 
adc = machine.ADC(machine.Pin(ADC_PIN_NUM))
adc.atten(machine.ADC.ATTN_11DB)
adc.width(machine.ADC.WIDTH_12BIT)

def read_sensor_formatted():
    # 1. Read Actual Sensor
    val = adc.read()
    
    # 2. Get Timestamp
    ts = time.ticks_ms()
    
    # 3. Format for your Heavy Python Script
    # Python expects: row[1]=X, row[2]=Y, row[3]=Pow
    # Math: mag = sqrt(x^2 + y^2) / pow
    # We set x=val, y=0, pow=1. Result = val.
    
    # Format: Samples, Xpos, Ypos, Pow, NebMode
    return f"{ts},{val},0,1,0"

def run_analysis():
    print("[i] Starting Capture...")
    
    try:
        os.mkdir("/captures")
    except:
        pass 

    # We overwrite 'capture.csv' every time so we don't fill up memory
    out_path = "/captures/capture.csv"
    print(f"[i] Saving to: {out_path}")

    data_count = 0
    
    try:
        with open(out_path, "w") as f:
            f.write(DEFAULT_HDR)
            buffer = []
            
            while data_count < TARGET_ROWS:
                line = read_sensor_formatted()
                buffer.append(line)
                data_count += 1
                
                # Write in chunks to save RAM
                if len(buffer) >= BUFFER_SIZE:
                    f.write('\n'.join(buffer) + '\n')
                    buffer = [] 
                
                if data_count % PROGRESS_EVERY == 0:
                    print(f"[i] {data_count}/{TARGET_ROWS} captured")
            
            # Flush remaining
            if buffer:
                f.write('\n'.join(buffer) + '\n')
                
        return "Capture Complete. Ready to Upload."

    except Exception as e:
        print(f"[!] Error: {e}")
        return f"Error: {e}"