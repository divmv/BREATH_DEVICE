import io
import base64
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from flask import Flask, request, jsonify
from sklearn.decomposition import PCA, KernelPCA
from sklearn.cluster import KMeans, OPTICS
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter, find_peaks
from io import StringIO 

app = Flask(__name__)

def encode_plot_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

@app.route('/analyze', methods=['POST'])
def analyze():
    print("--- Analysis Request Received ---")
    if 'file' not in request.files:
        return jsonify({"error": "No file"}), 400

    file = request.files['file']
    filename = "temp_batch.csv"
    file.save(filename)

    try:
        # 1. ROBUST LOAD DATA
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Filter bad lines
        valid_lines = [line for line in lines if line.count(',') == 4]
        if len(valid_lines) < 10: return jsonify({"error": "Not enough valid data"}), 400

        csv_data = "".join(valid_lines)
        df = pd.read_csv(StringIO(csv_data))
        df = df.apply(pd.to_numeric, errors='coerce').dropna()
        
        # 2. EXTRACT & CALCULATE SIGNAL
        # Header: T,X,Y,P,N
        t_raw = df.iloc[:, 0].values
        x_raw = df.iloc[:, 1].values
        y_raw = df.iloc[:, 2].values
        p_raw = df.iloc[:, 3].values
        
        # Normalize Time
        t_raw = (t_raw - t_raw[0]) / 1000.0

        # Calculate "Breath Signal" (Magnitude / Power)
        # Avoid div by zero
        p_safe = np.where(p_raw == 0, 1, p_raw)
        magnitude = np.sqrt(x_raw**2 + y_raw**2)
        breath_signal = magnitude / p_safe

        # 3. FILTERING
        win_len = min(51, len(breath_signal))
        if win_len % 2 == 0: win_len -= 1
        filtered = savgol_filter(breath_signal, window_length=win_len, polyorder=3)

        # 4. PEAK DETECTION
        # Dynamic distance based on signal length (approx 0.5s breathing assumption)
        peaks, _ = find_peaks(filtered, distance=10, prominence=0.5)

        # --- GENERATE PLOTS ---
        
        # Plot 1: Raw vs Filtered (The "First Ever" Plot)
        fig1, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t_raw, breath_signal, color='lightgray', label='Raw Signal')
        ax.plot(t_raw, filtered, color='#2196F3', label='Filtered', linewidth=2)
        ax.set_title("Signal Processing (Raw vs Filtered)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_signal = encode_plot_to_base64(fig1)

        # Plot 2: Peak Detection
        fig2, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t_raw, filtered, color='#4CAF50', label='Breath Signal')
        ax.plot(t_raw[peaks], filtered[peaks], "rx", label='Peaks', markersize=10)
        ax.set_title(f"Peak Detection (Count: {len(peaks)})")
        ax.set_xlabel("Time (s)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_peaks = encode_plot_to_base64(fig2)

        # --- ML PREP (Downsample) ---
        step = 10
        # Use raw features X, Y, P for PCA/Clustering structure
        X_ml = np.column_stack((x_raw[::step], y_raw[::step], p_raw[::step]))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_ml)
        t_color = t_raw[::step]

        # Plot 3: Linear PCA (3D)
        pca = PCA(n_components=3)
        X_pca = pca.fit_transform(X_scaled)
        
        fig3 = plt.figure(figsize=(8, 6))
        ax = fig3.add_subplot(111, projection='3d')
        ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c=t_color, cmap='plasma', s=20)
        ax.set_title("Linear PCA (3 Components)")
        plot_pca_lin = encode_plot_to_base64(fig3)

        # Plot 4: Non-Linear PCA (Kernel)
        if len(X_scaled) > 1000:
            idx = np.random.choice(len(X_scaled), 1000, replace=False)
            X_kpca_input = X_scaled[idx]
            t_kpca_color = t_color[idx]
        else:
            X_kpca_input = X_scaled
            t_kpca_color = t_color

        kpca = KernelPCA(n_components=3, kernel='rbf', gamma=0.1)
        X_kpca = kpca.fit_transform(X_kpca_input)

        fig4 = plt.figure(figsize=(8, 6))
        ax = fig4.add_subplot(111, projection='3d')
        ax.scatter(X_kpca[:,0], X_kpca[:,1], X_kpca[:,2], c=t_kpca_color, cmap='viridis', s=20)
        ax.set_title("Non-Linear Kernel PCA")
        plot_pca_nonlin = encode_plot_to_base64(fig4)

        # Plot 5: K-Means (K=3)
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        labels_km = kmeans.fit_predict(X_scaled)

        fig5 = plt.figure(figsize=(8, 6))
        ax = fig5.add_subplot(111, projection='3d')
        ax.scatter(X_scaled[:,0], X_scaled[:,1], X_scaled[:,2], c=labels_km, cmap='Set1', s=20)
        ax.set_title("K-Means Clustering (K=3)")
        plot_km = encode_plot_to_base64(fig5)

        # Plot 6: OPTICS
        optics = OPTICS(min_samples=20, xi=0.05, min_cluster_size=0.1)
        labels_opt = optics.fit_predict(X_scaled)

        fig6 = plt.figure(figsize=(8, 6))
        ax = fig6.add_subplot(111, projection='3d')
        ax.scatter(X_scaled[:,0], X_scaled[:,1], X_scaled[:,2], c=labels_opt, cmap='tab10', s=20)
        ax.set_title("OPTICS Clustering (Density)")
        plot_opt = encode_plot_to_base64(fig6)

        # Cleanup
        if os.path.exists(filename): os.remove(filename)

        return jsonify({
            "metrics": {
                "Samples": len(df),
                "Duration": f"{t_raw[-1]:.1f}s",
                "Peaks Detected": len(peaks)
            },
            "plot_signal": plot_signal,
            "plot_peaks": plot_peaks,
            "plot_pca_l": plot_pca_lin,
            "plot_pca_nl": plot_pca_nonlin,
            "plot_km": plot_km,
            "plot_opt": plot_opt
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)