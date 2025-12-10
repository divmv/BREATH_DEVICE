
import pickle
import numpy as np

def load_trained_model(model_path="auto_trained_random_forest_model.pkl"):
    """Load the best auto-trained model for respiratory peak detection"""
    with open(model_path, 'rb') as f:
        model_data = pickle.load(f)
    return model_data['model'], model_data['performance']

def extract_features(signal_window):
    """Extract features from a signal window for prediction"""
    features = [
        np.mean(signal_window),
        np.std(signal_window),
        np.max(signal_window),
        np.min(signal_window),
        np.ptp(signal_window),
        np.median(signal_window),
        np.var(signal_window),
        np.sum(np.diff(signal_window) > 0),
        np.sum(np.diff(signal_window) < 0),
        np.argmax(signal_window) / len(signal_window),
    ]
    
    # Add frequency features
    fft_vals = np.abs(np.fft.fft(signal_window))[:len(signal_window)//2]
    features.extend([
        np.mean(fft_vals),
        np.std(fft_vals),
        np.argmax(fft_vals),
    ])
    
    return np.array(features).reshape(1, -1)

def predict_peak(signal_window, model_path="auto_trained_random_forest_model.pkl"):
    """Predict if a signal window contains a respiratory peak"""
    model, performance = load_trained_model(model_path)
    features = extract_features(signal_window)
    
    # Get prediction and probability
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0, 1]
    
    return {
        'is_peak': bool(prediction),
        'probability': float(probability),
        'model_performance': performance
    }

# Example usage:
if __name__ == "__main__":
    # Test with random signal
    test_signal = np.random.randn(1000)  # 1-second window at 1kHz
    result = predict_peak(test_signal)
    print(f"Peak detected: {result['is_peak']}, Probability: {result['probability']:.3f}")
