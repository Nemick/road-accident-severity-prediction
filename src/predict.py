import pandas as pd
import joblib
import argparse
import sys

def load_model(model_path):
    """Load a trained ML model from disk."""
    return joblib.load(model_path)

def predict_severity(model, input_data):
    """Predict accident severity from input data.

    Args:
        model: Trained pipeline/model object
        input_data: Dict or DataFrame with feature values
    """
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    return prediction[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict accident severity.")
    parser.add_argument("--model", required=True, help="../model/accident_severity_pipeline_merged.joblib")
    parser.add_argument("--features", nargs='+', required=True, help="Feature values as key=value pairs")
    args = parser.parse_args()

    # Convert features to dictionary
    feature_dict = {}
    for feature in args.features:
        if "=" not in feature:
            print(f"Invalid feature format: {feature}. Use key=value.", file=sys.stderr)
            sys.exit(1)
        key, value = feature.split("=", 1)
        try:
            value = float(value)
        except ValueError:
            pass
        feature_dict[key] = value

    # Load model
    model = load_model(args.model)

    # Make prediction
    try:
        severity = predict_severity(model, feature_dict)
        print(f"Predicted Accident Severity: {severity}")
    except Exception as e:
        print(f"Prediction failed: {e}", file=sys.stderr)
        sys.exit(1)