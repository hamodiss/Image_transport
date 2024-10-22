import sys

def main():
    # Get the object label passed as an argument
    if len(sys.argv) > 1:
        object_detected = sys.argv[1]
    else:
        object_detected = "Unknown Object"

    # Simulate sending a warning to the train
    print(f"WARNING: {object_detected} detected in the restricted zone! Sending signal to the train...")

if __name__ == "__main__":
    main()
