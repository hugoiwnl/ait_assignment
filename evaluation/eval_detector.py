
import sys
import os
import detector_evaluation as de

if __name__ == '__main__':
    # check number of arguments
    if len(sys.argv) != 2:
        print("Usage: python eval_detector.py results_path")
        sys.exit(1)
    
    results_path = sys.argv[1]
    # check if results_path exists
    if not os.path.exists(results_path):
        print("Invalid results path.")
        sys.exit(1)
    
    result = de.compute_repeatability_HPatches(results_path, 300, 3)
    print(result)

