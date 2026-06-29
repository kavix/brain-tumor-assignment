import os
import sys

# Add project root to python path to import predict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from predict import main

if __name__ == '__main__':
    main()
