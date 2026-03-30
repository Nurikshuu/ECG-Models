# src/infer.py
import argparse
# load checkpoint, load example signals, output probs and plots


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True, help='Path to input signal file')
    parser.add_argument('--output', default='output.png', help='Path to save output plot')
    args = parser.parse_args()
    # TODO: load model, run inference, save results
