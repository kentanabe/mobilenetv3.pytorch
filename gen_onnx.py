import argparse
import torch

def main():
    parser = argparse.ArgumentParser(description='Generate MobileNetV3 ONNX model.')
    parser.add_argument(
        '--output_file', type=str, default='mobilenetv3_large.onnx',
        help='output ONNX model file name')
    parser.add_argument(
        '--input_size', type=int, default=[10, 3, 224, 224], nargs=4,
        help='input size')
    parser.add_argument('--model', '-m', metavar='MODEL', default='large',
                        help='model size (default: large)')
    args = parser.parse_args()
    if args.model == 'large':
        from mobilenetv3 import mobilenetv3_large
        model = mobilenetv3_large()
        model.load_state_dict(torch.load('pretrained/mobilenetv3-large-1cd25616.pth'))
    else:
        from mobilenetv3 import mobilenetv3_small
        model = mobilenetv3_small()
        model.load_state_dict(torch.load('pretrained/mobilenetv3-small-55df8e1f.pth'))

    input_size = args.input_size
    output_file = args.output_file
    input_names = ["data"]
    output_names = ["classifier"]

    dummy_input = torch.randn(input_size[0],
                              input_size[1],
                              input_size[2],
                              input_size[3])
    torch.onnx.export(model, dummy_input, output_file, 
        input_names=input_names, output_names=output_names, verbose=False)

if __name__ == '__main__':
    main()
