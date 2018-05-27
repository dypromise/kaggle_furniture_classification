import argparse
import os
import utils
import fur_model

model_names = sorted(name for name in fur_model.model_dict.keys())
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--checkpoint-file', default='/home/dingyang/best_val_weights.pth', type=str,
                    help='checkpoint file path (default: /home/dingyang/best_val_weights.pth)')
parser.add_argument('--model-name', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--mode', default='train', type=str,
                    help='train mode or test mode')
parser.add_argument('--input-size', default=224, type=int,
                    help='net input size (default: 224)')
parser.add_argument('--add-size', default=32, type=int,
                    help='net add size (default: 32)')
parser.add_argument('--test-prob', default=False, type=bool,
                    help='test output prob (default: False)')
parser.add_argument('--test-predsfile', default='test_predictions.csv', type=str,
                    help='train mode or test mode')


best_prec1 = 0


def main():

    global args, best_prec1
    args = parser.parse_args()

    train_dir = os.path.join(args.data, 'train_ori')
    val_dir = os.path.join(args.data, 'valid')
    test_dir = os.path.join(args.data, 'test')

    train_csv = os.path.join(args.data, 'train.csv')
    val_csv = os.path.join(args.data, 'val.csv')
    test_csv = os.path.join(args.data, 'test.csv')

    test_whole_file = os.path.join(args.data, 'whole_test.txt')
    whole_prediction_csv = os.path.join(args.data, 'new_test_predictions.csv')

    furniture_model = fur_model.DY_Model(model_name=args.model_name, num_classes=fur_model.NB_CLASSES,
                                         checkpoint_file=args.checkpoint_file,
                                         KFolds=1,
                                         batch_size=args.batch_size,
                                         input_size=args.input_size,
                                         add_size=args.add_size
                                         )

    if(args.mode == 'train'):
        furniture_model.train_single_model(
            train_dir, train_csv, val_dir, val_csv, args.epochs)

    elif(args.mode == 'test'):
        furniture_model.test_single_model(
            args.checkpoint_file, val_dir, val_csv, prediction_file_path=args.test_predsfile, ten_crop=True, prob=args.test_prob)

        # print("complement prediction...")
        # utils.complement_prediction(
        #     test_whole_file, preds_csv=prediction_file, new_csv=whole_prediction_csv)


if __name__ == "__main__":
    main()
