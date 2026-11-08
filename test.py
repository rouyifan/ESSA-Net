"""Evaluate an ESSA-Net checkpoint on a paired RGB/IR split."""
import argparse
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--split', choices=('val', 'test'), default='test')
    p.add_argument('--batch', type=int, default=12)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='0')
    args = p.parse_args()
    YOLO(args.weights).val(data=args.data, split=args.split, batch=args.batch,
                           imgsz=args.imgsz, device=args.device)

if __name__ == '__main__':
    main()
