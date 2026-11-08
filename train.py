"""Train the full ESSA-Net model on a paired RGB/IR dataset."""
import argparse
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True, help='Path to paired dataset YAML')
    p.add_argument('--model', default=str(ROOT / 'configs/essa_net.yaml'))
    p.add_argument('--epochs', type=int, default=260)
    p.add_argument('--batch', type=int, default=12)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--device', default='0')
    args = p.parse_args()
    YOLO(args.model).train(data=args.data, epochs=args.epochs, batch=args.batch,
                           imgsz=args.imgsz, device=args.device, copy_paste=0.1,
                           close_mosaic=5)

if __name__ == '__main__':
    main()
