from rfdetr import RFDETRBase

def main():
    model = RFDETRBase(
        pretrain_weights="models/checkpoints/checkpoint_best_ema.pth",
        num_classes=29
    )
    model.export(output_dir="models/engines")
    print("[DONE] Exported ONNX model to models/engines/")

if __name__ == "__main__":
    main()
