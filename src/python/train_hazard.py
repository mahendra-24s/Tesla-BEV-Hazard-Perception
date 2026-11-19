from rfdetr import RFDETRBase

def main():
    model = RFDETRBase()
    model.train(
        dataset_dir="data/coda",
        epochs=25,
        batch_size=4,
        grad_accum_steps=4,
        lr=1e-4,
        resolution=560,
        multi_scale=False,
        num_workers=2,
        progress_bar=True,
        output_dir="models/checkpoints"
    )

if __name__ == "__main__":
    main()
