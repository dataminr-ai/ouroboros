from ouroboros.evaluate import main as eval
import os
import argparse

def main(base_model, eval_file, chunk_size, batch_size, model_dir):
    
    base_decoder=os.path.join(model_dir, 'decoder')
    trained_decoder=os.path.join(model_dir, str(chunk_size))
    output_dir=os.path.join(trained_decoder, 'evaluate')

    os.makedirs(output_dir, exist_ok=True)

    print("Evaluating Base Model")
    eval(base_model, eval_file, chunk_size, batch_size, output_dir, base_decoder)

    for dir in os.listdir(trained_decoder):
        if dir.startswith("step_"):
            print(f"Evaluating {dir}")
            step_path = os.path.join(trained_decoder, dir)
            eval(base_model, eval_file, chunk_size, batch_size, output_dir, step_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference script for a pre-trained model"
    )
    parser.add_argument(
        "--base_model", type=str, required=True, help="Base model name or path"
    )
    parser.add_argument(
        "--eval_file", type=str, required=True, help="Data for inference (.jsonl)"
    )
    parser.add_argument("--chunk_size", type=int, required=True, help="Sequence Length")
    parser.add_argument("--batch_size", type=int, required=True, help="Sequence Length")
    parser.add_argument(
        "--model_dir", type=str, required=False, help="Path to model"
    )

    args = parser.parse_args()

    main(
        args.base_model,
        args.eval_file,
        args.chunk_size,
        args.batch_size,
        args.model_dir,
    )


