import argparse
import json
import os

import matplotlib.pyplot as plt


def main(model_dir, chunk_size):
    submodel_dir = f'{model_dir}/{chunk_size}'
    eval_dir = os.path.join(submodel_dir,'evaluate/')
    checkpoints = [int(file.split('_')[0]) for file in os.listdir(eval_dir) if file.endswith('_rouge.json')]
    checkpoints.sort()

    prec = []
    recall = []
    f1= []
    for file_num in checkpoints:
        print(file_num)
        filename = f'{eval_dir}{file_num}_rouge.json'
        with open(filename, 'r') as file:
            data = json.load(file) 
        prec.append(data['rouge1'][0][0])
        recall.append(data['rouge1'][0][1])
        f1.append(data['rouge1'][0][2])

    # Plot
    plt.clf()
    # Create subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    last_point_idx = len(checkpoints)-1
    # Plot Precision
    axs[0].plot(checkpoints, prec, marker='o', linestyle='-', color='b')
    axs[0].set_title('Precision')
    axs[0].set_xlabel('Checkpoints')
    axs[0].set_ylabel('Precision')
    axs[0].set_ylim(0, 1)
    for i, txt in enumerate(prec):
        if i%3 == 0 or i == last_point_idx:
            axs[0].annotate(f'{txt:.2f}', (checkpoints[i], prec[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # Plot Recall
    axs[1].plot(checkpoints, recall, marker='o', linestyle='-', color='g')
    axs[1].set_title('Recall')
    axs[1].set_xlabel('Checkpoints')
    axs[1].set_ylabel('Recall')
    axs[1].set_ylim(0, 1)
    for i, txt in enumerate(recall):
        if i%3 == 0 or i == last_point_idx:
            axs[1].annotate(f'{txt:.2f}', (checkpoints[i], recall[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # Plot F1
    axs[2].plot(checkpoints, f1, marker='o', linestyle='-', color='r')
    axs[2].set_title('F1 Score')
    axs[2].set_xlabel('Checkpoints')
    axs[2].set_ylabel('F1 Score')
    axs[2].set_ylim(0, 1)
    for i, txt in enumerate(f1):
        if i%3 == 0 or i == last_point_idx:
            axs[2].annotate(f'{txt:.2f}', (checkpoints[i], f1[i]), textcoords="offset points", xytext=(0,10), ha='center')

    # Adjust layout and add the main title
    plt.suptitle(f'Rouge 1 Score for Reconstruction: Sequence Length {chunk_size}', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Save the plot as a PNG file
    image_path=eval_dir + '/rouge_plot.png'
    plt.savefig(image_path)

    # Show plot (optional, can be omitted if running in a non-GUI environment)
    plt.show()

    # Clear the plot
    plt.clf()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--chunk_size", type=str, required=True, help="Sequence Length used for model training")
    args = parser.parse_args()
    main(args.model_dir,args.chunk_size)