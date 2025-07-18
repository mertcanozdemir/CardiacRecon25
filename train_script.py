import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataloader import CMRKspaceWithMaskDataset
from model import UNET
from helper_functions import EarlyStopping

NUM_EPOCHS = 100
BATCH_SIZE = 1

device = torch.device("cuda" if torch.mps.is_available() else "cpu")


def show_results(undersampled, output, ground_truth):
    """
    undersampled, output, ground_truth: torch.Tensor, shape = [1, H, W] (single channel)
    Görselleri normalize edip 1x3 grid olarak gösterir.
    """

    # Tensorleri numpy array’e çevir ve squeeze ile boyutları düşür
    undersampled = undersampled.squeeze().cpu().numpy()
    output = output.squeeze().cpu().detach().numpy()
    ground_truth = ground_truth.squeeze().cpu().numpy()

    # Normalize et (0-1 aralığına)
    def normalize_img(img):
        img_min = img.min()
        img_max = img.max()
        return (img - img_min) / (img_max - img_min + 1e-8)

    undersampled = normalize_img(undersampled)
    output = normalize_img(output)
    ground_truth = normalize_img(ground_truth)

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(undersampled, cmap='gray')
    axs[0].set_title("Undersampled (input)")
    axs[0].axis('off')

    axs[1].imshow(output, cmap='gray')
    axs[1].set_title("Output (trained model)")
    axs[1].axis('off')

    axs[2].imshow(ground_truth, cmap='gray')
    axs[2].set_title("Ground Truth")
    axs[2].axis('off')

    plt.show()


def train():
    dataset = CMRKspaceWithMaskDataset(root_dir="cmr_dataset", mask_type="Gaussian", kt=8)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    early_stopping = EarlyStopping(monitor='loss', patience=20, verbose=True, mode='min')
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

    model.train()
    for epoch in range(NUM_EPOCHS):
        train_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in loop:
            inputs = batch['zero_filled'].to(device)
            targets = batch['gt_image'].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_loss = train_loss / len(dataloader)
        if epoch % 5 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Loss: {avg_loss:.6f}")

        # Scheduler step
        scheduler.step()

        # Early stopping kontrolü
        early_stopping.on_epoch_end(avg_loss, model)
        if early_stopping.stop_training:
            print(f"Training stopped early at epoch {epoch+1}")
            break

    # Eğitim bittikten sonra ilk batch'i tekrar alıp görselleri gösterelim
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(dataloader))
        input_img = sample_batch['zero_filled'].to(device)
        target_img = sample_batch['gt_image'].to(device)
        output_img = model(input_img)

        show_results(input_img[0], output_img[0], target_img[0])


model = UNET().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)


if __name__ == "__main__":
    train()
