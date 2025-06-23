import torch
from tqdm.notebook import tqdm
import torch.nn.functional as F
from plotting import plot_images

import os


def train_img_fitting(
    model,
    optimizer,
    img_fitting,
    device=None,
    epochs=300,
    batch_size=4096,
    epochs_til_summary=30,
    model_save_path=None,
    checkpoint_save=False,
    plots_dir=None
    ):
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    img_fitting = img_fitting.to(device)

    # Setup tracking variables
    reconstructed = img_fitting.create_reconstruction_tensor().to(device)
    gt = img_fitting.pixels

    psnr_vals = []
    loss_history = []

    # Training loop
    model.train()
    for epoch in tqdm(range(epochs+1)):
        epoch_loss = 0.0

        # Get batches of indices
        batches = img_fitting.get_batch_indices(batch_size)

        for batch_indices in batches:
            batch_indices = batch_indices

            # Get coordinates for this batch
            batch_data = img_fitting.get_items_by_indices(batch_indices)
            b_coords = batch_data['coords']

            # Forward pass
            model_output, _ = model(b_coords)

            # Update reconstruction
            with torch.no_grad():
                reconstructed[batch_indices] = model_output

            # Calculate loss
            loss = F.mse_loss(model_output, gt[batch_indices])
            epoch_loss += loss.item() * len(batch_indices)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # Calculate metrics
        avg_loss = epoch_loss / len(gt)
        loss_history.append(avg_loss)

        with torch.no_grad():
            psnr_val = -10 * torch.log10(F.mse_loss(reconstructed, gt))
            psnr_vals.append(psnr_val.item())

        # Visualize results after each epoch
        if epoch % epochs_til_summary == 0:
            print(f"Epoch: {epoch} | Loss: {avg_loss:.5f} | PSNR: {psnr_val:.4f} | Normalized: {img_fitting.normalization}")
            
            suptitle = f"Epochs={epoch} | PSNR={psnr_val:.4f}"
            images = []
            titles = []
            img_rec = img_fitting.reconstruction_to_image(reconstructed.detach().cpu())

            images.append(img_fitting.img_tensor.permute(1, 2, 0).numpy())
            titles.append("Original")
            images.append(img_rec.numpy())
            titles.append("Reconstructed")

            if img_fitting.normalization:
                images.append(img_fitting.normalized_tensor.numpy().clip(0, 1))
                titles.append("Normalized")
                images.append(img_fitting.denormalize(img_rec).numpy())
                titles.append("Reconstruction Denormalized")

            save_path = None
            if plots_dir:
                save_path=os.path.join(plots_dir, f"epoch_{epoch}.png")

            plot_images(images, suptitle=suptitle, titles=titles, figsize=(20, 10), save_path=save_path)

            # Save the trained model
            if checkpoint_save:
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'psnr_history': psnr_vals,
                    'loss_history': loss_history,
                    'epochs': epochs
                }, model_save_path)
        
    return {
        'psnr_history': psnr_vals,
        'loss_history': loss_history,
        'final_psnr': psnr_vals[-1] if psnr_vals else None,
        # 'model_save_path': save_path,
        'reconstructed_image': img_fitting.reconstruction_to_image(reconstructed.detach().cpu()),
        'model': model,
        'optimizer': optimizer
    }