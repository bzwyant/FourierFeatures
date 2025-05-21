import torch
from tqdm.notebook import tqdm
import torch.nn.functional as F
import matplotlib.pyplot as plt

import os


def train_inr(
    model,
    optimizer,
    img_fitting,
    device=None,
    epochs=300,
    batch_size=4096,
    steps_til_summary=30,
    save_dir=None,
):
    # Setup device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    img_fitting = img_fitting.to(device)

    # Setup save directory
    if save_dir is None:
        hidden_layers = getattr(model, 'hidden_layers', 'unknown')
        hidden_dim = getattr(model, 'hidden_features', 'unknown')
        model_name = type(model).__name__
        if model_name == 'HybridSiren':
            siren_layers = getattr(model, 'siren_layers', 'unknown')
            save_dir = f'plots/hybrid_rep_hl-{hidden_layers}_hd-{hidden_dim}-sl-{siren_layers}'
        else:
            save_dir = f'plots/{model_name}'
        # save_dir = f'plots/implicit_rep_hl-{hidden_layers}_hd-{hidden_dim}'

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Setup tracking variables
    reconstructed = img_fitting.create_reconstruction_tensor().to(device)
    gt = img_fitting.pixels

    psnr_vals = []
    loss_history = []

    H, W = img_fitting.H, img_fitting.W

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
        if epoch % steps_til_summary == 0:
            print(f"Epoch: {epoch} | Loss: {avg_loss:.5f} | PSNR: {psnr_val:.4f} | Normalized: {img_fitting.normalization}")

            img_rec = img_fitting.reconstruction_to_image(reconstructed.detach().cpu())

            plt.figure(figsize=(20, 5))
            plt.suptitle(f"Epochs={epoch} | PSNR={psnr_val:.4f}")

            plt.subplot(1, 4, 1)
            plt.title("Normalized Image")
            plt.imshow(img_fitting.normalized_tensor.numpy().clip(0, 1))
            plt.axis('off')

            plt.subplot(1, 4, 2)
            plt.title("Original Image")
            plt.imshow(img_fitting.img_tensor.permute(1, 2, 0).numpy())
            plt.axis('off')

            plt.subplot(1, 4, 3)
            plt.title(f"Reconstruction")
            plt.imshow(img_rec.numpy())
            plt.axis('off')

            plt.subplot(1, 4, 4)
            plt.title("Reconstruction Denormalized")
            denormalized_rec = img_fitting.denormalize(img_rec)
            plt.imshow(denormalized_rec.numpy())
            plt.axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"epoch_{epoch}.png"),
                        dpi=300,
                        bbox_inches='tight')
            plt.show()
            plt.close()

    # Save the trained model
    model_name = f"implicit_model.pt"
    save_path = os.path.join(save_dir, model_name)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'psnr_history': psnr_vals,
        'loss_history': loss_history,
        'epochs': epochs
    }, save_path)

    return {
        'psnr_history': psnr_vals,
        'loss_history': loss_history,
        'final_psnr': psnr_vals[-1] if psnr_vals else None,
        'model_save_path': save_path,
        'reconstructed_image': img_fitting.reconstruction_to_image(reconstructed.detach().cpu())
    }