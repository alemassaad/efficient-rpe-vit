"""
CIFAR-10 Dataset Visualizer
A simple Streamlit dashboard to explore and visualize CIFAR-10 images
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

# Page configuration
st.set_page_config(
    page_title="CIFAR-10 Visualizer",
    page_icon="🖼️",
    layout="wide"
)

# CIFAR-10 class names (in order, indices 0-9)
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']


def load_cifar10_batch(file_path):
    """
    Load a CIFAR-10 batch file (pickle format)

    Args:
        file_path: Path to batch file (data_batch_1-5 or test_batch)

    Returns:
        Dictionary with 'images' (N, 32, 32, 3) and 'labels' (N,)
    """
    with open(file_path, 'rb') as f:
        batch_dict = pickle.load(f, encoding='bytes')

    # Extract data
    # Raw shape: (10000, 3072) = (10000, 3 * 32 * 32)
    # Channel order: RGB
    images = batch_dict[b'data']
    labels = batch_dict[b'labels']

    # Reshape from flat to (N, 3, 32, 32)
    images = images.reshape(-1, 3, 32, 32)

    # Transpose to (N, 32, 32, 3) for matplotlib display
    images = images.transpose(0, 2, 3, 1)

    return {
        'images': images,  # (N, 32, 32, 3) uint8
        'labels': np.array(labels)  # (N,) int
    }


def load_cifar10_meta(file_path):
    """
    Load CIFAR-10 metadata (class names)

    Args:
        file_path: Path to batches.meta file

    Returns:
        List of class names
    """
    with open(file_path, 'rb') as f:
        meta_dict = pickle.load(f, encoding='bytes')

    # Decode class names from bytes to strings
    class_names = [name.decode('utf-8') for name in meta_dict[b'label_names']]

    return class_names


@st.cache_data
def load_cifar10_data():
    """Load CIFAR-10 data from cifar-10-batches-py/ (cached for performance)"""

    data_dir = Path('./cifar-10-batches-py')

    # Load training data from 5 batches
    train_images_list = []
    train_labels_list = []

    for i in range(1, 6):
        batch_path = data_dir / f'data_batch_{i}'
        batch_data = load_cifar10_batch(batch_path)
        train_images_list.append(batch_data['images'])
        train_labels_list.append(batch_data['labels'])

    # Concatenate all training batches
    train_images = np.concatenate(train_images_list, axis=0)  # (50000, 32, 32, 3)
    train_labels = np.concatenate(train_labels_list, axis=0)  # (50000,)

    # Load test data
    test_batch_path = data_dir / 'test_batch'
    test_data = load_cifar10_batch(test_batch_path)
    test_images = test_data['images']  # (10000, 32, 32, 3)
    test_labels = test_data['labels']  # (10000,)

    # Load class names (verify against hardcoded list)
    meta_path = data_dir / 'batches.meta'
    class_names = load_cifar10_meta(meta_path)

    return {
        'train': {'images': train_images, 'labels': train_labels},
        'test': {'images': test_images, 'labels': test_labels},
        'class_names': class_names
    }


def normalize_images(images, method='standard'):
    """
    Normalize images using different methods

    Args:
        images: numpy array of shape (N, 32, 32, 3) with uint8 values 0-255
        method: 'standard' (CIFAR-10 mean/std) or 'minmax' (scale to [0,1])

    Returns:
        Normalized images as float32
    """
    images_float = images.astype(np.float32) / 255.0

    if method == 'standard':
        # CIFAR-10 standardization (same as training)
        # Per-channel RGB normalization
        CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 1, 1, 3)
        CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616]).reshape(1, 1, 1, 3)
        return (images_float - CIFAR10_MEAN) / CIFAR10_STD
    else:
        # Simple [0, 1] scaling
        return images_float


def sample_images(images, labels, n_samples=25, class_filter=None, random_seed=None):
    """
    Sample random images from the dataset

    Args:
        images: numpy array of images
        labels: numpy array of labels (0-9)
        n_samples: number of samples to return
        class_filter: if specified, only sample images of this class (0-9)
        random_seed: for reproducibility

    Returns:
        sampled_images, sampled_labels
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Filter by class if requested
    if class_filter is not None:
        indices = np.where(labels == class_filter)[0]
    else:
        indices = np.arange(len(labels))

    # Sample random indices
    sample_indices = np.random.choice(indices, size=min(n_samples, len(indices)), replace=False)

    return images[sample_indices], labels[sample_indices]


def plot_image_grid(images, labels, class_names, grid_size=(5, 5), normalize=False, norm_method='standard'):
    """
    Plot a grid of images with labels

    Args:
        images: numpy array of images (N, 32, 32, 3)
        labels: numpy array of labels (N,)
        class_names: list of class names
        grid_size: tuple (rows, cols)
        normalize: whether to show normalized version
        norm_method: 'standard' or 'minmax'
    """
    rows, cols = grid_size
    n_images = min(len(images), rows * cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 2.5))
    axes = axes.flatten()

    # Normalize if requested
    if normalize:
        display_images = normalize_images(images[:n_images], method=norm_method)
        title_suffix = f" ({norm_method})"
    else:
        # Convert uint8 to [0, 1] float for display
        display_images = images[:n_images].astype(np.float32) / 255.0
        title_suffix = " (raw)"

    for i in range(n_images):
        # For standard normalization, clip to visible range to show the effect
        if normalize and norm_method == 'standard':
            # Standard normalized values can be outside [0, 1]
            # Clip to [0, 1] for display but this shows the washed-out effect
            display_img = np.clip(display_images[i], 0, 1)
            axes[i].imshow(display_img)
        else:
            axes[i].imshow(display_images[i])
        class_name = class_names[labels[i]]
        axes[i].set_title(f'{class_name}' + title_suffix, fontsize=10)
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig


def show_dataset_statistics(data, dataset_name, class_names):
    """Display statistics about the dataset"""
    images = data['images']
    labels = data['labels']

    st.subheader(f"📊 {dataset_name} Dataset Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Images", f"{len(images):,}")

    with col2:
        st.metric("Image Shape", "32 × 32 × 3")

    with col3:
        st.metric("Data Type", "uint8")

    with col4:
        st.metric("Value Range", "0-255")

    # Class distribution
    st.write("**Class Distribution:**")
    unique, counts = np.unique(labels, return_counts=True)

    # Create bar chart
    fig_dist, ax = plt.subplots(figsize=(12, 4))
    ax.bar(unique, counts, color='steelblue', alpha=0.8)
    ax.set_xlabel('Class', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{dataset_name} Set - Class Distribution', fontsize=14)
    ax.set_xticks(unique)
    ax.set_xticklabels([class_names[i] for i in unique], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    for cls, count in zip(unique, counts):
        ax.text(cls, count + 100, str(count), ha='center', va='bottom', fontsize=9)

    st.pyplot(fig_dist)
    plt.close(fig_dist)

    # Pixel statistics
    st.write("**Pixel Statistics (per channel):**")

    # Calculate per-channel statistics
    for channel, channel_name, color in zip([0, 1, 2], ['Red', 'Green', 'Blue'], ['red', 'green', 'blue']):
        channel_data = images[:, :, :, channel]
        st.write(f"**{channel_name} Channel**: Mean={channel_data.mean():.2f}, Std={channel_data.std():.2f}, Min={channel_data.min()}, Max={channel_data.max()}")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("🖼️ CIFAR-10 Dataset Visualizer")
    st.markdown("---")

    # Load data
    with st.spinner("Loading CIFAR-10 data..."):
        data = load_cifar10_data()

    class_names = data['class_names']

    # Sidebar controls
    st.sidebar.header("⚙️ Controls")

    # Dataset selection
    dataset_choice = st.sidebar.radio(
        "Select Dataset:",
        options=['train', 'test'],
        format_func=lambda x: f"Training Set (50,000)" if x == 'train' else "Test Set (10,000)"
    )

    current_data = data[dataset_choice]

    # Class filter
    class_options = ['All'] + class_names
    class_filter_choice = st.sidebar.selectbox(
        "Filter by Class:",
        options=class_options
    )

    class_filter = None if class_filter_choice == 'All' else class_names.index(class_filter_choice)

    # Number of samples
    n_samples = st.sidebar.slider(
        "Number of Samples:",
        min_value=1,
        max_value=100,
        value=25,
        step=1
    )

    # Calculate grid size
    grid_cols = int(np.ceil(np.sqrt(n_samples)))
    grid_rows = int(np.ceil(n_samples / grid_cols))

    # Normalization toggle
    show_normalized = st.sidebar.checkbox("Show Normalized Images", value=False)

    if show_normalized:
        norm_method = st.sidebar.radio(
            "Normalization Method:",
            options=['standard', 'minmax'],
            format_func=lambda x: "CIFAR-10 Standard (per-channel RGB)" if x == 'standard' else "Min-Max [0, 1]"
        )
    else:
        norm_method = 'standard'

    # Random seed
    use_seed = st.sidebar.checkbox("Fixed Random Seed (reproducible)", value=False)
    random_seed = 42 if use_seed else None

    # Resample button
    if st.sidebar.button("🎲 Resample", type="primary"):
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **About:**

    This dashboard visualizes the CIFAR-10 dataset loaded from `cifar-10-batches-py/` (pickle format).

    - **Raw images**: uint8 RGB pixels (0-255)
    - **Training set**: 50,000 images (5 batches)
    - **Test set**: 10,000 images
    - **Image size**: 32×32×3 (RGB)
    - **Classes**: 10 object categories
    - **Source**: Python pickle format
    """)

    # Main content area
    # Show statistics
    show_dataset_statistics(current_data, dataset_choice.capitalize(), class_names)

    st.markdown("---")

    # Sample and display images
    st.subheader("🖼️ Random Image Samples")

    filter_text = f"Showing class **{class_filter_choice}**" if class_filter is not None else "Showing **all classes**"
    norm_text = "(**normalized**)" if show_normalized else "(**raw pixels**)"
    st.write(f"{filter_text} from the **{dataset_choice}** set {norm_text}")

    sampled_images, sampled_labels = sample_images(
        current_data['images'],
        current_data['labels'],
        n_samples=n_samples,
        class_filter=class_filter,
        random_seed=random_seed
    )

    if len(sampled_images) == 0:
        st.warning(f"No images found for class {class_filter_choice}")
    else:
        fig = plot_image_grid(
            sampled_images,
            sampled_labels,
            class_names,
            grid_size=(grid_rows, grid_cols),
            normalize=show_normalized,
            norm_method=norm_method
        )
        st.pyplot(fig)
        plt.close(fig)

        st.caption(f"Showing {len(sampled_images)} samples in a {grid_rows}×{grid_cols} grid")


if __name__ == "__main__":
    main()
