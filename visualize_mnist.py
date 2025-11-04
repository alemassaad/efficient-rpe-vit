"""
MNIST Dataset Visualizer
A simple Streamlit dashboard to explore and visualize MNIST images
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import struct
import gzip

# Page configuration
st.set_page_config(
    page_title="MNIST Visualizer",
    page_icon="🔢",
    layout="wide"
)

def read_idx_images(file_path):
    """
    Read IDX image file format (MNIST images)

    IDX format:
    - magic number (4 bytes): 0x00000803 for images
    - number of images (4 bytes)
    - number of rows (4 bytes): 28
    - number of columns (4 bytes): 28
    - pixel data: uint8 values 0-255
    """
    # Check if file is gzipped
    if file_path.suffix == '.gz':
        with gzip.open(file_path, 'rb') as f:
            data = f.read()
    else:
        with open(file_path, 'rb') as f:
            data = f.read()

    # Parse header
    magic = struct.unpack('>I', data[0:4])[0]  # Big-endian 4-byte integer
    n_images = struct.unpack('>I', data[4:8])[0]
    n_rows = struct.unpack('>I', data[8:12])[0]
    n_cols = struct.unpack('>I', data[12:16])[0]

    # Verify magic number
    assert magic == 2051, f"Invalid magic number for images: {magic}"

    # Read pixel data
    images = np.frombuffer(data[16:], dtype=np.uint8)
    images = images.reshape(n_images, n_rows, n_cols)

    return images


def read_idx_labels(file_path):
    """
    Read IDX label file format (MNIST labels)

    IDX format:
    - magic number (4 bytes): 0x00000801 for labels
    - number of labels (4 bytes)
    - label data: uint8 values 0-9
    """
    # Check if file is gzipped
    if file_path.suffix == '.gz':
        with gzip.open(file_path, 'rb') as f:
            data = f.read()
    else:
        with open(file_path, 'rb') as f:
            data = f.read()

    # Parse header
    magic = struct.unpack('>I', data[0:4])[0]
    n_labels = struct.unpack('>I', data[4:8])[0]

    # Verify magic number
    assert magic == 2049, f"Invalid magic number for labels: {magic}"

    # Read label data
    labels = np.frombuffer(data[8:], dtype=np.uint8)

    return labels


@st.cache_data
def load_mnist_data():
    """Load MNIST data from data/MNIST/raw/ IDX files (cached for performance)"""

    data_dir = Path('./data/MNIST/raw')

    # Load training data (prefer uncompressed files, fall back to .gz)
    train_images_path = data_dir / 'train-images-idx3-ubyte'
    if not train_images_path.exists():
        train_images_path = data_dir / 'train-images-idx3-ubyte.gz'

    train_labels_path = data_dir / 'train-labels-idx1-ubyte'
    if not train_labels_path.exists():
        train_labels_path = data_dir / 'train-labels-idx1-ubyte.gz'

    # Load test data
    test_images_path = data_dir / 't10k-images-idx3-ubyte'
    if not test_images_path.exists():
        test_images_path = data_dir / 't10k-images-idx3-ubyte.gz'

    test_labels_path = data_dir / 't10k-labels-idx1-ubyte'
    if not test_labels_path.exists():
        test_labels_path = data_dir / 't10k-labels-idx1-ubyte.gz'

    # Read files
    train_images = read_idx_images(train_images_path)  # (60000, 28, 28)
    train_labels = read_idx_labels(train_labels_path)  # (60000,)

    test_images = read_idx_images(test_images_path)    # (10000, 28, 28)
    test_labels = read_idx_labels(test_labels_path)    # (10000,)

    return {
        'train': {'images': train_images, 'labels': train_labels},
        'test': {'images': test_images, 'labels': test_labels}
    }


def normalize_images(images, method='standard'):
    """
    Normalize images using different methods

    Args:
        images: numpy array of shape (N, 28, 28) with uint8 values 0-255
        method: 'standard' (MNIST mean/std) or 'minmax' (scale to [0,1])

    Returns:
        Normalized images as float32
    """
    images_float = images.astype(np.float32) / 255.0

    if method == 'standard':
        # MNIST standardization (same as training)
        MNIST_MEAN = 0.1307
        MNIST_STD = 0.3081
        return (images_float - MNIST_MEAN) / MNIST_STD
    else:
        # Simple [0, 1] scaling
        return images_float


def sample_images(images, labels, n_samples=25, digit_filter=None, random_seed=None):
    """
    Sample random images from the dataset

    Args:
        images: numpy array of images
        labels: numpy array of labels
        n_samples: number of samples to return
        digit_filter: if specified, only sample images of this digit (0-9)
        random_seed: for reproducibility

    Returns:
        sampled_images, sampled_labels
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Filter by digit if requested
    if digit_filter is not None:
        indices = np.where(labels == digit_filter)[0]
    else:
        indices = np.arange(len(labels))

    # Sample random indices
    sample_indices = np.random.choice(indices, size=min(n_samples, len(indices)), replace=False)

    return images[sample_indices], labels[sample_indices]


def plot_image_grid(images, labels, grid_size=(5, 5), normalize=False, norm_method='standard'):
    """
    Plot a grid of images with labels

    Args:
        images: numpy array of images
        labels: numpy array of labels
        grid_size: tuple (rows, cols)
        normalize: whether to show normalized version
        norm_method: 'standard' or 'minmax'
    """
    rows, cols = grid_size
    n_images = min(len(images), rows * cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    axes = axes.flatten()

    # Normalize if requested
    if normalize:
        display_images = normalize_images(images[:n_images], method=norm_method)
        title_suffix = f" ({norm_method})"
    else:
        display_images = images[:n_images]
        title_suffix = " (raw)"

    for i in range(n_images):
        axes[i].imshow(display_images[i], cmap='gray')
        axes[i].set_title(f'Label: {labels[i]}' + title_suffix, fontsize=10)
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    return fig


def show_dataset_statistics(data, dataset_name):
    """Display statistics about the dataset"""
    images = data['images']
    labels = data['labels']

    st.subheader(f"📊 {dataset_name} Dataset Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Images", f"{len(images):,}")

    with col2:
        st.metric("Image Shape", "28 × 28")

    with col3:
        st.metric("Data Type", "uint8")

    with col4:
        st.metric("Value Range", "0-255")

    # Class distribution
    st.write("**Class Distribution:**")
    unique, counts = np.unique(labels, return_counts=True)
    dist_data = {f"Digit {digit}": count for digit, count in zip(unique, counts)}

    # Create bar chart
    fig_dist, ax = plt.subplots(figsize=(10, 3))
    ax.bar(unique, counts, color='steelblue', alpha=0.8)
    ax.set_xlabel('Digit', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'{dataset_name} Set - Digit Distribution', fontsize=14)
    ax.set_xticks(unique)
    ax.grid(axis='y', alpha=0.3)

    for digit, count in zip(unique, counts):
        ax.text(digit, count + 100, str(count), ha='center', va='bottom', fontsize=9)

    st.pyplot(fig_dist)
    plt.close(fig_dist)

    # Pixel statistics
    st.write("**Pixel Statistics:**")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write(f"Mean pixel value: **{images.mean():.2f}**")

    with col2:
        st.write(f"Std pixel value: **{images.std():.2f}**")

    with col3:
        st.write(f"Min/Max: **{images.min()}** / **{images.max()}**")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.title("🔢 MNIST Dataset Visualizer")
    st.markdown("---")

    # Load data
    with st.spinner("Loading MNIST data..."):
        data = load_mnist_data()

    # Sidebar controls
    st.sidebar.header("⚙️ Controls")

    # Dataset selection
    dataset_choice = st.sidebar.radio(
        "Select Dataset:",
        options=['train', 'test'],
        format_func=lambda x: f"Training Set (60,000)" if x == 'train' else "Test Set (10,000)"
    )

    current_data = data[dataset_choice]

    # Digit filter
    digit_options = ['All'] + list(range(10))
    digit_filter_choice = st.sidebar.selectbox(
        "Filter by Digit:",
        options=digit_options
    )

    digit_filter = None if digit_filter_choice == 'All' else digit_filter_choice

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
            format_func=lambda x: "MNIST Standard (mean=0.1307, std=0.3081)" if x == 'standard' else "Min-Max [0, 1]"
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

    This dashboard visualizes the MNIST handwritten digit dataset loaded from `data/MNIST/` (torchvision format).

    - **Raw images**: uint8 pixels (0-255)
    - **Training set**: 60,000 images
    - **Test set**: 10,000 images
    - **Image size**: 28×28 grayscale
    - **Source**: IDX binary format
    """)

    # Main content area
    # Show statistics
    show_dataset_statistics(current_data, dataset_choice.capitalize())

    st.markdown("---")

    # Sample and display images
    st.subheader("🖼️ Random Image Samples")

    filter_text = f"Showing digit **{digit_filter}**" if digit_filter is not None else "Showing **all digits**"
    norm_text = "(**normalized**)" if show_normalized else "(**raw pixels**)"
    st.write(f"{filter_text} from the **{dataset_choice}** set {norm_text}")

    sampled_images, sampled_labels = sample_images(
        current_data['images'],
        current_data['labels'],
        n_samples=n_samples,
        digit_filter=digit_filter,
        random_seed=random_seed
    )

    if len(sampled_images) == 0:
        st.warning(f"No images found for digit {digit_filter}")
    else:
        fig = plot_image_grid(
            sampled_images,
            sampled_labels,
            grid_size=(grid_rows, grid_cols),
            normalize=show_normalized,
            norm_method=norm_method
        )
        st.pyplot(fig)
        plt.close(fig)

        st.caption(f"Showing {len(sampled_images)} samples in a {grid_rows}×{grid_cols} grid")


if __name__ == "__main__":
    main()
