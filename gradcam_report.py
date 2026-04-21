import os

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


IMG_SIZE = (128, 128)
BATCH_SIZE = 12
MODEL_PATH = "pneumonia.keras"
OUTPUT_FIG = "run7_gradcam_examples.png"
NUM_EXAMPLES = 3


def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]
    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(pred_index)


def _forward_without_input_layer(layers, x):
    for layer in layers:
        if isinstance(layer, tf.keras.layers.InputLayer):
            continue
        try:
            x = layer(x, training=False)
        except TypeError:
            x = layer(x)
    return x


def overlay_heatmap_on_image(image_uint8, heatmap, alpha=0.4):
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((image_uint8.shape[1], image_uint8.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
    superimposed = jet_heatmap * alpha + image_uint8
    return np.uint8(np.clip(superimposed, 0, 255))


def main():
    data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chest_xray")
    test_dir = os.path.join(data_root, "test")

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        seed=123,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        labels="inferred",
        shuffle=True,
    )
    class_names = test_ds.class_names

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    _ = model(tf.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32))

    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and "mobilenet" in layer.name.lower():
            base_model = layer
            break
    if base_model is None:
        raise RuntimeError("Could not find MobileNet base model in pneumonia.keras")

    base_index = model.layers.index(base_model)
    pre_layers = model.layers[:base_index]
    post_layers = model.layers[base_index + 1 :]

    images, labels = next(iter(test_ds))
    n = min(NUM_EXAMPLES, images.shape[0])
    plt.figure(figsize=(12, 4 * n))

    for i in range(n):
        image = images[i]
        image_uint8 = image.numpy().astype("uint8")
        img_array = tf.expand_dims(image, axis=0)
        with tf.GradientTape() as tape:
            x = _forward_without_input_layer(pre_layers, img_array)
            conv_output = base_model(x, training=False)
            tape.watch(conv_output)
            preds = _forward_without_input_layer(post_layers, conv_output)
            pred_idx = int(tf.argmax(preds[0]))
            class_channel = preds[:, pred_idx]

        grads = tape.gradient(class_channel, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_map = conv_output[0]
        heatmap = conv_map @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()

        overlay = overlay_heatmap_on_image(image_uint8, heatmap)
        true_idx = int(labels[i].numpy())

        plt.subplot(n, 3, (i * 3) + 1)
        plt.imshow(image_uint8)
        plt.title(f"Original\nTrue: {class_names[true_idx]}")
        plt.axis("off")

        plt.subplot(n, 3, (i * 3) + 2)
        plt.imshow(heatmap, cmap="jet")
        plt.title("Grad-CAM Heatmap")
        plt.axis("off")

        plt.subplot(n, 3, (i * 3) + 3)
        plt.imshow(overlay)
        plt.title(f"Overlay\nPred: {class_names[pred_idx]}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, bbox_inches="tight")
    plt.show()
    print(f"Saved: {OUTPUT_FIG}")


if __name__ == "__main__":
    main()
