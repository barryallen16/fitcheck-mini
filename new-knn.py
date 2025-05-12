import gradio as gr
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
from google import genai
import torch
from torchvision import transforms
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights
import re
import os
from dotenv import load_dotenv

load_dotenv() 
try:
    import torch
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Install with: pip install torch torchvision")

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("Ultralytics not available. Install with: pip install ultralytics")

FASHION_MNIST_CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def load_fashion_mnist_model(model_path="best-fitcheck.pt"):
    if not ULTRALYTICS_AVAILABLE:
        print("Ultralytics not available. Clothing classification disabled.")
        return None
    try:
        model = YOLO(model_path)
        print(f"Loaded YOLOv8 classification model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}. Classification disabled.")
        return None

def rgb_to_hsl(r, g, b):
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    delta = cmax - cmin
    l = (cmax + cmin) / 2
    if delta == 0:
        s = 0
        h = 0
    else:
        s = delta / (1 - abs(2 * l - 1))
        if cmax == r:
            h = 60 * (((g - b) / delta) % 6)
        elif cmax == g:
            h = 60 * ((b - r) / delta + 2)
        else:
            h = 60 * ((r - g) / delta + 4)
    return h, s * 100, l * 100

def get_color_name(r, g, b):
    h, s, l = rgb_to_hsl(r, g, b)
    if 50 <= r <= 100 and 30 <= g <= 70 and 10 <= b <= 40 and l < 30:
        return "Dark Brown"
    elif 80 <= r <= 150 and 40 <= g <= 100 and 20 <= b <= 60:
        return "Brown"
    elif 200 <= r <= 255 and 0 <= g <= 50 and 0 <= b <= 50:
        return "Red"
    elif 0 <= r <= 50 and 0 <= g <= 100 and 150 <= b <= 255:
        return "Blue"
    elif 180 <= r <= 220 and 160 <= g <= 200 and 120 <= b <= 160:
        return "Beige"
    return "Unknown Color"

class BackgroundRemover:
    def __init__(self, method="deeplabv3", device=None):
        if method not in ["deeplabv3", "grabcut"]:
            raise ValueError(f"Invalid method: {method}. Choose 'deeplabv3' or 'grabcut'.")
        self.method = method
        self.device = device if device else ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.models = {}

    def _init_deeplabv3(self):
        if "deeplabv3" not in self.models and TORCH_AVAILABLE:
            try:
                self.models["deeplabv3"] = torch.hub.load(
                    "pytorch/vision:v0.10.0",
                    "deeplabv3_resnet50",
                    weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
                )
                self.models["deeplabv3"].eval().to(self.device)
                print(f"DeepLabV3 model loaded on {self.device}")
                self.preprocess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            except Exception as e:
                print(f"Error loading DeepLabV3 model: {e}")
                self.models["deeplabv3"] = None

    def _remove_bg_deeplabv3(self, image):
        self._init_deeplabv3()
        if self.models["deeplabv3"] is None:
            print("DeepLabV3 model not available. Falling back to GrabCut.")
            return self._remove_bg_grabcut(image)
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.models["deeplabv3"](input_tensor)["out"][0]
        person_mask = output.argmax(0).eq(15).cpu().numpy()
        if isinstance(image, Image.Image):
            np_image = np.array(image)
        result = np.zeros((np_image.shape[0], np_image.shape[1], 4), dtype=np.uint8)
        result[:, :, :3] = np_image
        result[:, :, 3] = person_mask.astype(np.uint8) * 255
        return result

    def _remove_bg_grabcut(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        elif isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mask = np.zeros(image.shape[:2], np.uint8)
        height, width = image.shape[:2]
        rect_margin = min(width, height) // 6
        rect = (rect_margin, rect_margin, width - 2 * rect_margin, height - 2 * rect_margin)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
        result = np.zeros((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        result[:, :, :3] = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result[:, :, 3] = mask2 * 255
        return result

    def remove_background(self, image):
        if self.method == "deeplabv3":
            return self._remove_bg_deeplabv3(image)
        elif self.method == "grabcut":
            return self._remove_bg_grabcut(image)
        else:
            raise ValueError(f"Unknown background removal method: {self.method}")

class ClothingColorExtractor:
    def __init__(self, bg_removal_method="deeplabv3", color_threshold=15, device=None, model_path="best-fitcheck.pt"):
        if bg_removal_method not in ["deeplabv3", "grabcut"]:
            raise ValueError(f"Invalid bg_removal_method: {bg_removal_method}. Choose 'deeplabv3' or 'grabcut'.")
        self.bg_remover = BackgroundRemover(method=bg_removal_method, device=device)
        self.color_threshold = color_threshold
        self.device = device if device else ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.classification_model = load_fashion_mnist_model(model_path)
        self.classification_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])
        ])

    def remove_background(self, image):
        return self.bg_remover.remove_background(image)

    def color_distance(self, color1, color2):
        return np.sqrt(sum((np.array(color1) - np.array(color2)) ** 2))

    def merge_similar_colors(self, colors_info):
        if not colors_info:
            return []
        colors_info.sort(key=lambda x: x[1], reverse=True)
        merged_colors = []
        processed = set()
        for i, (color1, pct1) in enumerate(colors_info):
            if i in processed:
                continue
            similar_colors = [(color1, pct1)]
            processed.add(i)
            for j, (color2, pct2) in enumerate(colors_info):
                if j != i and j not in processed:
                    if self.color_distance(color1, color2) <= self.color_threshold:
                        similar_colors.append((color2, pct2))
                        processed.add(j)
            if len(similar_colors) > 1:
                total_percentage = sum(pct for _, pct in similar_colors)
                weighted_color = np.zeros(3)
                for color, pct in similar_colors:
                    weighted_color += np.array(color) * (pct / total_percentage)
                merged_color = tuple(int(c) for c in weighted_color)
                merged_colors.append((merged_color, total_percentage))
            else:
                merged_colors.append((color1, pct1))
        merged_colors.sort(key=lambda x: x[1], reverse=True)
        return merged_colors

    def extract_dominant_colors(self, image, n_colors=15, min_percentage=5.0, merge_similar=True):
        if isinstance(image, str):
            image = Image.open(image)
        if isinstance(image, Image.Image):
            image = np.array(image)
        if image.shape[2] == 3:
            image = self.remove_background(image)
        pixels = image[image[:, :, 3] > 0]
        pixels = pixels[:, :3]
        if len(pixels) == 0:
            return []
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels)
        colors = kmeans.cluster_centers_.astype(int)
        labels = kmeans.labels_
        label_counts = Counter(labels)
        total_pixels = len(pixels)
        color_info = []
        for label, count in label_counts.items():
            percentage = (count / total_pixels) * 100
            if percentage >= min_percentage:
                color_rgb = tuple(colors[label])
                color_info.append((color_rgb, percentage))
        if merge_similar:
            color_info = self.merge_similar_colors(color_info)
        return color_info

    def classify_clothing(self, image):
        if self.classification_model is None:
            return "Classification model not available", 0.0
        if isinstance(image, np.ndarray):
            if image.shape[2] == 4:
                bg = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255
                alpha = image[:, :, 3:4] / 255.0
                rgb = image[:, :, :3]
                image = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
            image = Image.fromarray(image)
        input_tensor = self.classification_preprocess(image).unsqueeze(0).to(self.device)
        input_image = transforms.ToPILImage()(input_tensor.squeeze(0).cpu())
        results = self.classification_model(input_image)
        probs = results[0].probs
        confidence = probs.top1conf.item() * 100
        predicted_idx = probs.top1
        predicted_class = FASHION_MNIST_CLASSES[predicted_idx]
        return predicted_class, confidence

    def create_visualization(self, image, colors_info, min_percentage=5.0, title=None):
        if isinstance(image, Image.Image):
            image = np.array(image)
        fig = plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        if image.shape[2] == 4:
            bg = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255
            alpha = image[:, :, 3:4] / 255.0
            rgb = image[:, :, :3]
            img_display = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
            plt.imshow(img_display)
        else:
            plt.imshow(image[:, :, :3])
        plt.title("Original Image" if title is None else title)
        plt.axis("off")
        if colors_info:
            plt.subplot(2, 1, 2)
            num_colors = len(colors_info)
            swatch_height = 100
            swatch_width = 600 // num_colors
            swatches = np.zeros((swatch_height, swatch_width * num_colors, 3), dtype=np.uint8)
            for i, (color, percentage) in enumerate(colors_info):
                swatches[:, i * swatch_width:(i + 1) * swatch_width] = color
            plt.imshow(swatches)
            subtitle = f"Dominant Colors (>= {min_percentage}%)"
            plt.title(subtitle)
            plt.axis("off")
            for i, (color, percentage) in enumerate(colors_info):
                x_pos = i * swatch_width + swatch_width // 2
                text_color = "white" if sum(color) < 384 else "black"
                plt.text(
                    x_pos, swatch_height // 2, f"RGB: {color}\n{percentage:.1f}%",
                    ha="center", va="center", color=text_color, fontsize=10, fontweight="bold"
                )
        plt.tight_layout()
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        data = np.asarray(buf)
        data = data[:, :, :3]
        img = Image.fromarray(data)
        plt.close(fig)
        return img

    def process_image(self, image, min_percentage=5.0, color_threshold=None):
        if color_threshold is not None:
            self.color_threshold = color_threshold
        processed_image = self.remove_background(image)
        colors_info = self.extract_dominant_colors(processed_image, min_percentage=min_percentage)
        clothing_type, confidence = self.classify_clothing(processed_image)
        visualization = self.create_visualization(processed_image, colors_info, min_percentage=min_percentage)
        if isinstance(processed_image, np.ndarray):
            processed_image_pil = Image.fromarray(processed_image)
        else:
            processed_image_pil = processed_image
        formatted_colors = []
        for color, percentage in colors_info:
            hex_color = "#{:02x}{:02x}{:02x}".format(*color)
            r, g, b = color
            name = get_color_name(r, g, b)
            formatted_colors.append(f"RGB {color} ({hex_color}) - {percentage:.2f}% - {name}")
        return processed_image_pil, visualization, formatted_colors, clothing_type, confidence

def process_clothing_image(image, bg_removal_method, min_percentage, color_threshold, gender):
    min_percentage = float(min_percentage)
    color_threshold = float(color_threshold)
    extractor = ClothingColorExtractor(
        bg_removal_method=bg_removal_method,
        color_threshold=color_threshold,
        model_path="best-fitcheck.pt"
    )
    try:
        processed_image, visualization, color_list, clothing_type, confidence = extractor.process_image(
            image, min_percentage=min_percentage, color_threshold=color_threshold
        )
        color_text = "\n".join(color_list)
        classification_text = f"{clothing_type} ({confidence:.2f}% confidence)"
        if color_list:
            max_percentage = -1
            dominant_color_name = ""
            for color_entry in color_list:
                match = re.search(r"- (\d+\.\d+)% - (.+)$", color_entry)
                if match:
                    percentage = float(match.group(1))
                    color_name = match.group(2)
                    if percentage > max_percentage:
                        max_percentage = percentage
                        dominant_color_name = color_name
            if dominant_color_name:
                combined_output = f"{dominant_color_name} {clothing_type}"
                client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=f"Suggest a single-item outfit complement for {combined_output} for {gender}, in one phrase, like 'White Shirt.'"
                )
                outfit_recommendation = f"Pair with a {response.text.strip()}"
            else:
                combined_output = f"{clothing_type} (No dominant color detected)"
                outfit_recommendation = "Unable to recommend due to no dominant color"
        else:
            combined_output = f"{clothing_type} (No colors extracted)"
            outfit_recommendation = "Unable to recommend due to no colors extracted"
        return processed_image, visualization, color_text, classification_text, combined_output, outfit_recommendation
    except Exception as e:
        return None, None, f"Error processing image: {str(e)}", "Classification failed", "Error", "Error"

def create_interface():
    with gr.Blocks(title="Fitcheck", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Fitcheck: Outfit Recommendation System
            Upload a clothing image to identify its type, dominant colors, and receive a personalized outfit recommendation.
            """
        )
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="Upload Clothing Image")
                bg_method = gr.Radio(
                    choices=["deeplabv3", "grabcut"], value="deeplabv3", label="Background Removal Method"
                )
                gender = gr.Radio(choices=["Men", "Women"], value="Men", label="Gender")
                min_percent = gr.Slider(
                    minimum=1, maximum=20, value=13, step=0.5, label="Minimum Color Percentage (%)"
                )
                color_thresh = gr.Slider(
                    minimum=5, maximum=30, value=20, step=1, label="Color Similarity Threshold"
                )
                submit_btn = gr.Button("Analyze and Recommend", variant="primary")
            with gr.Column(scale=2):
                with gr.Tabs():
                    with gr.TabItem("Results"):
                        with gr.Row():
                            processed_img = gr.Image(label="Processed Image")
                            color_viz = gr.Image(label="Color Analysis")
                        color_text = gr.Textbox(label="Extracted Colors", lines=5)
                        classification_text = gr.Textbox(label="Clothing Type", lines=1)
                        combined_output = gr.Textbox(label="Detected Clothing", lines=1)
                        outfit_recommendation = gr.Textbox(label="Outfit Recommendation", lines=2)
        submit_btn.click(
            process_clothing_image,
            inputs=[input_image, bg_method, min_percent, color_thresh, gender],
            outputs=[
                processed_img, color_viz, color_text, classification_text, combined_output, outfit_recommendation
            ]
        )
        gr.Markdown(
            """
            ## Instructions
            1. Upload a clear image of a clothing item.
            2. Choose a background removal method: `deeplabv3` (accurate for human subjects) or `grabcut` (faster, less precise).
            3. Select gender for tailored outfit recommendations.
            4. Adjust the minimum color percentage to filter minor colors.
            5. Set the color similarity threshold to group similar colors.
            6. Click "Analyze and Recommend" to process the image and view results.

            ## About
            This app uses DeepLabV3 or GrabCut for background removal, K-means clustering for color extraction, and a YOLOv8 model trained on Fashion-MNIST for clothing classification. Outfit recommendations are generated using Google Gemini AI.
            """
        )
    return demo

demo = create_interface()

if __name__ == "__main__":
    demo.launch(show_error=True, server_name="0.0.0.0", server_port=7860)