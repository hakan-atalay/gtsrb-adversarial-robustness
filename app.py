import argparse
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from signguard.attacks import fgsm, pgd
from signguard.config import ASSETS_DIR, CKPT_DIR, CLASS_NAMES_TR, DEFAULT_CACHE, IMG_SIZE, SPEED60_CLASS, STOP_CLASS
from signguard.gradcam import grad_cam
from signguard.utils import get_device, load_model
from signguard.visualize import overlay_cam, perturbation_image, to_hwc

EXAMPLES_DIR = ASSETS_DIR / "examples"
DISPLAY_SIZE = 224
EXAMPLE_CLASSES = (STOP_CLASS, SPEED60_CLASS, 17, 13, 1, 25, 27, 38)
ATTACKS = ["FGSM", "PGD", "Hedefli PGD"]
MODEL_TITLES = {
    "standard": "Standart model",
    "fast_at": "Fast-AT (dayanıklı)",
    "pgd_at": "PGD-AT (dayanıklı)",
    "trades": "TRADES (dayanıklı)",
    "resnet18_surrogate": "ResNet-18",
}


def export_examples(cache_path):
    existing = sorted(EXAMPLES_DIR.glob("*.png"))
    if existing:
        return existing
    if not Path(cache_path).exists():
        return []
    cache = torch.load(cache_path, map_location="cpu", weights_only=True)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for label in EXAMPLE_CLASSES:
        matches = (cache["y_test"] == label).nonzero()
        if len(matches) == 0:
            continue
        image = cache["x_test"][int(matches[0])].permute(1, 2, 0).numpy()
        path = EXAMPLES_DIR / f"{label:02d}.png"
        Image.fromarray(image).save(path)
        paths.append(path)
    return paths


def to_tensor(image, size):
    array = np.asarray(image.convert("RGB").resize((size, size), Image.BICUBIC), dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)


def to_display(array):
    image = Image.fromarray((np.clip(array, 0, 1) * 255).astype(np.uint8))
    return image.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.NEAREST)


def top_probs(probs, k=5):
    values, indices = probs.topk(k)
    return {CLASS_NAMES_TR[i]: float(v) for v, i in zip(values.tolist(), indices.tolist())}


def build_demo(models, device):
    def analyze(image, attack, eps, steps, target_name):
        if image is None:
            raise gr.Error("Lütfen bir trafik işareti görseli yükleyin ya da örneklerden birini seçin.")
        outputs = []
        for model, size in models.values():
            x = to_tensor(image, size).to(device)
            with torch.no_grad():
                p_clean = F.softmax(model(x), dim=1)[0].cpu()
            radius = eps / 255
            if attack == "FGSM":
                x_adv = fgsm(model, x, p_clean.argmax().view(1).to(device), radius)
            elif attack == "PGD":
                x_adv = pgd(model, x, p_clean.argmax().view(1).to(device), radius, steps=int(steps))
            else:
                target = torch.tensor([CLASS_NAMES_TR.index(target_name)], device=device)
                x_adv = pgd(model, x, target, radius, steps=int(steps), targeted=True)
            with torch.no_grad():
                p_adv = F.softmax(model(x_adv), dim=1)[0].cpu()
            cam, _ = grad_cam(model, x_adv)
            outputs += [
                top_probs(p_clean),
                to_display(to_hwc(x_adv[0])),
                top_probs(p_adv),
                to_display(perturbation_image(x[0], x_adv[0], radius)),
                to_display(overlay_cam(to_hwc(x_adv[0]), cam[0].cpu().numpy())),
            ]
        return outputs

    with gr.Blocks(title="SignGuard: Trafik İşaretlerinde Adversarial Saldırılar") as demo:
        gr.Markdown(
            "# SignGuard\n"
            "Bir trafik işareti seçin, ardından saldırı türünü ve gücünü (ε) belirleyin. "
            "Her model kendi gradyanları kullanılarak saldırıya uğrar (white-box). "
            "Standart model ile adversarial eğitimli modelin tepkisini karşılaştırın."
        )
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Trafik işareti")
                attack = gr.Radio(ATTACKS, value="PGD", label="Saldırı türü")
                eps = gr.Slider(0, 16, value=8, step=1, label="Saldırı gücü ε (x/255)")
                steps = gr.Slider(1, 100, value=20, step=1, label="PGD adım sayısı")
                target = gr.Dropdown(CLASS_NAMES_TR, value=CLASS_NAMES_TR[SPEED60_CLASS], label="Hedef sınıf (hedefli saldırı için)")
                button = gr.Button("Saldır", variant="primary")
                examples = export_examples(DEFAULT_CACHE)
                if examples:
                    gr.Examples([[str(p)] for p in examples], inputs=[image])
            outputs = []
            for title in models:
                with gr.Column(scale=1):
                    gr.Markdown(f"### {MODEL_TITLES.get(title, title)}")
                    outputs.append(gr.Label(num_top_classes=3, label="Saldırı öncesi tahmin"))
                    outputs.append(gr.Image(label="Saldırı sonrası görüntü"))
                    outputs.append(gr.Label(num_top_classes=3, label="Saldırı sonrası tahmin"))
                    outputs.append(gr.Image(label="Eklenen gürültü (büyütülmüş)"))
                    outputs.append(gr.Image(label="Grad-CAM: model nereye bakıyor?"))
        button.click(analyze, inputs=[image, attack, eps, steps, target], outputs=outputs)
    return demo


def main():
    parser = argparse.ArgumentParser(description="SignGuard Gradio arayüzü")
    parser.add_argument("--models", nargs="+", default=["standard", "pgd_at"])
    parser.add_argument("--device", default=None)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    device = get_device(args.device)
    models = {}
    for name in args.models:
        path = Path(name) if name.endswith(".pt") else CKPT_DIR / f"{name}.pt"
        if not path.exists():
            print(f"Atlandı, bulunamadı: {path}")
            continue
        model, checkpoint = load_model(path, device)
        models[path.stem] = (model, checkpoint.get("img_size", IMG_SIZE))
    if not models:
        raise SystemExit("Hiç model bulunamadı. Önce 'python -m signguard.train' ile model eğitin.")
    build_demo(models, device).launch(share=args.share, server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()
