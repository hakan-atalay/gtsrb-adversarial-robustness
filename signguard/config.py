from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "checkpoints"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
ASSETS_DIR = ROOT / "assets"

IMG_SIZE = 48
NUM_CLASSES = 43
DEFAULT_CACHE = DATA_DIR / f"gtsrb_{IMG_SIZE}.pt"

STOP_CLASS = 14
SPEED60_CLASS = 3

CLASS_NAMES = [
    "Speed limit 20",
    "Speed limit 30",
    "Speed limit 50",
    "Speed limit 60",
    "Speed limit 70",
    "Speed limit 80",
    "End of speed limit 80",
    "Speed limit 100",
    "Speed limit 120",
    "No passing",
    "No passing for trucks",
    "Right-of-way at intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "No trucks",
    "No entry",
    "General caution",
    "Dangerous curve left",
    "Dangerous curve right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all restrictions",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing for trucks",
]

CLASS_NAMES_TR = [
    "Hız sınırı 20",
    "Hız sınırı 30",
    "Hız sınırı 50",
    "Hız sınırı 60",
    "Hız sınırı 70",
    "Hız sınırı 80",
    "Hız sınırı 80 sonu",
    "Hız sınırı 100",
    "Hız sınırı 120",
    "Sollama yasağı",
    "Kamyonlara sollama yasağı",
    "Kavşakta geçiş üstünlüğü",
    "Ana yol",
    "Yol ver",
    "Dur",
    "Taşıt giremez",
    "Kamyon giremez",
    "Girilmez",
    "Dikkat",
    "Sola tehlikeli viraj",
    "Sağa tehlikeli viraj",
    "Çift viraj",
    "Engebeli yol",
    "Kaygan yol",
    "Sağdan daralan yol",
    "Yol çalışması",
    "Trafik ışıkları",
    "Yaya geçidi",
    "Okul geçidi",
    "Bisiklet geçidi",
    "Buzlanma",
    "Vahşi hayvan",
    "Tüm yasakların sonu",
    "İleriden sağa mecburi",
    "İleriden sola mecburi",
    "İleri mecburi",
    "İleri veya sağa mecburi",
    "İleri veya sola mecburi",
    "Sağdan gidiniz",
    "Soldan gidiniz",
    "Döner kavşak",
    "Sollama yasağı sonu",
    "Kamyon sollama yasağı sonu",
]
