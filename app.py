import json
import urllib.request
import zipfile
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import BertConfig, BertModel, BertTokenizer

BUNDLE_URL = "https://github.com/Yamsheen/fashion-retrieval/releases/download/v1/deploy_bundle.zip"
DATA = Path("/tmp/fashion_data")

st.set_page_config(page_title="Fashion Retrieval", page_icon="👗", layout="wide")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---- Model definitions (identical to the training notebook) ----
class SmallVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Flatten(),
        )
        self.flatten_dim = 64 * 28 * 28
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, self.flatten_dim)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 28, 28)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1), nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.decoder(self.fc_decode(z)), mu, logvar


class SimpleCLIP(nn.Module):
    def __init__(self, image_embed_dim=256, text_embed_dim=256):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(32, image_embed_dim),
        )
        # Build BERT from its config only (weights come from our checkpoint), saving a 440 MB download
        self.text_encoder = BertModel(BertConfig.from_pretrained("bert-base-uncased"))
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, text_embed_dim)

    def encode_image(self, images):
        return F.normalize(self.image_encoder(images), dim=-1)

    def encode_text(self, input_ids, attention_mask):
        out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(self.text_proj(out.last_hidden_state[:, 0]), dim=-1)


# ---- Load everything once per server start ----
def find_root():
    """Folder containing the model files, even if the zip nests them inside a subfolder."""
    for p in DATA.rglob("vae_trained.pth"):
        return p.parent
    return None


@st.cache_resource(show_spinner="Loading models (first start takes about a minute)...")
def load_all():
    root = find_root()
    if root is None:
        DATA.mkdir(parents=True, exist_ok=True)
        zip_path = DATA / "bundle.zip"
        urllib.request.urlretrieve(BUNDLE_URL, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA)
        zip_path.unlink()
        root = find_root()
        assert root is not None, "vae_trained.pth not found in the downloaded bundle"

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    clip = SimpleCLIP()
    sd = torch.load(root / "clip_model_fp16.pth", map_location="cpu")
    sd = {k: v.float() for k, v in sd.items() if not k.endswith("position_ids")}
    missing, _ = clip.load_state_dict(sd, strict=False)
    assert not missing, f"Missing CLIP weights: {missing}"
    clip.eval()

    vae = SmallVAE(latent_dim=64)
    vae.load_state_dict(torch.load(root / "vae_trained.pth", map_location="cpu"))
    vae.eval()

    clip_emb = F.normalize(torch.load(root / "clip_gallery_emb.pt").float(), dim=1)
    vae_emb = F.normalize(torch.load(root / "vae_gallery_emb.pt").float(), dim=1)
    captions = [m["text"] for m in json.load(open(root / "gallery_meta.json"))]

    ex_file = root / "example_queries.json"
    examples = json.load(open(ex_file)) if ex_file.exists() else \
        ["black tank top", "green floral", "a red half sleeves top", "cute girlish summer top"]
    return root, tokenizer, clip, vae, clip_emb, vae_emb, captions, examples


ROOT_DIR, tokenizer, clip, vae, CLIP_EMB, VAE_EMB, CAPTIONS, EXAMPLES = load_all()


def to_paths(indices):
    return [str(ROOT_DIR / "gallery_images" / f"{i}.jpg") for i in indices]


def search_by_text(query, k):
    with torch.no_grad():
        tokens = tokenizer(query, return_tensors="pt", truncation=True, max_length=128)
        t = clip.encode_text(tokens["input_ids"], tokens["attention_mask"])
        sims = (CLIP_EMB @ t.T).squeeze(1)
    return to_paths(sims.topk(k).indices.tolist())


def search_by_image(img, k):
    x = transform(img.convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        _, mu, _ = vae(x)
        sims = F.cosine_similarity(mu, VAE_EMB)
    out, seen = [], set()
    for i in sims.topk(min(k * 5, len(sims))).indices.tolist():
        if CAPTIONS[i] in seen:      # skip duplicate-looking items
            continue
        seen.add(CAPTIONS[i])
        out.append(i)
        if len(out) == k:
            break
    return to_paths(out)


def show_grid(paths, per_row=5):
    for r in range(0, len(paths), per_row):
        cols = st.columns(per_row)
        for col, p in zip(cols, paths[r:r + per_row]):
            col.image(p, use_container_width=True)


# ---- UI ----
st.title("👗 Multimodal Fashion Retrieval")
st.write(
    "Search a gallery of tops by **text description** (CLIP-style dual encoder: CNN + BERT, contrastive loss) "
    "or by **image** (Variational Autoencoder latent space). Both models were trained from scratch on a ~7K "
    "Fashion200K subset, so results are best on simple colour and style queries."
)

tab_text, tab_img = st.tabs(["Text → Image", "Image → Image"])

with tab_text:
    if "q" not in st.session_state:
        st.session_state.q = ""

    def set_q(text):
        st.session_state.q = text

    st.caption("Try an example:")
    for col, ex in zip(st.columns(len(EXAMPLES)), EXAMPLES):
        col.button(ex, on_click=set_q, args=(ex,), use_container_width=True)

    query = st.text_input("Describe a top", key="q", placeholder="e.g. black tank top")
    k1 = st.slider("Number of results", 1, 10, 5, key="k1")
    if query.strip():
        show_grid(search_by_text(query, k1))

with tab_img:
    up = st.file_uploader("Upload a top", type=["jpg", "jpeg", "png"])
    k2 = st.slider("Number of results", 1, 10, 5, key="k2")
    if up is not None:
        img = Image.open(up)
        st.image(img, caption="Your image", width=200)
        show_grid(search_by_image(img, k2))

st.caption("Built by Shanzay Omar & Yamsheen Saqib · Dataset: Fashion200K (Han et al., ICCV 2017)")
