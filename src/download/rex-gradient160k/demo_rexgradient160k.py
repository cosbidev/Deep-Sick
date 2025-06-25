import os
from datasets import load_dataset, load_from_disk, DatasetDict

# === CONFIGURAZIONE ===
DATASET_NAME = "rajpurkarlab/ReXGradient-160K"
BASE_DIR = "/mimer/NOBACKUP/groups/naiss2023-6-336/Deep-Sick/data/rex-gradient160k"  # Cambia se necessario
SAVE_CSV = True  # Imposta a False se non vuoi i CSV
TOKEN = None     # Inserisci il tuo token Hugging Face se necessario

def download_and_save(dataset_name: str, base_dir: str, token=None):
    """Scarica il dataset e salva ogni split su disco."""
    print(f"🔽 Download del dataset: {dataset_name}")
    ds = load_dataset(dataset_name, use_auth_token=token)

    os.makedirs(base_dir, exist_ok=True)

    for split in ds:
        out_dir = os.path.join(base_dir, split)
        print(f"💾 Salvataggio split '{split}' in {out_dir}")
        ds[split].save_to_disk(out_dir)

        if SAVE_CSV:
            df = ds[split].to_pandas()
            df.to_csv(os.path.join(base_dir, f"{split}.csv"), index=False)
            print(f"📄 CSV salvato in {split}.csv")

    print("✅ Download e salvataggio completati.")


def explore_sample(dataset: DatasetDict):
    """Stampa un esempio da ciascuno split."""
    print("\n🔎 Esempio da 'train':")
    print(dataset["train"][0])

    print("\n📊 Dimensioni degli split:")
    for split in dataset:
        print(f"{split}: {len(dataset[split])} esempi")

def filter_with_findings(dataset: DatasetDict):
    """Filtra i campioni che hanno almeno una label (finding)."""
    filtered = dataset["train"].filter(lambda x: len(x["labels"]) > 0)
    print(f"\n🧼 Campioni con findings: {len(filtered)} su {len(dataset['train'])}")
    return filtered

if __name__ == "__main__":
    # Step 1: scarica solo se i dati non esistono
    if not os.path.exists(os.path.join(BASE_DIR, "train")):
        download_and_save(DATASET_NAME, BASE_DIR, TOKEN)

    # Step 2: carica
    dataset = load_all_splits(BASE_DIR)

