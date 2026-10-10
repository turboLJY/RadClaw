#!/usr/bin/env python3
"""Pre-download all models required by the evaluation metrics.

Handles SSL certificate issues common in corporate/research environments
by disabling SSL verification for HuggingFace downloads.

Models downloaded:
  - bert-base-uncased (CheXbert dependency)
  - CheXbert checkpoint (stanfordmlgroup/CheXbert via Box)
  - Angelakeke/RaTE-NER-Deberta (RaTEScore NER)
  - FremyCompany/BioLORD-2023-C (RaTEScore embedding)
  - StanfordAIMI/GREEN-RadLlama2-7b (GREEN scorer) [optional, 7B model]
"""

import os
import sys

def disable_ssl_verification():
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    import requests
    original_send = requests.Session.send
    def patched_send(self, request, **kwargs):
        kwargs["verify"] = False
        return original_send(self, request, **kwargs)
    requests.Session.send = patched_send

disable_ssl_verification()

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_chexbert_deps():
    logger.info("Downloading bert-base-uncased...")
    from transformers import BertTokenizer, BertModel
    BertTokenizer.from_pretrained("bert-base-uncased")
    BertModel.from_pretrained("bert-base-uncased")
    logger.info("  bert-base-uncased OK")

    logger.info("Downloading CheXbert checkpoint...")
    from huggingface_hub import hf_hub_download
    cache_dir = os.path.expanduser("/mnt/yulan_pretrain_131/clinical_project/hf/chexbert")
    os.makedirs(cache_dir, exist_ok=True)
    target = os.path.join(cache_dir, "chexbert.pth")
    if not os.path.exists(target):
        hf_hub_download(
            repo_id="StanfordAIMI/RRG_scorers",
            filename="chexbert.pth",
            cache_dir=cache_dir,
            force_filename="chexbert.pth",
        )
    logger.info("  CheXbert checkpoint OK: %s", target)

    logger.info("Verifying CheXbert loads...")
    from f1chexbert import F1CheXbert
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    f1 = F1CheXbert(device=device)
    logger.info("  CheXbert OK (device=%s)", device)


def download_ratescore_deps():
    logger.info("Downloading RaTE-NER-Deberta...")
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    AutoTokenizer.from_pretrained("Angelakeke/RaTE-NER-Deberta")
    AutoModelForTokenClassification.from_pretrained("Angelakeke/RaTE-NER-Deberta")
    logger.info("  RaTE-NER-Deberta OK")

    logger.info("Downloading BioLORD-2023-C...")
    from transformers import AutoTokenizer as AT2, AutoModel
    AT2.from_pretrained("FremyCompany/BioLORD-2023-C")
    AutoModel.from_pretrained("FremyCompany/BioLORD-2023-C")
    logger.info("  BioLORD-2023-C OK")


def download_green_model(model_name="StanfordAIMI/GREEN-RadLlama2-7b"):
    logger.info("Downloading GREEN model: %s (this is ~14GB)...", model_name)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    logger.info("  Tokenizer OK")
    AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype="auto",
    )
    logger.info("  Model OK")


def main():
    parser = argparse.ArgumentParser(description="Download evaluation metric models")
    parser.add_argument("--all", action="store_true", help="Download all models including GREEN (7B)")
    parser.add_argument("--chexbert", action="store_true", help="Download CheXbert dependencies")
    parser.add_argument("--ratescore", action="store_true", help="Download RaTEScore dependencies")
    parser.add_argument("--green", action="store_true", help="Download GREEN model (~14GB)")
    parser.add_argument("--green_model", type=str, default="StanfordAIMI/GREEN-RadLlama2-7b")
    args = parser.parse_args()

    if not any([args.all, args.chexbert, args.ratescore, args.green]):
        args.chexbert = True
        args.ratescore = True
        logger.info("No flags specified, downloading CheXbert + RaTEScore (use --all for GREEN too)")

    if args.all or args.chexbert:
        try:
            download_chexbert_deps()
        except Exception as e:
            logger.error("CheXbert download failed: %s", e)

    if args.all or args.ratescore:
        try:
            download_ratescore_deps()
        except Exception as e:
            logger.error("RaTEScore download failed: %s", e)

    if args.all or args.green:
        try:
            download_green_model(args.green_model)
        except Exception as e:
            logger.error("GREEN download failed: %s", e)

    logger.info("Done.")


if __name__ == "__main__":
    main()