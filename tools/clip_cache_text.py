#!/usr/bin/env python3
import argparse
import json
import os
import os.path as osp
import sys

import types

import torch


def _ensure_ftfy():
    try:
        import ftfy  # noqa: F401
        return
    except Exception:
        pass

    stub = types.ModuleType("ftfy")

    def _fix_text(text):
        return text

    stub.fix_text = _fix_text
    sys.modules["ftfy"] = stub


def _ensure_regex():
    try:
        import regex  # noqa: F401
        return
    except Exception:
        pass

    import re as _re

    def _fix_pattern(pattern):
        if hasattr(pattern, "pattern"):
            return pattern
        return pattern.replace(r"\p{L}", "A-Za-z").replace(r"\p{N}", "0-9")

    stub = types.ModuleType("regex")
    stub.IGNORECASE = _re.IGNORECASE

    def _compile(pattern, flags=0):
        fixed = _fix_pattern(pattern)
        if hasattr(fixed, "pattern"):
            return fixed
        return _re.compile(fixed, flags)

    def _sub(pattern, repl, string, count=0, flags=0):
        fixed = _fix_pattern(pattern)
        if hasattr(fixed, "pattern"):
            return _re.sub(fixed, repl, string, count=count)
        return _re.sub(fixed, repl, string, count=count, flags=flags)

    def _findall(pattern, string, flags=0):
        fixed = _fix_pattern(pattern)
        if hasattr(fixed, "pattern"):
            return _re.findall(fixed, string)
        return _re.findall(fixed, string, flags)

    stub.compile = _compile
    stub.sub = _sub
    stub.findall = _findall
    sys.modules["regex"] = stub


def _ensure_clip_import(clip_repo):
    try:
        import clip  # noqa: F401
        return
    except Exception:
        pass

    _ensure_ftfy()
    _ensure_regex()

    if clip_repo is None:
        raise ImportError(
            "Could not import clip. Provide --clip-repo pointing to OpenAI/CLIP."
        )
    clip_repo = osp.expanduser(clip_repo)
    if not osp.isdir(clip_repo):
        raise FileNotFoundError(f"CLIP repo not found: {clip_repo}")
    sys.path.insert(0, clip_repo)
    try:
        import clip  # noqa: F401
    except Exception as exc:
        raise ImportError(
            f"Failed to import clip from {clip_repo}: {exc}"
        ) from exc


def _get_hrsc_classes(classwise):
    from mmrotate.datasets import HRSCDataset

    if classwise:
        return list(HRSCDataset.HRSC_CLASSES)
    return list(HRSCDataset.HRSC_CLASS)


def _build_prompts(classes, template):
    return [template.format(cls_name=cls_name) for cls_name in classes]


def main():
    parser = argparse.ArgumentParser(
        description="Cache CLIP text embeddings for HRSC classes."
    )
    parser.add_argument(
        "--clip-repo",
        default="~/workspace/CLIP",
        help="Path to OpenAI/CLIP repo.",
    )
    parser.add_argument(
        "--model",
        default="ViT-B/32",
        help="CLIP model name.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for CLIP text encoder.",
    )
    parser.add_argument(
        "--classwise",
        action="store_true",
        help="Use 31-class HRSC classwise labels.",
    )
    parser.add_argument(
        "--template",
        default="a {cls_name} in an aerial image",
        help="Prompt template; use {cls_name} placeholder.",
    )
    parser.add_argument(
        "--out-dir",
        default="resources/clip",
        help="Output directory for cached embeddings.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 for CLIP text encoding.",
    )
    args = parser.parse_args()

    _ensure_clip_import(args.clip_repo)
    import clip

    device = torch.device(args.device)
    model, _ = clip.load(args.model, device=device, jit=False)
    model.eval()
    if args.fp16:
        model = model.half()

    classes = _get_hrsc_classes(args.classwise)
    prompts = _build_prompts(classes, args.template)
    tokens = clip.tokenize(prompts).to(device)

    with torch.no_grad():
        text_features = model.encode_text(tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    text_features = text_features.detach().cpu()

    os.makedirs(args.out_dir, exist_ok=True)
    class_tag = "hrsc_classwise" if args.classwise else "hrsc_ship"
    stem = f"{class_tag}_{args.model.replace('/', '-')}"
    emb_path = osp.join(args.out_dir, f"{stem}.pt")
    meta_path = osp.join(args.out_dir, f"{stem}.json")

    torch.save(text_features, emb_path)
    meta = {
        "dataset": "HRSC2016",
        "classwise": args.classwise,
        "classes": classes,
        "template": args.template,
        "model": args.model,
        "embedding_shape": list(text_features.shape),
        "normalized": True,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=True)

    norms = text_features.norm(dim=-1)
    print(f"Saved: {emb_path}")
    print(f"Saved: {meta_path}")
    print(f"Embedding shape: {tuple(text_features.shape)}")
    print(
        f"Norm stats: min={norms.min().item():.4f}, "
        f"max={norms.max().item():.4f}, mean={norms.mean().item():.4f}"
    )


if __name__ == "__main__":
    main()
