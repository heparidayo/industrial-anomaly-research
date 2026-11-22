"""
외부에서 받은 사전 증류 Teacher 가중치를 우리 pdn.py 형식으로 변환하고 검증합니다.

공개 구현마다 저장 방식이 다릅니다.
  - nn.Sequential 순서 이름 (예: "0.weight", "3.weight", ...)
  - 층 이름 (예: "conv1.weight", "conv2.weight", ...)
  - state_dict를 {"state_dict": ...} 등으로 한 번 감싼 형태
  - 모델 객체 전체를 저장한 형태
이 스크립트는 형식과 관계없이 "합성곱 가중치를 순서대로" 꺼내, 우리 PDN의 합성곱
층과 모양(shape)이 정확히 맞는지 확인한 뒤 옮겨 담습니다. 하나라도 안 맞으면 멈춥니다.

사용 예:
  python convert_teacher.py --src teacher_small.pth --size s --out teacher_s.pth
  python convert_teacher.py --src teacher_medium.pth --size m --out teacher_m.pth

결과 파일은 train.py --teacher 에 그대로 쓸 수 있습니다.
"""

import argparse

import torch
import torch.nn as nn

from pdn import get_pdn


def load_any(path):
    # state_dict와 기존 Sequential PDN 체크포인트에 쓰이는 PyTorch 기본 층을 읽습니다.
    with torch.serialization.safe_globals([
        nn.Sequential, nn.Conv2d, nn.ReLU, nn.AvgPool2d, nn.Identity, nn.Dropout, set,
    ]):
        return torch.load(path, map_location="cpu", weights_only=True)


def to_state_dict(obj):
    if isinstance(obj, nn.Module):
        return obj.state_dict()
    if isinstance(obj, dict):
        for key in ("teacher", "state_dict", "model", "teacher_state_dict"):
            if key in obj and isinstance(obj[key], (dict, nn.Module)):
                print(f"  감싼 키 '{key}' 안의 가중치를 사용합니다.")
                return to_state_dict(obj[key])
        if all(torch.is_tensor(v) for v in obj.values()):
            return obj
    raise SystemExit(f"가중치를 찾지 못했습니다. 최상위 형식: {type(obj).__name__}, "
                     f"키: {list(obj.keys())[:10] if isinstance(obj, dict) else '-'}")


def conv_pairs(sd):
    """(weight, bias) 쌍을 저장 순서대로. 4차원 텐서를 합성곱 가중치로 봄."""
    keys = list(sd.keys())
    pairs = []
    for k in keys:
        t = sd[k]
        if torch.is_tensor(t) and t.dim() == 4 and k.endswith("weight"):
            bk = k[: -len("weight")] + "bias"
            pairs.append((k, t, sd.get(bk)))
    return pairs


def main():
    ap = argparse.ArgumentParser(description="외부 Teacher 가중치 → pdn.py 형식 변환")
    ap.add_argument("--src", required=True)
    ap.add_argument("--size", choices=["s", "m"], required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"읽는 중: {args.src}")
    src_sd = to_state_dict(load_any(args.src))
    pairs = conv_pairs(src_sd)

    target = get_pdn(args.size, out_channels=384, padding=True)
    tgt_convs = [(n, m) for n, m in target.named_modules() if isinstance(m, nn.Conv2d)]
    print(f"원본 합성곱 {len(pairs)}개, 우리 PDN-{args.size.upper()} 합성곱 {len(tgt_convs)}개")
    if len(pairs) != len(tgt_convs):
        raise SystemExit("합성곱 층 수가 다릅니다. --size가 맞는지(S: 4개, M: 6개) 확인하세요.")

    new_sd = {}
    for (src_k, w, b), (name, conv) in zip(pairs, tgt_convs):
        ok = tuple(w.shape) == tuple(conv.weight.shape)
        print(f"  {src_k:28s} {tuple(w.shape)} → {name}.weight {tuple(conv.weight.shape)}  {'OK' if ok else '불일치'}")
        if not ok:
            raise SystemExit("모양이 맞지 않습니다. 다른 구조의 가중치일 수 있습니다.")
        if b is None:
            raise SystemExit(f"{src_k}의 bias가 없습니다.")
        new_sd[f"{name}.weight"] = w.float()
        new_sd[f"{name}.bias"] = b.float()
    target.load_state_dict(new_sd)

    # 간단한 동작 확인: 출력 크기와 출력이 한 값으로 뭉개지지 않았는지
    target.eval()
    with torch.no_grad():
        y = target(torch.randn(4, 3, 256, 256))
    ch_std = y.std(dim=(0, 2, 3))
    print(f"출력 {tuple(y.shape)}, 채널별 표준편차 중앙값 {ch_std.median():.4f}, "
          f"0에 가까운 채널 {(ch_std < 1e-6).sum().item()}개")
    if not torch.isfinite(y).all():
        raise SystemExit("출력에 NaN/Inf가 있습니다.")

    torch.save({"teacher": target.state_dict(), "size": args.size, "source": str(args.src)}, args.out)
    print(f"저장: {args.out}  (train.py --teacher {args.out})")


if __name__ == "__main__":
    main()
