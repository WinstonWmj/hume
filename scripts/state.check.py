import os

import safetensors.torch as st


def print_shapes(state_dict, keyword: str | None = None):
    """按名字排序打印所有权重的 shape，可选按关键字过滤。"""
    items = sorted(state_dict.items(), key=lambda x: x[0])
    for name, tensor in items:
        if keyword is not None and keyword not in name:
            continue
        # 有些 entry 不是 tensor（例如 metadata），直接跳过
        if not hasattr(tensor, "shape"):
            continue
        print(f"{name:80s}  {list(tensor.shape)}")


def main():
    model_dir = "/mnt/mnt/public_zgc/models/Hume-vla/Hume-System2"
    ckpt_path = os.path.join(model_dir, "model.safetensors")

    print(f"模型目录: {model_dir}")
    print(f"权重文件: {ckpt_path}")

    print("\n=== 从 model.safetensors 读取 state_dict ===")
    sd = st.load_file(ckpt_path, device="cpu")
    print(f"参数条目总数: {len(sd)}")

    print("\n=== 所有参数名及其 shape（按名字排序）=== ")
    print_shapes(sd)

    print("\n=== 只看归一化相关的权重（名字里含 normalize / unnormalize）=== ")
    print("\n-- normalize* --")
    print_shapes(sd, "normalize")
    print("\n-- unnormalize* --")
    print_shapes(sd, "unnormalize")


if __name__ == "__main__":
    main()

