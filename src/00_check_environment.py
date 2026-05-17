import sys
import importlib.util


def check_package(package_name):
    """
    Check whether a Python package is installed.
    """
    spec = importlib.util.find_spec(package_name)

    if spec is None:
        return False

    return True


def main():
    print("=== Environment Check ===")
    print(f"Python version: {sys.version}")

    required_packages = [
        "numpy",
        "pandas",
        "sklearn",
        "torch",
        "transformers",
        "sentence_transformers",
        "tqdm",
        "matplotlib",
        "yaml",
    ]

    print("\n=== Package Check ===")

    for package_name in required_packages:
        is_installed = check_package(package_name)

        status = "OK" if is_installed else "MISSING"
        print(f"{package_name}: {status}")

    print("\n=== CUDA Check ===")

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

    except Exception as error:
        print(f"CUDA check failed: {error}")

    print("\nEnvironment check completed.")


if __name__ == "__main__":
    main()