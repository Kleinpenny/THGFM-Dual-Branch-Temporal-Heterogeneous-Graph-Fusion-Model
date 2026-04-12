import importlib.util
import os
import sys


def _load_impl_module():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    oag_dir = os.path.abspath(os.path.join(this_dir, "..", ".."))
    if oag_dir not in sys.path:
        sys.path.insert(0, oag_dir)
    impl_path = os.path.join(oag_dir, "data_preprocess", "build_shared_pv_base.py")
    if not os.path.exists(impl_path):
        raise FileNotFoundError(f"Shared-base builder not found: {impl_path}")
    spec = importlib.util.spec_from_file_location("build_shared_pv_base_impl", impl_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from: {impl_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    module = _load_impl_module()
    module.main()


if __name__ == "__main__":
    main()
