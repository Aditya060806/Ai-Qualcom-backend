"""Execution-provider / NPU verification and QNN registration.

On Snapdragon X the QNN execution provider ships in the separate, Qualcomm-authored
``onnxruntime_qnn`` package as a *plugin EP*: it is NOT auto-listed by
``get_available_providers()``. You register the shipped
``onnxruntime_providers_qnn.dll`` with ONNX Runtime, after which the NPU shows up
in ``get_ep_devices()``. That registration is exactly what ``register_qnn()`` does,
and ``verify_npu()`` performs it before enumerating so the boot log proves the NPU
is reachable.

Everything here is defensive: in mock mode (or before the packages are installed)
it simply reports that the runtime/EP is unavailable and never raises.
"""
from __future__ import annotations

from typing import Any, Dict

_QNN_REGISTERED = False


def register_qnn() -> Dict[str, Any]:
    """Register the QNN plugin EP library. Idempotent within a process."""
    global _QNN_REGISTERED
    result: Dict[str, Any] = {"registered": False, "library_path": None, "error": None}

    try:
        import onnxruntime as ort
        import onnxruntime_qnn as oq
    except Exception as e:
        result["error"] = f"import failed: {e}"
        return result

    try:
        library_path = oq.get_library_path()
        registration_name = oq.get_ep_name()
    except Exception as e:
        result["error"] = f"onnxruntime_qnn helper failed: {e}"
        return result

    result["library_path"] = library_path

    if _QNN_REGISTERED:
        result["registered"] = True
        return result

    try:
        ort.register_execution_provider_library(registration_name, library_path)
        _QNN_REGISTERED = True
        result["registered"] = True
    except Exception as e:
        # A second registration in the same process reports "already registered".
        if "already" in str(e).lower():
            _QNN_REGISTERED = True
            result["registered"] = True
        else:
            result["error"] = str(e)

    return result


def verify_npu() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "onnxruntime_available": False,
        "onnxruntime_version": None,
        "qnn_package_available": False,
        "qnn_registered": False,
        "qnn_library_path": None,
        "providers": [],
        "ep_devices": [],
        "qnn_available": False,
        "error": None,
    }

    try:
        import onnxruntime as ort
    except Exception as e:  # onnxruntime not installed (e.g. mock mode)
        info["error"] = f"import onnxruntime failed: {e}"
        return info

    info["onnxruntime_available"] = True
    info["onnxruntime_version"] = getattr(ort, "__version__", None)

    # Classic provider list (QNN plugin EP will NOT appear here; that's expected).
    try:
        info["providers"] = list(ort.get_available_providers())
    except Exception as e:
        info["error"] = f"get_available_providers: {e}"

    # Register the QNN plugin EP if the Qualcomm package is present.
    reg = register_qnn()
    info["qnn_package_available"] = reg["library_path"] is not None
    info["qnn_registered"] = reg["registered"]
    info["qnn_library_path"] = reg["library_path"]
    if reg["error"]:
        info["error"] = reg["error"]

    # Enumerate EP devices (ONNX Runtime >= 1.22). QNN appears here once registered.
    get_ep_devices = getattr(ort, "get_ep_devices", None)
    if callable(get_ep_devices):
        devices = []
        try:
            for d in get_ep_devices():
                device = getattr(d, "device", None)
                device_type = getattr(device, "type", None)
                devices.append(
                    {
                        "ep_name": getattr(d, "ep_name", None),
                        "ep_vendor": getattr(d, "ep_vendor", None),
                        "device_type": str(device_type) if device_type is not None else None,
                    }
                )
        except Exception as e:
            info["error"] = f"get_ep_devices: {e}"
        info["ep_devices"] = devices
        if any("QNN" in (dev.get("ep_name") or "").upper() for dev in devices):
            info["qnn_available"] = True

    # Fallback: some builds surface QNN via the classic list too.
    if any("QNN" in p for p in info["providers"]):
        info["qnn_available"] = True

    return info
