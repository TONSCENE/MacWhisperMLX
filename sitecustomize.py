import os
import sys

# Only run compatibility patches if the main application is running (marked by our custom env variable)
if os.environ.get("MAC_WHISPER_RUNNING") == "1":
    # Allow OpenMP duplicate libraries to prevent kmp/omp initialisation crashes at runtime
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    try:
        import numpy as np
        if not hasattr(np, "NaN"):
            np.NaN = np.nan
        if not hasattr(np, "Infinity"):
            np.Infinity = np.inf
        if not hasattr(np, "infty"):
            np.infty = np.inf
        if not hasattr(np, "float"):
            np.float = float
        if not hasattr(np, "int"):
            np.int = int
        if not hasattr(np, "bool"):
            np.bool = bool
    except ImportError:
        pass

    try:
        import torchaudio
        if not hasattr(torchaudio, "set_audio_backend"):
            torchaudio.set_audio_backend = lambda *args, **kwargs: None
        if not hasattr(torchaudio, "list_audio_backends"):
            torchaudio.list_audio_backends = lambda *args, **kwargs: ["soundfile"]
    except ImportError:
        pass
