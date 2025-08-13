# engine/photonic_gbs.py
# Enhanced photonic GBS with Xanadu Cloud hardware support.
# Maps top-K embedding vectors -> small number of photonic modes via PCA,
# runs a Fock-basis simulation (Strawberry Fields) to sample photon counts,
# then projects mode activity back onto candidates to get a "community density" score.

from typing import Tuple, Optional
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

try:
    from .hardware.xanadu_integration import photonic_gbs as hw_gbs
    XANADU_AVAILABLE = True
except ImportError:
    XANADU_AVAILABLE = False
    logger.warning("Xanadu integration not available, using fallback implementation")

def _zscore(x: np.ndarray) -> np.ndarray:
    mu = float(np.mean(x))
    sd = float(np.std(x) + 1e-9)
    return (x - mu) / sd if sd else np.zeros_like(x)

def have_photonics() -> bool:
    try:
        import strawberryfields as sf  # noqa
        return True
    except Exception:
        return False

def pca_project(cand_vecs: np.ndarray, modes: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Z, var) where Z is (K, modes) PCA projection of cand_vecs and
    var are the per-mode variances used to set squeezing strengths."""
    X = cand_vecs - cand_vecs.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    W = Vt[:modes].T                  # (d, modes)
    Z = X @ W                         # (K, modes)
    var = np.var(Z, axis=0) + 1e-9
    return Z, var

def gbs_boost(seed_vec: np.ndarray,
              cand_vecs: np.ndarray,
              modes: int = 4,
              shots: int = 120,
              cutoff: int = 5,
              r_max: float = 0.45,
              use_hardware: bool = True,
              backend: Optional[str] = None) -> np.ndarray:
    """
    Returns a vector of length K with a photonic 'community density' score.
    Enhanced with Xanadu Cloud hardware support.
    
    Args:
        seed_vec: Reference vector (not used in current implementation)
        cand_vecs: Candidate vectors for GBS analysis
        modes: Number of photonic modes
        shots: Number of GBS samples
        cutoff: Fock space cutoff dimension
        r_max: Maximum squeezing parameter
        use_hardware: Whether to use Xanadu hardware if available
        backend: Specific hardware backend to use
    """
    K, d = cand_vecs.shape
    if K == 0:
        return np.zeros((K,), dtype=float)
    
    # Try Xanadu hardware-accelerated GBS
    if XANADU_AVAILABLE and use_hardware:
        try:
            # Set hardware usage in photonic GBS instance
            hw_gbs.use_hardware = use_hardware
            
            # Execute hardware GBS
            return hw_gbs.gbs_sampling(
                cand_vecs=cand_vecs,
                modes=modes,
                shots=shots,
                cutoff=cutoff,
                r_max=r_max,
                backend=backend
            )
            
        except Exception as e:
            logger.warning(f"Hardware GBS failed: {e}")
            logger.info("Falling back to local simulation")
    
    # Fallback to local Strawberry Fields simulation
    if not have_photonics():
        logger.warning("Strawberry Fields not available")
        return np.zeros((K,), dtype=float)

    import strawberryfields as sf
    from strawberryfields import ops as O

    # 1) Reduce to 'modes' dims via PCA
    m = max(2, min(modes, min(K, d, 8)))  # Hardware limit
    Z, var = pca_project(cand_vecs, m)

    # 2) Convert per-mode variance to squeezing params (keep small for laptops)
    #    r in [0, r_max] roughly proportional to sqrt(var) normalized.
    v = np.sqrt(var)
    r = (v / (np.max(v) + 1e-9)) * r_max

    # 3) Build simple Gaussian state: per-mode Sgate(r_i), measure Fock
    try:
        prog = sf.Program(m)
        with prog.context as q:
            for i in range(m):
                O.Sgate(float(r[i])) | q[i]
            O.MeasureFock() | q

        eng = sf.Engine("fock", backend_options={"cutoff_dim": int(cutoff)})
        res = eng.run(prog, shots=int(shots))
        # samples shape: (shots, m). Count photons per mode.
        counts = res.samples.astype(float)
        mode_activity = counts.mean(axis=0)  # (m,)

        # 4) Project mode activity back to candidates via |Z| weights
        weights = np.abs(Z)                   # (K,m)
        weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-9)
        cand_score = (weights @ mode_activity.reshape(-1, 1)).ravel()

        # Normalize for blending
        return _zscore(cand_score)
        
    except Exception as e:
        logger.error(f"Local GBS simulation failed: {e}")
        return np.zeros((K,), dtype=float)
