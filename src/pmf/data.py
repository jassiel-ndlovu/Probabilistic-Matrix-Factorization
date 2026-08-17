"""MovieLens acquisition, parsing and splitting.

Everything downstream consumes a :class:`RatingData` — a contiguous, zero-indexed
triple array plus the metadata needed to size a factor model. Keeping the
re-indexing here (rather than in the model, as the original implementation did)
is what makes ``n_users``/``n_items`` honest: they count entities that actually
appear, not ``max(id) + 1``.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np

__all__ = [
    "DATASETS",
    "RatingData",
    "Split",
    "load_movielens",
    "download",
    "train_test_split",
    "train_val_test_split",
    "describe",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data"

BASE_URL = "https://files.grouplens.org/datasets/movielens"


@dataclass(frozen=True)
class _Spec:
    """Static description of a MovieLens release."""

    key: str
    url: str
    dirname: str
    ratings_file: str
    sep: str
    items_file: str
    items_sep: str
    items_encoding: str
    n_ratings: int
    header: bool = False


DATASETS: dict[str, _Spec] = {
    "ml-100k": _Spec(
        key="ml-100k",
        url=f"{BASE_URL}/ml-100k.zip",
        dirname="ml-100k",
        ratings_file="u.data",
        sep="\t",
        items_file="u.item",
        items_sep="|",
        items_encoding="latin-1",
        n_ratings=100_000,
    ),
    "ml-1m": _Spec(
        key="ml-1m",
        url=f"{BASE_URL}/ml-1m.zip",
        dirname="ml-1m",
        ratings_file="ratings.dat",
        sep="::",
        items_file="movies.dat",
        items_sep="::",
        items_encoding="latin-1",
        n_ratings=1_000_209,
    ),
    "ml-10m": _Spec(
        key="ml-10m",
        url=f"{BASE_URL}/ml-10m.zip",
        dirname="ml-10M100K",
        ratings_file="ratings.dat",
        sep="::",
        items_file="movies.dat",
        items_sep="::",
        items_encoding="latin-1",
        n_ratings=10_000_054,
    ),
}


@dataclass
class RatingData:
    """A zero-indexed rating triple set.

    Attributes
    ----------
    triples:
        ``(n_ratings, 3)`` float array of ``[user_index, item_index, rating]``.
        Indices are contiguous in ``[0, n_users)`` and ``[0, n_items)``.
    n_users, n_items:
        Counts of *distinct entities present*, not ``max(raw_id) + 1``.
    rating_min, rating_max:
        Observed rating bounds, used for prediction clipping and for the
        logistic variant's ``t(x) = (x - min) / (max - min)`` rescaling.
    titles:
        Optional item-index -> title map, used only for qualitative output.
    """

    name: str
    triples: np.ndarray
    n_users: int
    n_items: int
    rating_min: float
    rating_max: float
    titles: dict[int, str] | None = None

    def __len__(self) -> int:
        return int(self.triples.shape[0])

    @property
    def users(self) -> np.ndarray:
        return self.triples[:, 0].astype(np.int64)

    @property
    def items(self) -> np.ndarray:
        return self.triples[:, 1].astype(np.int64)

    @property
    def ratings(self) -> np.ndarray:
        return self.triples[:, 2].astype(np.float64)

    @property
    def sparsity(self) -> float:
        """Fraction of the user-item matrix that is unobserved."""
        return 1.0 - len(self) / (self.n_users * self.n_items)

    def subset(self, mask: np.ndarray) -> RatingData:
        """Row-subset the triples, keeping the ambient index space unchanged."""
        return RatingData(
            name=self.name,
            triples=self.triples[mask],
            n_users=self.n_users,
            n_items=self.n_items,
            rating_min=self.rating_min,
            rating_max=self.rating_max,
            titles=self.titles,
        )


@dataclass
class Split:
    """A train/test (and optionally validation) partition of one dataset."""

    train: RatingData
    test: RatingData
    val: RatingData | None = None
    seed: int | None = None


def download(key: str, root: Path | str = DEFAULT_DATA_ROOT, force: bool = False) -> Path:
    """Fetch and extract a MovieLens release. Returns the extracted directory.

    Idempotent: if the ratings file is already on disk the network is not touched.
    """
    spec = _spec(key)
    root = Path(root)
    target = root / spec.dirname

    # Both files must be present: this repo was forked with only `u.data`
    # committed, so checking the ratings file alone silently skips the fetch and
    # leaves item titles unavailable.
    complete = (target / spec.ratings_file).exists() and (target / spec.items_file).exists()
    if complete and not force:
        return target

    root.mkdir(parents=True, exist_ok=True)
    print(f"[data] downloading {spec.url} ...")
    with urlopen(spec.url) as response:  # noqa: S310 - fixed, trusted URL
        payload = response.read()
    print(f"[data] got {len(payload) / 1e6:.1f} MB (sha256 {hashlib.sha256(payload).hexdigest()[:16]})")

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(root)

    if not (target / spec.ratings_file).exists():
        raise FileNotFoundError(
            f"expected {target / spec.ratings_file} after extracting {spec.url}"
        )
    return target


def load_movielens(
    key: str = "ml-100k",
    root: Path | str = DEFAULT_DATA_ROOT,
    auto_download: bool = True,
    with_titles: bool = True,
) -> RatingData:
    """Load a MovieLens release as contiguously re-indexed rating triples."""
    spec = _spec(key)
    root = Path(root)
    directory = root / spec.dirname
    ratings_path = directory / spec.ratings_file
    items_path = directory / spec.items_file

    if not ratings_path.exists() or (with_titles and not items_path.exists()):
        if not auto_download:
            if not ratings_path.exists():
                raise FileNotFoundError(
                    f"{ratings_path} not found; run `python scripts/download_data.py {key}`"
                )
        else:
            directory = download(key, root=root)
            ratings_path = directory / spec.ratings_file

    raw = _read_triples(ratings_path, spec.sep)

    user_ids, user_index = np.unique(raw[:, 0], return_inverse=True)
    item_ids, item_index = np.unique(raw[:, 1], return_inverse=True)

    triples = np.column_stack(
        [user_index.astype(np.float64), item_index.astype(np.float64), raw[:, 2]]
    )

    titles = None
    if with_titles:
        titles = _read_titles(directory / spec.items_file, spec, item_ids)

    data = RatingData(
        name=spec.key,
        triples=triples,
        n_users=int(user_ids.size),
        n_items=int(item_ids.size),
        rating_min=float(raw[:, 2].min()),
        rating_max=float(raw[:, 2].max()),
        titles=titles,
    )

    if len(data) != spec.n_ratings:
        print(
            f"[data] warning: {spec.key} has {len(data)} ratings, expected {spec.n_ratings}"
        )
    return data


def _read_triples(path: Path, sep: str) -> np.ndarray:
    """Parse ``user::item::rating::timestamp`` style files into a float array.

    ``np.loadtxt`` cannot handle MovieLens' multi-character ``::`` separator, so
    the 1M/10M files are normalised in a single pass first.
    """
    if sep == "\t":
        return np.loadtxt(path, delimiter="\t", usecols=(0, 1, 2), dtype=np.float64)

    with open(path, encoding="latin-1") as handle:
        text = handle.read().replace(sep, "\t")
    return np.loadtxt(io.StringIO(text), delimiter="\t", usecols=(0, 1, 2), dtype=np.float64)


def _read_titles(path: Path, spec: _Spec, item_ids: np.ndarray) -> dict[int, str] | None:
    """Map contiguous item index -> human-readable title, if the file is present."""
    if not path.exists():
        return None

    raw_to_title: dict[int, str] = {}
    with open(path, encoding=spec.items_encoding) as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(spec.items_sep)
            try:
                raw_to_title[int(parts[0])] = parts[1]
            except (ValueError, IndexError):
                continue

    return {
        index: raw_to_title[int(raw_id)]
        for index, raw_id in enumerate(item_ids)
        if int(raw_id) in raw_to_title
    }


def train_test_split(
    data: RatingData, test_size: float = 0.2, seed: int = 0
) -> Split:
    """Uniformly random split over observed ratings, reproducible given ``seed``.

    Note this is a *rating-level* split: a user may appear in both partitions.
    That is the protocol used by Salakhutdinov & Mnih and by the original report,
    so it is kept for comparability. Cold-start behaviour is instead probed by
    stratifying test error on training-set user activity (see
    ``pmf.experiments.cold_start``).
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    perm = rng.permutation(n)
    n_test = int(round(test_size * n))

    test_mask = np.zeros(n, dtype=bool)
    test_mask[perm[:n_test]] = True

    return Split(
        train=data.subset(~test_mask), test=data.subset(test_mask), seed=seed
    )


def train_val_test_split(
    data: RatingData, val_size: float = 0.1, test_size: float = 0.2, seed: int = 0
) -> Split:
    """Three-way split. Validation is carved out for early stopping and tuning,
    so the test partition is only ever touched for the final reported number."""
    rng = np.random.default_rng(seed)
    n = len(data)
    perm = rng.permutation(n)
    n_test = int(round(test_size * n))
    n_val = int(round(val_size * n))

    test_idx = perm[:n_test]
    val_idx = perm[n_test : n_test + n_val]
    train_idx = perm[n_test + n_val :]

    def mask_of(idx: np.ndarray) -> np.ndarray:
        m = np.zeros(n, dtype=bool)
        m[idx] = True
        return m

    return Split(
        train=data.subset(mask_of(train_idx)),
        val=data.subset(mask_of(val_idx)),
        test=data.subset(mask_of(test_idx)),
        seed=seed,
    )


def describe(data: RatingData) -> dict[str, float | int | str]:
    """Summary statistics reported in the paper's dataset table."""
    counts_user = np.bincount(data.users, minlength=data.n_users)
    counts_item = np.bincount(data.items, minlength=data.n_items)
    ratings = data.ratings
    return {
        "dataset": data.name,
        "n_users": data.n_users,
        "n_items": data.n_items,
        "n_ratings": len(data),
        "sparsity": round(data.sparsity, 6),
        "density_pct": round(100 * (1 - data.sparsity), 4),
        "rating_min": data.rating_min,
        "rating_max": data.rating_max,
        "rating_mean": round(float(ratings.mean()), 4),
        "rating_std": round(float(ratings.std()), 4),
        "ratings_per_user_mean": round(float(counts_user.mean()), 2),
        "ratings_per_user_median": float(np.median(counts_user)),
        "ratings_per_user_min": int(counts_user.min()),
        "ratings_per_item_mean": round(float(counts_item.mean()), 2),
        "ratings_per_item_median": float(np.median(counts_item)),
        "items_with_lt_5_ratings_pct": round(
            100 * float((counts_item < 5).mean()), 2
        ),
        "users_with_lt_20_ratings_pct": round(
            100 * float((counts_user < 20).mean()), 2
        ),
    }


def _spec(key: str) -> _Spec:
    try:
        return DATASETS[key]
    except KeyError:
        raise KeyError(f"unknown dataset {key!r}; choose from {sorted(DATASETS)}") from None
