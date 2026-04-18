from src import config


def test_as_dict_has_expected_keys() -> None:
    data = config.as_dict()
    assert data["PDF_DIR"] == str(config.PDF_DIR)
    assert data["CHROMA_DIR"] == str(config.CHROMA_DIR)
    assert data["CHUNK_SIZE"] == str(config.CHUNK_SIZE)
    assert data["CHUNK_OVERLAP"] == str(config.CHUNK_OVERLAP)


def test_ensure_dirs_creates_paths() -> None:
    config.ensure_dirs()
    assert config.PDF_DIR.exists()
    assert config.CHROMA_DIR.exists()
    assert config.BM25_DIR.exists()
