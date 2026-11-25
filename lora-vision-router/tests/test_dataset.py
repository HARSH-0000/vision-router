import pytest

from src.dataset import DOMAINS, get_domain_indices


def test_vehicle_indices_keep_original_cifar10_labels():
    targets = list(range(10)) * 2
    assert get_domain_indices(targets, "vehicles") == [0, 1, 8, 9, 10, 11, 18, 19]


def test_animal_indices_keep_original_cifar10_labels():
    targets = list(range(10))
    assert get_domain_indices(targets, "animals") == [2, 3, 4, 5, 6, 7]


def test_all_domains_share_the_same_ten_class_classifier():
    assert set(DOMAINS["vehicles"]["original_ids"]).isdisjoint(
        DOMAINS["animals"]["original_ids"]
    )
    assert sorted(
        DOMAINS["vehicles"]["original_ids"] + DOMAINS["animals"]["original_ids"]
    ) == list(range(10))


def test_unknown_domain_is_rejected():
    with pytest.raises(ValueError, match="Unknown domain"):
        get_domain_indices([0, 1, 2], "planes")
