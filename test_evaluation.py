from src.evaluation import needle_success


def test_needle_success_detects_passcode():
    assert needle_success("The secret passcode is ZX9QWERTY7731.", "ZX9QWERTY7731")
    assert needle_success("zx9-qwerty-7731", "ZX9QWERTY7731")
    assert not needle_success("I do not know the code.", "ZX9QWERTY7731")


if __name__ == "__main__":
    test_needle_success_detects_passcode()
    print("Stage 9 evaluation helper tests passed.")
