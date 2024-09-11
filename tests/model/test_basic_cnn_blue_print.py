# tests/model/test_basic_cnn_blue_print.py

from source.model import VGGceptionCnnBluePrint

def test_basic_cnn_blue_print_build_model() -> None:
    blue_print = VGGceptionCnnBluePrint()
    model = blue_print.instantiate_model(795, 3, (72, 11), 16, 2, 2, 2)
    model.summary()