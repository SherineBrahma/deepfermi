from deepfermi.main import call_odin


def test_call_odin():
    assert call_odin("Odin")== "Hello Odin"