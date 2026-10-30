import pytest

from inventario.config import cargar_yaml, costos_para, cuantil_objetivo


def test_razon_critica_perecedero_menor_que_alta_rotacion():
    cfg = cargar_yaml("costos.yaml")
    assert cuantil_objetivo("GROCERY I", cfg) > 0.5 > cuantil_objetivo("DAIRY", cfg)


def test_familia_desconocida_usa_default():
    cfg = cargar_yaml("costos.yaml")
    assert costos_para("FAMILIA_NUEVA", cfg) == (2.0, 1.0)


def test_cambiar_costos_cambia_cuantil_sin_tocar_codigo():
    cfg = {"default": {"costo_quiebre": 1.0, "costo_merma": 1.0}}
    q1 = cuantil_objetivo("X", cfg)
    cfg["default"]["costo_quiebre"] = 9.0
    assert cuantil_objetivo("X", cfg) > q1


def test_costos_no_positivos_fallan():
    with pytest.raises(ValueError):
        costos_para("X", {"default": {"costo_quiebre": 0, "costo_merma": 1}})
