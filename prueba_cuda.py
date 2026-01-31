import torch
import qiskit
from qiskit_aer import AerSimulator

print("-" * 30)
print(f"PyTorch Version: {torch.__version__}")

# LA PREGUNTA DEL MILLÓN
cuda_ok = torch.cuda.is_available()
print(f"¿CUDA Disponible?: {cuda_ok}")

if cuda_ok:
    print(f"Dispositivo actual: {torch.cuda.get_device_name(0)}")
    print("¡Felicidades! Usarás la GPU para el entrenamiento RL.")
else:
    print("¡OJO! Estás usando CPU. Algo falló en la instalación.")

print("-" * 30)
# Prueba rápida de Qiskit
qc = qiskit.QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)
print("Circuito cuántico creado correctamente.")
print("-" * 30)