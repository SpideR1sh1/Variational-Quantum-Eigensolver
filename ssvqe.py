import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
import numpy as np
import random
import matplotlib.pyplot as plt
from copy import deepcopy
from functools import reduce
import operator


# Define a single layer of the quantum circuit ansatz
def build_layer(circuit, qubits, parameters):
    for idx, qubit in enumerate(qubits):
        # Apply rotational gates parameterized by `parameters`
        circuit.append([
            cirq.ry(parameters[3 * idx]).on(qubit),
            cirq.rz(parameters[3 * idx + 1]).on(qubit),
            cirq.ry(parameters[3 * idx + 2]).on(qubit)
        ])
    # Add entangling CNOT gates
    for idx in range(len(qubits) - 1):
        circuit.append(cirq.CNOT(qubits[idx], qubits[idx + 1]))
    circuit.append(cirq.CNOT(qubits[-1], qubits[0]))  # Close the loop
    return circuit


# Construct the full ansatz with multiple layers
def construct_ansatz(circuit, qubits, num_layers, parameters):
    for layer_idx in range(num_layers):
        # Slice out the parameters for the current layer
        layer_params = parameters[3 * layer_idx * len(qubits):3 * (layer_idx + 1) * len(qubits)]
        circuit = build_layer(circuit, qubits, layer_params)
    return circuit


# Apply measurement basis transformation based on the Hamiltonian
def apply_measurement_basis(circuit, qubits, hamiltonian):
    for idx, term in enumerate(hamiltonian):
        if term == "x":
            circuit.append(cirq.ry(-np.pi / 2).on(qubits[idx]))
        elif term == "y":
            circuit.append(cirq.rx(np.pi / 2).on(qubits[idx]))
    return circuit


# Combine ansatz and measurement to form the full VQE circuit
def create_vqe_circuit(init_circuit, qubits, layers, parameters, hamiltonian):
    circuit = construct_ansatz(init_circuit, qubits, layers, parameters)
    circuit = apply_measurement_basis(circuit, qubits, hamiltonian)
    return circuit


# Utility function for product of operators
def product_of_operators(operators):
    return reduce(operator.mul, operators, 1)


# Define the cost function for the Hamiltonian
def hamiltonian_expectation_cost(qubits, hamiltonian):
    return product_of_operators([cirq.Z(qubits[idx]) for idx, term in enumerate(hamiltonian) if term != "i"])


# Calculate exact eigenvalues of a Hamiltonian matrix for validation
def compute_exact_eigenvalues(hamiltonian_terms, weights, num_eigenvalues):
    def map_to_matrix(term):
        return {
            "x": np.array([[0, 1], [1, 0]]),
            "y": np.array([[0, -1j], [1j, 0]]),
            "z": np.array([[1, 0], [0, -1]]),
            "i": np.eye(2)
        }[term]

    # Convert Hamiltonian terms to matrices
    matrices = [[map_to_matrix(term) for term in terms] for terms in hamiltonian_terms]

    # Build full Hamiltonian matrix
    full_matrix = sum(weight * reduce(np.kron, matrix) for weight, matrix in zip(weights, matrices))
    eigenvalues = np.real(np.linalg.eigvals(full_matrix))

    return sorted(eigenvalues) if num_eigenvalues == "all" else sorted(eigenvalues)[:num_eigenvalues]


# Subclass for VQE components
class VQEComponent(tf.keras.layers.Layer):
    def __init__(self, circuits, operators):
        super().__init__()
        self.layers = [
            tfq.layers.ControlledPQC(circuit, operator, repetitions=1000, differentiator=tfq.differentiators.ParameterShift())
            for circuit, operator in zip(circuits, operators)
        ]

    def call(self, inputs):
        return sum(layer([inputs[0], inputs[1]]) for layer in self.layers)


# SSVQE model to compute multiple eigenvalues
class SSVQEModel(tf.keras.layers.Layer):
    def __init__(self, num_params, circuits, operators, num_eigenvalues):
        super().__init__()
        self.weights = tf.Variable(np.random.uniform(0, np.pi, (1, num_params)), dtype=tf.float32)
        self.vqe_subsystems = [
            VQEComponent(circuits[i], operators[i]) for i in range(num_eigenvalues)
        ]
        self.num_eigenvalues = num_eigenvalues

    def call(self, inputs):
        total_cost = 0
        eigenvalue_estimations = []

        for idx, subsystem in enumerate(self.vqe_subsystems):
            current_estimate = subsystem([inputs, self.weights])
            eigenvalue_estimations.append(current_estimate)
            penalty = 0.65 - (idx - 1) * 0.1 if idx > 0 else 1.0
            total_cost += penalty * current_estimate

        return total_cost, eigenvalue_estimations


# Initialize parameters
num_layers = 3
num_qubits = 3
num_terms = 4
num_eigenvalues = 3

qubits = [cirq.GridQubit(0, i) for i in range(num_qubits)]
parameters = sympy.symbols(f'param0:{num_layers * 3 * num_qubits}')

hamiltonian_terms = [["x", "i", "i"], ["i", "x", "i"], ["i", "z", "z"], ["i", "i", "x"]]
hamiltonian_weights = [0.497, 0.563, 0.326, 0.189]

# Prepare circuits and operators
circuits, operators = [], []
for eigen_idx in range(num_eigenvalues):
    eigen_circuits = []
    eigen_operators = []

    for term, weight in zip(hamiltonian_terms, hamiltonian_weights):
        circuit = create_vqe_circuit(cirq.Circuit(), qubits, num_layers, parameters, term)
        eigen_circuits.append(circuit)
        eigen_operators.append(weight * hamiltonian_expectation_cost(qubits, term))

    circuits.append(eigen_circuits)
    operators.append(eigen_operators)

# Build and train the model
inputs = tf.keras.layers.Input(shape=(), dtype=tf.string)
ssvqe_model = SSVQEModel(len(parameters), circuits, operators, num_eigenvalues)
vqe_model = tf.keras.models.Model(inputs=inputs, outputs=ssvqe_model(inputs)[0])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
epochs = 140

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        loss, estimates = vqe_model(inputs)
    gradients = tape.gradient(loss, vqe_model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vqe_model.trainable_variables))

    if epoch % 20 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}, Estimates: {[e.numpy() for e in estimates]}")

# Plot the results
real_eigenvalues = compute_exact_eigenvalues(deepcopy(hamiltonian_terms), hamiltonian_weights, num_eigenvalues)
plt.figure(figsize=(10, 6))
plt.plot(real_eigenvalues, label="Exact Eigenvalues")
plt.legend()
plt.show()
