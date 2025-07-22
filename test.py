import warp as wp
import numpy as np
import newton

# Initialize Warp
wp.init()

# Define a simple kernel
@wp.kernel
def add_vectors(a: wp.array(dtype=float), b: wp.array(dtype=float), result: wp.array(dtype=float)):
    tid = wp.tid()
    result[tid] = a[tid] + b[tid]

# Simulation setup
def run_simulation():
    # Define vector size
    n = 10

    # Create input data
    a = np.random.rand(n).astype(np.float32)
    b = np.random.rand(n).astype(np.float32)

    # Allocate device memory
    a_device = wp.array(a, dtype=float)
    b_device = wp.array(b, dtype=float)
    result_device = wp.zeros(n, dtype=float)

    # Launch kernel
    wp.launch(kernel=add_vectors, dim=n, inputs=[a_device, b_device, result_device])

    # Copy result back to host
    result = result_device.numpy()

    # Print results
    print("Vector A:", a)
    print("Vector B:", b)
    print("Result (A + B):", result)

if __name__ == "__main__":
    run_simulation()