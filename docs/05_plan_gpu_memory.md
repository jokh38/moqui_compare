### Development Plan: 05 - GPU Memory Optimization

This document provides a development plan for integrating the advanced GPU memory optimization architecture designed in `moqui_rev`. This is a future development task aimed at improving performance by changing how data is allocated and accessed on the GPU.

#### **1. Development Goal**

The goal is to refactor the GPU data management to use the features defined in `mqi_memory_optimization.hpp`. This involves two main changes:
1.  Uploading small, frequently-accessed, and unchanging data (like physics constants) to the GPU's fast, cached **constant memory**.
2.  Replacing frequent `cudaMalloc` calls with a **segmented memory pool** that pre-allocates large chunks of memory, organizing data by access pattern to improve cache efficiency and reduce allocation overhead.

#### **2. Code Revision Guide**

The revision will focus on the data upload functions and the main simulation initialization sequence.

*   **Files for Revision**:
    1.  `moqui_rev/moqui/kernel_functions/mqi_upload_data.hpp` (for data allocation)
    2.  `moqui_rev/moqui/base/environments/mqi_tps_env.hpp` (for pool management)

##### **A. Before Revision**

In `mqi_upload_data.hpp`, the `upload_node` function (around line 200) performs numerous individual `cudaMalloc` calls for each component of the geometry. This is inefficient.

```cpp
// In moqui_rev/moqui/kernel_functions/mqi_upload_data.hpp, around line 205

template <typename R>
void upload_node(mqi::node_t<R> *c_node, mqi::node_t<R> *&g_node) {
    // ...
    mqi::density_t *density = nullptr;
    mqi::vec3<mqi::ijk_t> dim = c_node->geo->get_nxyz();

    // Inefficient, repeated calls to cudaMalloc
    gpu_err_chk(cudaMalloc(&x_edges, (dim.x + 1) * sizeof(R)));
    gpu_err_chk(cudaMalloc(&y_edges, (dim.y + 1) * sizeof(R)));
    gpu_err_chk(cudaMalloc(&z_edges, (dim.z + 1) * sizeof(R)));
    gpu_err_chk(
        cudaMalloc(&density, (dim.x * dim.y * dim.z) * sizeof(mqi::density_t)));
    // ... and many more allocations ...
}
```

##### **B. After Revision (Proposed)**

The `upload_node` function is refactored to receive a memory pool object and use its allocators. The main environment (`tps_env`) is responsible for creating this pool.

```cpp
// In moqui_rev/moqui/base/environments/mqi_tps_env.hpp

// 1. ADD memory pool management to the environment
class tps_env : public x_environment<R> {
public:
    mqi::segmented_memory_pool_t mem_pool_;
    // ...

    // In the constructor or an init function:
    initialize_phase2_memory_optimization(&mem_pool_, 32, 64, false, this->gpu_id);

    // In the destructor:
    shutdown_phase2_memory_optimization(&mem_pool_);

    // Before simulation, upload to constant memory:
    mqi::physics_constants_device_t host_consts = {...};
    upload_physics_constants_to_constant_memory(&host_consts);
};


// In moqui_rev/moqui/kernel_functions/mqi_upload_data.hpp

// 2. REVISE upload_node to use the memory pool
template <typename R>
void upload_node(mqi::node_t<R>* c_node, mqi::node_t<R>*& g_node, mqi::segmented_memory_pool_t* mem_pool) {
    // ...
    mqi::density_t* density = nullptr;
    mqi::vec3<mqi::ijk_t> dim = c_node->geo->get_nxyz();

    // REVISION: Use the pool allocator instead of cudaMalloc.
    // Geometry data is read-only during transport, so the physics pool is appropriate.
    density = (mqi::density_t*) allocate_from_physics_pool(
        mem_pool, 
        (dim.x * dim.y * dim.z) * sizeof(mqi::density_t));
    
    // ... repeat for other allocations (x_edges, y_edges, etc.) ...

    // The cudaMemcpy calls remain the same
    gpu_err_chk(cudaMemcpy(density, c_node->geo->get_data(), (dim.x * dim.y * dim.z) * sizeof(mqi::density_t), cudaMemcpyHostToDevice));
    // ...
}
```

#### **3. Impact and Guidance**

*   **Primary Impact**: Performance. This change directly addresses a major performance bottleneck in GPU applications: memory allocation overhead and inefficient memory access.
*   **Guidance**:
    *   **Pool Management**: The `tps_env` class must correctly manage the lifecycle of the `segmented_memory_pool_t` object, ensuring it is initialized before any uploads occur and shut down properly to free GPU memory. The memory pool object should be passed down through the function calls to where allocations are needed (e.g., `initialize()` -> `setup_world()` -> `upload_node()`).
    *   **Constant Memory Access**: After uploading data to constant memory (e.g., `g_physics_constants`), the GPU kernels in `mqi_transport.hpp` must be updated to read from these constant memory variables instead of from pointers to global memory. This is a simple but critical change to realize the performance benefit.
    *   **Data Categorization**: When converting `cudaMalloc` calls to use the pool, carefully consider the data's access pattern.
        *   `allocate_from_physics_pool`: For read-only data like geometry, material tables, etc.
        *   `allocate_from_particle_pool`: For data that is frequently read and written, like particle track and vertex information.
        *   `allocate_from_statistics_pool`: For scorer data where results are accumulated.
    *   **No Logic Change**: This is a pure performance refactoring. No changes to the core physics or simulation algorithms are required. The goal is to change *how* memory is allocated, not *what* is allocated.