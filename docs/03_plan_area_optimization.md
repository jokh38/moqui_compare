### Development Plan: 03 - Area Optimization (Voxel-BVH & Dynamic LOD)

This document provides a development plan for fully integrating the Dynamic Level of Detail (LOD) system to actively manage geometric complexity during a simulation, thereby optimizing performance.

#### **1. Development Goal**

The goal is to transition from a static geometry model to a dynamic one where the level of detail is chosen on-the-fly. This involves integrating the `dynamic_lod_manager` into the main simulation loop to select the most appropriate geometry representation (e.g., a simple bounding box, the Voxel-BVH, or a full voxel grid) based on performance metrics or other heuristics. This requires making the particle transport kernel more flexible.

#### **2. Code Revision Guide**

The revision involves two main parts: changing the signature of the transport kernel to make it generic, and then modifying the simulation loop to use the LOD manager to select which geometry to pass to the kernel.

##### **A. Generalize the Transport Kernel**

*   **File for Revision**: `moqui_rev/moqui/kernel_functions/mqi_transport.hpp`
*   **Line for Revision**: Approximately **line 45** (the signature of `transport_particles_patient`).

*   **Before Revision**: The kernel is hardcoded to use a global `mc::mc_world` object.

    ```cpp
    // In moqui_rev/moqui/kernel_functions/mqi_transport.hpp, around line 45

    template <typename R>
    __global__ void
    transport_particles_patient(mqi::thrd_t*      worker_threads,
                                mqi::node_t<R>*   world, // This is currently mc::mc_world
                                mqi::vertex_t<R>* vertices,
                                size_t            n_particles,
                                uint32_t*         tracked_particles) 
    {
        // ... kernel logic uses 'world' ...
    }
    ```

*   **After Revision**: The kernel is modified to accept a pointer to the abstract `geometry_interface<R>`, which all geometry representations (including the Voxel-BVH) should inherit from.

    ```cpp
    // In moqui_rev/moqui/kernel_functions/mqi_transport.hpp, around line 45

    template <typename R>
    __global__ void
    transport_particles_patient(mqi::thrd_t*              worker_threads,
                                mqi::geometry_interface<R>* geometry, // REVISION: Use the abstract interface
                                mqi::vertex_t<R>*           vertices,
                                size_t                      n_particles,
                                uint32_t*                   tracked_particles) 
    {
        // ... kernel logic now uses 'geometry' ...
        // e.g., geometry->intersect(particle_pos, particle_dir);
    }
    ```

##### **B. Integrate LOD Manager into Simulation Loop**

*   **File for Revision**: `moqui_rev/moqui/base/environments/mqi_tps_env.hpp`
*   **Line for Revision**: Approximately **line 980** (within the `run_simulation` function).

*   **Before Revision**: The `run_simulation` function directly calls the transport kernel, implicitly using the globally defined `mc::mc_world`.

    ```cpp
    // In moqui_rev/moqui/base/environments/mqi_tps_env.hpp, around line 980

    mc::transport_particles_patient<R>
        <<<n_blocks, n_threads>>>(worker_threads, mc::mc_world, mc::mc_vertices,
                                  histories_in_batch, d_tracked_particles);
    ```

*   **After Revision**: The function now uses a `dynamic_lod_manager` to select and retrieve the appropriate geometry, which is then passed to the modified kernel.

    ```cpp
    // In moqui_rev/moqui/base/environments/mqi_tps_env.hpp, around line 980

    // Assumes 'lod_manager' is an initialized member of the tps_env class
    
    // 1. Get the optimal LOD for the current batch
    geometry_complexity_t optimal_lod = lod_manager.get_optimal_lod(...);
    lod_manager.set_complexity(optimal_lod);

    // 2. Get a pointer to the chosen geometry representation
    mqi::geometry_interface<R>* current_geometry = lod_manager.get_current_geometry();

    // 3. Pass the chosen geometry to the kernel
    mc::transport_particles_patient<R>
        <<<n_blocks, n_threads>>>(worker_threads, 
                                  current_geometry, // REVISION: Pass the dynamic geometry
                                  mc::mc_vertices,
                                  histories_in_batch, 
                                  d_tracked_particles);
    ```

#### **3. Impact and Guidance**

*   **Primary Impact**: This is a significant architectural change that affects the core of the simulation engine.
*   **Guidance**:
    *   **Abstract Interface is Key**: The success of this change hinges on all geometry classes (`grid3d`, `voxel_bvh_hybrid`, etc.) correctly inheriting from and implementing the `geometry_interface`. This ensures the transport kernel can interact with any of them polymorphically.
    *   **Data Management**: The `dynamic_lod_manager` must be responsible for managing the memory of the different geometry representations on the GPU. When a level of detail is not in use, its memory can potentially be freed or swapped.
    *   **Kernel Modification**: The `transport_particles_patient` kernel must be carefully refactored. All direct references to `mc::mc_world` must be replaced with calls to the `geometry` interface pointer passed as an argument. This is the most critical and potentially complex part of the implementation.
    *   **Incremental Approach**: It is recommended to implement this change incrementally. First, modify the kernel signature and pass the existing `mc::mc_world` through the new argument to ensure the refactoring works. Second, implement the `dynamic_lod_manager` with only two simple levels (e.g., a full grid and a single bounding box) to test the switching logic. Finally, integrate the Voxel-BVH as the medium level of detail.