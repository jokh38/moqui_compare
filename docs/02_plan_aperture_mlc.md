### Development Plan: 02 - Voxelized Aperture/MLC

This document provides a development plan for implementing a full 3D voxelized geometry for apertures and MLCs, replacing the current simplified 2D plane model.

#### **1. Development Goal**

The goal is to complete the feature envisioned in the `moqui_rev` architecture by implementing the `create_voxelized_aperture` function. This will create a physically realistic, 3D aperture object in the simulation world, allowing for more accurate modeling of particle interactions, such as scattering and absorption within the aperture body.

#### **2. Code Revision Guide**

The primary revision will take place in `mqi_tps_env.hpp`, where the current `BLOCK` geometry is a placeholder.

*   **File for Revision**: `moqui_rev/moqui/base/environments/mqi_tps_env.hpp`
*   **Line for Revision**: Approximately **line 788**.

##### **A. Before Revision**

Currently, the code block for handling `mqi::BLOCK` geometries contains a `TODO` and is commented out, meaning no special geometry is created for apertures.

```cpp
// In moqui_rev/moqui/base/environments/mqi_tps_env.hpp, around line 788

} else if (beamline_geometries[i]->geotype == mqi::BLOCK) {
    /// TODO: defining voxelized aperture
    // beamline_objects[i+1] = this->create_voxelixed_aperture(...);
}
```

##### **B. After Revision**

The plan is to implement a `create_voxelized_aperture` function and call it from this block.

```cpp
// In moqui_rev/moqui/base/environments/mqi_tps_env.hpp

// 1. ADD the new function to the tps_env class (e.g., around line 1150)
virtual mqi::node_t<R>*
create_voxelized_aperture(mqi::aperture* aperture_geom, mqi::coordinate_transform<R> p_coord) {
    mqi::node_t<R>* aperture_node = new mqi::node_t<R>;
    aperture_node->n_scorers = 0;
    aperture_node->n_children = 0;
    aperture_node->scorers = nullptr;
    aperture_node->children = nullptr;

    // Use the 3D position and thickness from the parsed DICOM info (aperture_geom)
    aperture_node->geo = new grid3d<mqi::density_t, R>(
        aperture_geom->pos.x - aperture_geom->volume.x / 2, aperture_geom->pos.x + aperture_geom->volume.x / 2, 2,
        aperture_geom->pos.y - aperture_geom->volume.y / 2, aperture_geom->pos.y + aperture_geom->volume.y / 2, 2,
        aperture_geom->pos.z - aperture_geom->volume.z / 2, aperture_geom->pos.z + aperture_geom->volume.z / 2, 2,
        p_coord.rotation);

    // Fill the entire volume with a high-density material (e.g., Brass)
    aperture_node->geo->fill_data(mqi::brass_t<R>().rho_mass);
    aperture_node->geo->translation_vector = p_coord.translation;

    /*
     * CRITICAL NEXT STEP: A separate algorithm is needed here to "carve out" the opening.
     * This algorithm would iterate through the voxels of the grid3d and set the density
     * of any voxel whose center falls within the 2D opening contour (aperture_geom->xypts)
     * to a low-density material like air (mqi::air_t<R>().rho_mass).
     */

    printf("Created Voxelized Aperture Geometry.\n");
    return aperture_node;
}


// 2. REVISE the code block around line 788
} else if (beamline_geometries[i]->geotype == mqi::BLOCK) {
    // REVISION: Uncomment and implement the function call
    beamline_objects[i] = this->create_voxelized_aperture(
        dynamic_cast<mqi::aperture*>(beamline_geometries[i]), p_coord);
}
```

#### **3. Impact and Guidance**

*   **Primary Impact**: The particle transport kernel in `mqi_transport.hpp`.
*   **Guidance**:
    *   The transport kernel is already designed to handle `grid3d` geometries, as it does for the patient phantom. Therefore, no changes to the transport *logic* are required. The kernel will naturally begin to simulate particle transport through this new, high-density `grid3d` object placed in the beamline.
    *   **Performance Consideration**: This change will add a significant number of voxels to the simulation geometry, increasing the computational workload. It is crucial that this work is done in conjunction with the Voxel-BVH and Dynamic LOD optimizations to ensure simulation times remain practical. The BVH will be essential for quickly skipping over the solid parts of the aperture when particles are far from it.
    *   **Carve-out Algorithm**: The most complex part of this task is implementing the "carve-out" logic. This will require a robust point-in-polygon test to determine which voxels of the 3D grid fall inside the 2D aperture opening defined in the DICOM plan. This algorithm must be efficient and accurate.