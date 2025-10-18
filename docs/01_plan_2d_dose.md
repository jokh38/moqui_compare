### Development Plan: 01 - 2D Dose Calculation

This document details the implementation of the `TwoCentimeterMode` feature, which is new in `moqui_rev` and did not exist in `moqui_v0`. It provides a guide to how the feature was implemented and how it impacts different parts of the codebase.

#### **1. Feature Implementation Guide**

The `TwoCentimeterMode` is enabled by a boolean flag in the input file, which triggers specialized logic in two key areas: geometry creation and DICOM file saving.

##### **A. Geometry Creation (`mqi_tps_env.hpp`)**

*   **Background**: To calculate a 2D dose slice efficiently, `moqui_rev` dynamically creates a virtual water phantom where the scoring volume is a very thin slice at the target depth.
*   **File for Revision**: `moqui_rev/moqui/base/environments/mqi_tps_env.hpp`
*   **Code Example**: The core logic is inside the `setup_world` function.

    *   **Code to add (conceptual `moqui_v0` -> `moqui_rev` change):**
        Within the `setup_world` function, around **line 805**, logic was added to handle `twoCentimeterMode`.

        ```cpp
        // In moqui_rev/moqui/base/environments/mqi_tps_env.hpp, around line 805

        // ... existing code to setup world ...
        
        // REVISION START: Add specialized logic for 2D dose mode
        if (this->twoCentimeterMode) {
            // Create a multi-part virtual phantom:
            // 1. Front Phantom (water up to just before the scoring slice)
            this->world->children[beamline_geometries.size() + 1] = frontPhantom;
            frontPhantom->geo =
                new grid3d<density_t, R>(-200, 200, 401, -200, 200, 401, 1, 20, 2,
                                         transformPhantom.rotation);
            frontPhantom->geo->set_data(rho_mass_parent1);

            // 2. Scoring Phantom (the thin slice at 2cm where dose is recorded)
            this->world->children[beamline_geometries.size() + 2] = phantom;
            phantom->geo =
                new grid3d<density_t, R>(-200, 200, 401, -200, 200, 401, -1, 1, 2, // Z-axis is only 2mm thick
                                         transformPhantom.rotation);
            phantom->geo->set_data(rho_mass);

            // 3. Back Phantom (water behind the scoring slice)
            this->world->children[beamline_geometries.size() + 3] = backPhantom;
            backPhantom->geo =
                new grid3d<density_t, R>(-200, 200, 401, -200, 200, 401, -380, -1,
                                         2, transformPhantom.rotation);
            backPhantom->geo->set_data(rho_mass_parent2);
        } else {
            // Original logic for full 3D phantom
            this->world->children[beamline_geometries.size()] = phantom;
            phantom->geo = new grid3d<density_t, R>(...); // Full 3D grid
            phantom->geo->set_data(rho_mass);
        }
        // REVISION END
        ```

##### **B. DICOM File Saving (`mqi_io.hpp`)**

*   **Background**: When saving the result, the code must recognize that the data is from a 2D slice and write the DICOM headers accordingly (e.g., setting the number of frames to 1).
*   **Affected File**: `moqui_rev/moqui/base/mqi_io.hpp`
*   **Impact**: The `save_to_dcm` function, around **line 580**, is affected. It needs to check if `two_cm_mode` is active.

    *   **Guidance**: The `save_reshaped_files` function in `mqi_tps_env.hpp` (around line 1118) passes the `this->twoCentimeterMode` flag to `save_to_dcm`. You must ensure this flag is correctly handled in `save_to_dcm` to produce a valid 2D DICOM file instead of a 3D one.

    *   **Code Example**:
        ```cpp
        // In moqui_rev/moqui/base/mqi_io.hpp, around line 580

        // REVISION START: Add conditional logic for 2D vs 3D DICOM headers
        if (two_cm_mode) {
            // For 2D mode, set NumberOfFrames to 1
            dataset->putAndInsertUint16(DCM_NumberOfFrames, 1);
            
            // ... set other 2D-specific tags like ImagePositionPatient for a single slice ...
            sprintf(buffer, "%f\%f\%f", geo.get_x_edges()[0], geo.get_y_edges()[0], geo.get_z_edges()[z_index]);
            put_string(DCM_ImagePositionPatient, buffer);

            // Set the pixel data for the single slice
            dataset->putAndInsertUint16Array(DCM_PixelData, pixel_data.data(), slice_size);

        } else {
            // Original logic for 3D mode
            dataset->putAndInsertUint16(DCM_NumberOfFrames, nxyz.z);

            // ... set 3D-specific tags ...
            put_string(DCM_GridFrameOffsetVector, grid_frame_offset_vector.c_str());

            // Set the full 3D pixel data
            dataset->putAndInsertUint16Array(DCM_PixelData, pixel_data.data(), length);
        }
        // REVISION END
        ```

#### **2. Summary of Impact**

*   The implementation of `TwoCentimeterMode` is self-contained and enabled by a single flag.
*   It directly impacts geometry creation (`mqi_tps_env.hpp`) by substituting the patient CT geometry with a virtual phantom.
*   It also impacts the final output stage (`mqi_io.hpp`) to ensure the saved DICOM file correctly represents a 2D dose slice.
*   **Caution**: When modifying either `setup_world` or `save_to_dcm`, you must be careful to preserve this dual-path logic. Changes to the standard 3D workflow should be tested to ensure they do not inadvertently break the 2D dose calculation path.