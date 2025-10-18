### Development Plan: 04 - DICOM RT Dose Output

This document provides a maintenance and improvement plan for the DICOM RT Dose output feature. This feature is complete and functional in `moqui_rev`. The focus is on ensuring its long-term stability and correctness.

#### **1. Maintenance and Improvement Goals**

The primary goal is to ensure the DICOM output remains compliant with medical imaging standards and is robust against variations in input data. A secondary goal is to establish automated testing to guarantee its functionality over time.

#### **2. Code Revision and Testing Guide**

The key logic is encapsulated within the `save_to_dcm` function. The plan involves adding more robust error handling and creating a dedicated verification test.

##### **A. Improving Robustness**

*   **File for Revision**: `moqui_rev/moqui/base/mqi_io.hpp`
*   **Line for Revision**: Approximately **line 510** (within `save_to_dcm`).

*   **Before Revision**: The function currently checks for the presence of mandatory tags from the source RTPLAN file and exits on failure.

    ```cpp
    // In moqui_rev/moqui/base/mqi_io.hpp, around line 510
    
    if (plan_dataset->findAndGetOFString(DCM_PatientName, patient_name).bad() ||
        plan_dataset->findAndGetOFString(DCM_PatientID, patient_id).bad() ||
        /* ... more checks ... */) {
      std::cerr << "Warning: Some patient/study information could not be read from RTPLAN" << std::endl;
    }
    ```

*   **After Revision (Proposed)**: The logic is enhanced to provide more specific warnings for missing optional-but-recommended tags, improving the user's ability to diagnose problems with their input DICOM files.

    ```cpp
    // In moqui_rev/moqui/base/mqi_io.hpp, around line 510

    // REVISION: Add more granular checks and warnings
    if (plan_dataset->findAndGetOFString(DCM_PatientName, patient_name).bad()) {
        std::cerr << "Warning: DCM_PatientName not found in source RTPLAN." << std::endl;
    }
    if (plan_dataset->findAndGetOFString(DCM_PatientID, patient_id).bad()) {
        std::cerr << "Warning: DCM_PatientID not found in source RTPLAN." << std::endl;
    }
    // Mandatory check - error out if missing
    if (plan_dataset->findAndGetOFString(DCM_StudyInstanceUID, study_instance_uid).bad()) {
        std::cerr << "Error: Mandatory tag DCM_StudyInstanceUID not found in source RTPLAN. Cannot create valid RTDOSE." << std::endl;
        return; // Exit function
    }
    // ... continue for other tags
    ```

##### **B. Adding Automated Verification (New Test File)**

*   **Goal**: Create a new test that runs a minimal simulation and verifies the content of the output DICOM file. This is a high-value addition to prevent future regressions.
*   **Proposed New File**: `moqui_rev/tests/test_dicom_output.cpp` (assuming a testing framework is set up).

*   **Code Example (Conceptual)**:

    ```cpp
    // In a new test file, e.g., moqui_rev/tests/test_dicom_output.cpp

    TEST(DicomOutput, GeneratesAndVerifiesDoseFile) {
        // 1. SETUP: Run a minimal simulation that produces a predictable dose cube.
        //    (This would involve setting up a minimal tps_env and running it).
        
        // 2. EXECUTE: Call the save_to_dcm function.
        mqi::io::save_to_dcm(...); // Call with predictable inputs

        // 3. VERIFY: Use DCMTK to read the file back.
        DcmFileFormat readFile;
        OFCondition status = readFile.loadFile("path/to/generated_dose.dcm");
        ASSERT_TRUE(status.good());

        DcmDataset* dataset = readFile.getDataset();
        OFString patientName;
        dataset->findAndGetOFString(DCM_PatientName, patientName);
        
        // Assert that key metadata is correct
        ASSERT_STREQ(patientName.c_str(), "TEST^PATIENT");

        double doseGridScaling;
        dataset->findAndGetFloat64(DCM_DoseGridScaling, doseGridScaling);
        ASSERT_NEAR(doseGridScaling, 0.001, 1e-6); // Check against expected scaling factor

        // ... Add more checks for pixel data, dimensions, etc.
    }
    ```

#### **3. Impact and Guidance**

*   **Primary Impact**: These changes improve the long-term quality and reliability of the software without altering the core simulation logic.
*   **Guidance**:
    *   **DCMTK Dependency**: The proposed test adds a development-time dependency on DCMTK. This is acceptable as the main code already requires it. Ensure the `CMakeLists.txt` for the testing suite is configured to link against DCMTK.
    *   **Caller Function**: The main caller of `save_to_dcm` is `save_reshaped_files` in `mqi_tps_env.hpp` (around line 1118). No changes are needed in the caller function, but it's important to be aware of this connection when testing.
    *   **Standard Compliance**: When writing tests, refer to the DICOM standard to ensure the tags being verified are correct for an RT Dose object. This prevents testing against incorrect assumptions.