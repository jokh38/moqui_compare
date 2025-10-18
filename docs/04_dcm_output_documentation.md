# 4. DCM 파일 타입으로 결과 저장 (Save Results as DCM File Type)

**요약: `moqui_rev`는 시뮬레이션 결과를 의료 영상 표준인 DICOM RT Dose 형식으로 저장하는 기능을 추가했습니다. 이를 위해 오픈 소스 라이브러리인 DCMTK를 사용하며, 원본 치료 계획과의 메타데이터 연동, 정확한 선량 스케일링 등 표준을 준수하여 다른 의료 영상 소프트웨어와의 높은 호환성을 보장합니다.**

---

# 기능: DCM 파일 타입으로 결과 저장

`moqui_rev`에서는 몬테카를로 시뮬레이션으로 계산된 3차원 선량 분포 결과를 DICOM RT Dose 형식으로 저장하는 기능이 추가되었습니다. 이 기능은 의료 영상 표준인 DICOM을 준수하여, 계산된 선량 결과를 다양한 치료 계획 시스템(TPS) 및 DICOM 뷰어에서 직접 열어보고 분석할 수 있도록 합니다.

## 핵심 라이브러리: DCMTK

이 기능은 오픈 소스 DICOM 라이브러리인 **DCMTK (Digital Imaging and Communications in Medicine Toolkit)**를 기반으로 구현되었습니다.

- **의존성 관리**: `CMakeLists.txt`에서 `DCMTK` 라이브러리를 찾아보고, 라이브러리가 시스템에 설치되어 있을 경우에만 DICOM 출력 기능이 활성화됩니다. (`#if DCMTK_FOUND` 전처리기 지시문 사용)
- **대체 기능 (Fallback)**: 만약 `DCMTK`가 설치되어 있지 않다면, 프로그램은 오류 메시지를 출력하고 대신 결과를 `.mhd` (MetaImage) 형식으로 저장합니다. 이를 통해 프로그램의 안정적인 실행을 보장합니다.

## 구현 분석

DICOM RT Dose 파일 생성의 핵심 로직은 `moqui/base/mqi_io.hpp` 파일에 정의된 `save_to_dcm` 함수에 구현되어 있습니다.

```cpp
// moqui_rev/moqui/base/mqi_io.hpp

template <typename R>
void mqi::io::save_to_dcm(const mqi::node_t<R> *child, const double *src,
                          const R scale, const std::string &filepath,
                          const std::string &filename, const uint32_t length,
                          const mqi::dicom_t &dcm_info, bool two_cm_mode) {
#if DCMTK_FOUND
  // ... DCMTK를 사용한 DICOM 파일 생성 로직 ...
#else
  // ... DCMTK가 없을 경우의 대체 로직 (MHD로 저장) ...
#endif
}
```

### `save_to_dcm` 함수의 주요 동작 과정

1.  **원본 RTPLAN 파일 읽기**:
    - 함수의 인자로 받은 `dcm_info.plan_name` (치료 계획 파일 경로)을 `DCMTK`를 이용해 읽습니다.
    - 원본 파일에서 환자 정보(이름, ID), 연구 정보(Study UID), 좌표계 정보(Frame of Reference UID) 등 필수적인 메타데이터를 추출하여 새로 생성할 RTDOSE 파일에 복사합니다. 이는 생성된 선량 파일이 원본 환자 데이터와 정확히 정합되도록 보장합니다.

    ```cpp
    // moqui_rev/moqui/base/mqi_io.hpp

    DcmFileFormat plan_ff;
    OFCondition status = plan_ff.loadFile(dcm_info.plan_name.c_str());
    // ...
    DcmDataset *plan_dataset = plan_ff.getDataset();
    // ...
    plan_dataset->findAndGetOFString(DCM_PatientName, patient_name);
    // ...
    put_string(DCM_PatientName, patient_name.c_str());
    ```

2.  **새로운 DICOM 객체 생성 및 UID 할당**:
    - `DcmFileFormat` 객체를 생성하고, `SOPClassUID`를 `UID_RTDoseStorage`로 설정하여 이 파일이 RTDOSE 객체임을 명시합니다.
    - `dcmGenerateUniqueIdentifier` 함수를 호출하여, 전 세계적으로 고유한 식별자인 **SOP Instance UID**와 **Series Instance UID**를 생성하고 파일에 기록합니다.

    ```cpp
    // moqui_rev/moqui/base/mqi_io.hpp

    put_string(DCM_SOPClassUID, UID_RTDoseStorage);
    char new_uid[100];
    dcmGenerateUniqueIdentifier(new_uid, SITE_INSTANCE_UID_ROOT);
    put_string(DCM_SOPInstanceUID, new_uid);
    ```

3.  **선량 데이터 처리 및 스케일링**:
    - **DoseGridScaling**: DICOM RT Dose는 선량 값을 부동소수점 형태가 아닌 정수 형태로 저장합니다. 따라서 계산된 선량(float 또는 double)을 16비트 정수(0-65535)로 변환해야 합니다. `save_to_dcm` 함수는 먼저 전체 선량 데이터에서 최댓값을 찾습니다.
    - 이 최댓값을 65535로 나눈 값을 `DoseGridScaling` 태그(0028,1054)에 저장합니다. DICOM 뷰어는 이 스케일링 팩터를 이용해 저장된 정수 픽셀 값을 다시 실제 선량 값(Gy)으로 복원합니다.
      `실제 선량 값 = 픽셀 값 * DoseGridScaling`

    ```cpp
    // moqui_rev/moqui/base/mqi_io.hpp

    double max_dose = 0.0;
    for (uint32_t i = 0; i < length; ++i) {
        if (src[i] > max_dose) {
            max_dose = src[i];
        }
    }
    dose_grid_scaling = (max_dose * scale) / 65535.0;
    dataset->putAndInsertFloat64(DCM_DoseGridScaling, dose_grid_scaling);
    ```

4.  **기하학 정보 및 픽셀 데이터 저장**:
    - 선량 그리드의 크기(`Rows`, `Columns`), 각 복셀의 x, y 간격(`PixelSpacing`), 첫 복셀의 3차원 좌표(`ImagePositionPatient`) 등 기하학적 정보를 설정합니다.
    - 계산된 선량 배열(`src`)의 각 값을 `DoseGridScaling`으로 나누어 16비트 정수로 변환하고, 이 정수 배열을 `PixelData` 태그(7FE0,0010)에 저장합니다.

    ```cpp
    // moqui_rev/moqui/base/mqi_io.hpp

    std::vector<uint16_t> pixel_data(length);
    for (uint32_t i = 0; i < length; ++i) {
        pixel_data[i] = static_cast<uint16_t>((src[i] * scale) / dose_grid_scaling);
    }
    dataset->putAndInsertUint16Array(DCM_PixelData, pixel_data.data(), length);
    ```

5.  **파일 저장**:
    - 모든 태그 설정이 완료되면, `DcmFileFormat::saveFile` 함수를 호출하여 최종적으로 `.dcm` 파일을 디스크에 씁니다.

## 결론

`moqui_rev`의 DICOM RT Dose 출력 기능은 `DCMTK` 라이브러리를 효과적으로 활용하여 구현되었습니다. 원본 치료 계획 파일과의 메타데이터 연동, 고유 UID 생성, `DoseGridScaling`을 이용한 정확한 선량 값 표현 등 DICOM 표준을 충실히 따르고 있어, 생성된 파일을 다양한 상용 및 오픈 소스 의료 영상 소프트웨어와 높은 호환성을 가집니다. 이는 `moqui_rev`의 시뮬레이션 결과를 임상 워크플로우에 통합하는 데 있어 매우 중요한 개선점입니다.
