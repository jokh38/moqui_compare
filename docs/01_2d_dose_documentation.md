# 1. 2D 선량 계산 방식 변경 (2D Dose Calculation Method Change)

**요약: `moqui_rev`는 `TwoCentimeterMode`라는 새로운 기능을 통해 특정 깊이(2cm)에서의 2D 선량 계산을 최적화했습니다. 기존 `moqui_v0`가 전체 3D 계산 후 2D 데이터를 추출하는 비효율적인 방식을 사용한 것과 달리, `moqui_rev`는 시뮬레이션 시작 단계에서부터 계산 대상을 2D 슬라이스로 한정하는 가상의 팬텀을 동적으로 생성하여 계산 효율성을 크게 향상시켰습니다.**

---

# 기능: 2cm 깊이 2D 선량 파일 생성 방식 변경

`moqui_rev`에서는 특정 깊이(예: 2cm)에서의 2차원(2D) 선량 분포를 계산하고 저장하는 방식이 기존 `moqui_v0`에 비해 크게 개선되었습니다. 이는 주로 환자별 정도 관리(Patient-Specific QA) 목적으로 사용되는 기능으로, 계산 효율성과 정확성을 높이는 방향으로 변경되었습니다.

## `moqui_v0`의 방식 (추정)

`moqui_v0`에는 2D 선량 계산을 위한 명시적인 기능이 존재하지 않았습니다. 따라서 2cm 깊이의 2D 선량을 얻기 위해서는 다음과 같은 과정을 거쳤을 것으로 추정됩니다.

1.  전체 3D 팬텀에 대해 몬테카를로 시뮬레이션을 수행하여 3D 선량 분포를 계산합니다.
2.  계산이 완료된 후, 3D 선량 데이터(`*.raw` 또는 `*.mhd`)에서 2cm 깊이에 해당하는 2D 슬라이스 데이터를 후처리(post-processing) 스크립트를 통해 추출합니다.

이 방식은 전체 3D 볼륨에 대해 계산을 수행해야 하므로 비효율적이며, 후처리를 위한 별도의 과정이 필요하다는 단점이 있습니다.

## `moqui_rev`의 새로운 방식: 지오메트리 재구성

`moqui_rev`는 시뮬레이션 **설정 단계에서부터 계산 대상을 2D 슬라이스로 한정**하는 효율적인 방식을 도입했습니다. 이는 입력 파일(`moqui_tps.in`)의 `TwoCentimeterMode` 플래그를 통해 제어됩니다.

### 1. `TwoCentimeterMode` 활성화

사용자는 입력 파일에서 `TwoCentimeterMode`를 `true`로 설정하여 2D 선량 계산 모드를 활성화할 수 있습니다.

```ini
# moqui_rev/tps_env/moqui_tps.in

## While you use phantom geometry, program with two centimeter mode can get exact 2 cm position slice dose
TwoCentimeterMode true
```

### 2. 동적 가상 팬텀 생성

`TwoCentimeterMode`가 `true`이면, `mqi_tps_env.hpp`의 `setup_world()` 함수는 기존의 3D CT 지오메트리를 사용하는 대신, **2D 선량 계산에 최적화된 가상의 물 팬텀 지오메트리를 동적으로 생성**합니다.

이 가상 팬텀은 다음과 같이 3개의 부분으로 구성됩니다.

1.  **Front Phantom**: 빔 소스부터 2cm 깊이 직전까지의 물 팬텀.
2.  **Scoring Phantom**: 정확히 2cm 깊이에 위치하는 **매우 얇은(Z축으로 1-voxel 두께) 물 팬텀 슬라이스**. 이 부분이 실제 선량이 기록되는 영역입니다.
3.  **Back Phantom**: Scoring Phantom 뒤에 위치하는 나머지 물 팬텀.

```cpp
// moqui_rev/moqui/base/environments/mqi_tps_env.hpp (setup_world() 내부)

if (this->twoCentimeterMode) {
    // ...
    // Scoring phantom (선량 기록 대상이 되는 2cm 깊이 팬텀)
    this->world->children[beamline_geometries.size() + 2] = phantom;
    phantom->geo =
        new grid3d<density_t, R>(-200, 200, 401, -200, 200, 401, -1, 1, 2, // Z축으로 단일 복셀
                                 transformPhantom.rotation);
    // ...
}
```

### 3. 효율적인 선량 기록 (Scoring)

- 시뮬레이션 지오메트리 자체가 2D 슬라이스 형태이므로, `scorer`는 별도의 복잡한 필터링 로직 없이 이 `Scoring Phantom` 영역에 들어오는 입자에 대해서만 자연스럽게 선량을 기록합니다.
- `scorer`에 설정되는 ROI(`roi_t`)는 `mqi::DIRECT` 모드로, `Scoring Phantom`의 모든 복셀을 대상으로 합니다.

### 4. 2D DICOM 파일 저장

- 시뮬레이션이 완료된 후, `save_reshaped_files()` 함수는 `this->twoCentimeterMode` 값을 `save_to_dcm` 함수에 전달합니다.
- `save_to_dcm` 함수는 이 플래그 값을 확인하고, DICOM 헤더에 3D 볼륨 정보가 아닌 2D 단일 프레임(Single Frame) 정보를 기록하여 최종적으로 2D RT Dose 파일을 생성합니다.

```cpp
// moqui_rev/moqui/base/mqi_io.hpp (save_to_dcm() 내부)

if (two_cm_mode) {
    dataset->putAndInsertUint16(DCM_NumberOfFrames, 1); // 프레임 수를 1로 설정
    // ... 2D 슬라이스에 맞는 ImagePositionPatient, SliceThickness 등 설정 ...
} else {
    dataset->putAndInsertUint16(DCM_NumberOfFrames, nxyz.z); // 3D의 경우 전체 Z 슬라이스 수
    // ... 3D 볼륨에 맞는 정보 설정 ...
}
```

## 결론

`moqui_rev`는 `TwoCentimeterMode`라는 새로운 기능을 통해 2D 선량 계산 방식을 근본적으로 개선했습니다. 3D 계산 후 2D 데이터를 추출하는 `moqui_v0`의 방식과 달리, **시뮬레이션 단계에서부터 지오메트리를 2D 슬라이스로 재구성**하여 계산 자원을 효율적으로 사용하고 불필요한 계산을 제거했습니다. 이로 인해 QA용 2D 선량 분포를 더 빠르고 직접적으로 생성할 수 있게 되었습니다.
