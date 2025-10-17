# 기능: Block (Aperture) 및 MLC 구현

`moqui_rev`는 빔의 모양을 종양 형태에 맞게 조절하는 핵심 장치인 Block(Aperture) 및 MLC(Multi-Leaf Collimator)를 시뮬레이션에 반영하기 위한 아키텍처를 개선했습니다. 그러나 분석 결과, 3차원 복셀(Voxel) 기반의 정교한 모델링은 아직 구현되지 않았으며, `moqui_v0`와 유사한 방식의 기능이 확장성 있는 구조로 개선된 것으로 확인됩니다.

## `moqui_v0`의 방식 (추정)

`moqui_v0`는 `mqi_aperture3d.hpp`에 정의된 `aperture3d` 클래스를 사용하여 조리개를 모델링합니다.

-   **가상 평면 모델**: `aperture3d`는 3차원 공간을 차지하는 실제적인 복셀 덩어리가 아니라, **가상의 2D 평면**처럼 동작합니다.
-   **통과/차단 판정**: 입자가 이 가상 평면에 도달하면, `is_inside` 함수가 2D 다각형으로 정의된 개구부(opening) 정보를 바탕으로 입자의 통과 여부만을 판정합니다.
    -   개구부 내부에 위치하면 통과(`APERTURE_OPEN`).
    -   개구부 외부에 위치하면 차단(`APERTURE_CLOSE`).
-   **한계**: 이 방식은 구현이 간단하지만, 입자가 조리개의 측면으로 입사하는 경우 등 복잡한 상호작용을 정확하게 모델링하기 어렵습니다.

```cpp
// moqui_v0/moqui/base/mqi_aperture3d.hpp

template<typename T, typename R>
class aperture3d : public grid3d<T, R> {
    // ...
    CUDA_HOST_DEVICE
    bool is_inside(mqi::vec3<R> pos) {
        // Point-in-polygon 알고리즘으로 2D 개구부 내부에 있는지 검사
    }

    CUDA_HOST_DEVICE
    virtual intersect_t<R> intersect(mqi::vec3<R>& p, mqi::vec3<R>& d, mqi::vec3<ijk_t> idx) {
        if (!is_inside(p)) {
            its_out.type = mqi::APERTURE_CLOSE; // 차단
            return its_out;
        } else {
            its_in.type = mqi::APERTURE_OPEN;   // 통과
            return its_in;
        }
    }
};
```

## `moqui_rev`의 변경된 아키텍처

`moqui_rev`의 `mqi_aperture.hpp`와 `mqi_aperture3d.hpp` 파일 자체는 `moqui_v0`와 동일합니다. 하지만 이를 사용하는 방식, 즉 아키텍처가 개선되었습니다.

### 1. 치료 기계 모델의 구체화

`moqui_rev`는 `mqi_treatment_machine_smc_gtr2.hpp`와 같이 **특정 치료 장비(예: 삼성서울병원 GTR2)를 클래스로 모델링**했습니다. 이는 각 장비의 고유한 사양에 맞는 빔라인 구성 요소(Aperture, MLC 등)를 생성할 수 있는 기반을 마련합니다.

### 2. DICOM 정보 기반의 객체 생성

-   `mqi_treatment_session.hpp`이 DICOM RTPLAN 파일을 읽어 `gtr2`와 같은 치료 기계 모델 객체를 생성합니다.
-   `gtr2` 객체는 `characterize_aperture`와 같은 멤버 함수를 통해, DICOM 파일의 `BeamLimitingDeviceSequence`에 포함된 정보를 파싱하여 `mqi::aperture` 객체를 생성합니다. 이 객체에는 2D 개구부 모양, 3D 위치, 두께 등의 정보가 포함됩니다.

```cpp
// moqui_rev/moqui/treatment_machines/mqi_treatment_machine_smc_gtr2.hpp

mqi::aperture*
characterize_aperture(const mqi::dataset* ds, mqi::modality_type m) {
    // DICOM에서 2D 개구부 좌표(xypts)를 읽음
    auto xypts = this->characterize_aperture_opening(ds, m);
    // DICOM에서 Block의 두께와 위치 정보를 읽음
    auto blk_ds = (*ds)(seq_tags->at("blk"));
    blk_ds[0]->get_values("BlockThickness", ftmp);
    // ...
    // 읽어온 정보로 mqi::aperture 객체를 생성하여 반환
    return new mqi::aperture(xypts, lxyz, pxyz, rxyz);
}
```

### 3. 현재 구현 상태 및 향후 확장성

`mqi_tps_env.hpp`의 `setup_world()` 함수를 분석한 결과, `create_voxelized_aperture` 기능이 `TODO`로 명시되어 있으며 현재 주석 처리되어 있습니다.

```cpp
// moqui_rev/moqui/base/environments/mqi_tps_env.hpp

} else if (beamline_geometries[i]->geotype == mqi::BLOCK) {
    /// TODO: defining voxelized aperture
    // beamline_objects[i+1] = this->create_voxelixed_aperture(...);
}
```

이는 `moqui_rev`가 아직 Aperture나 MLC를 **3차원 복셀 지오메트리로 모델링하는 기능은 갖추지 않았음**을 의미합니다. 현재는 `moqui_v0`와 마찬가지로 `aperture3d`의 가상 평면 모델을 사용하는 것으로 보입니다.

하지만 `moqui_rev`는 치료 기계별 모델링, DICOM 파싱 로직 분리 등 **향후 3D 복셀화된 조리개 및 MLC 기능을 쉽게 추가할 수 있는 유연하고 확장성 높은 소프트웨어 아키텍처를 구축**했다는 점에서 `moqui_v0`보다 크게 발전했습니다. 이 구조를 바탕으로 `create_voxelized_aperture` 함수를 구현하면, `characterize_aperture`에서 반환된 `mqi::aperture` 객체 정보를 바탕으로 실제 3D 복셀 그리드를 생성하고, 이를 시뮬레이션 월드에 추가할 수 있게 될 것입니다.

## 결론

`moqui_rev`의 Block/Aperture/MLC 구현은 기능적으로 `moqui_v0`와 유사한 수준에 머물러 있으나, 향후 정교한 3D 모델링을 도입하기 위한 **소프트웨어 아키텍처 측면의 중요한 기반을 마련**했다는 데 의의가 있습니다.