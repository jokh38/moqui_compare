# 기능: GPU 메모리 사용 방식 변경

`moqui_rev`는 `moqui_v0`에 비해 GPU 메모리를 더 효율적으로 사용하고 관리하기 위한 새로운 아키텍처를 도입했습니다. 이 변경은 코드의 **설계(Design)** 측면에서 큰 발전을 이루었으나, 아직 **구현(Implementation)**이 완전히 통합되지는 않은 점진적인 개발 단계에 있는 것으로 분석됩니다.

## `moqui_v0`의 메모리 관리 방식

`moqui_v0`의 GPU 데이터 업로드 로직(`mqi_upload_data.hpp`)은 다음과 같은 특징을 가집니다.

- **직접적이고 단순한 방식**: `cudaMalloc`과 `cudaMemcpy`를 사용하여 CPU의 데이터를 GPU의 전역 메모리(Global Memory)로 직접 복사합니다.
- **개별적인 메모리 할당**: 시뮬레이션에 필요한 지오메트리, scorer, 물질 정보 등 모든 객체와 그 구성 요소들을 개별적으로 GPU에 할당하고 복사합니다.
- **최적화 기법 부재**: 상수 메모리(Constant Memory), 텍스처 메모리(Texture Memory), 메모리 풀링(Memory Pooling)과 같은 고급 GPU 메모리 최적화 기법이 적용되어 있지 않습니다.

이 방식은 구현이 간단하지만, 빈번한 메모리 할당/해제로 인한 오버헤드, 전역 메모리 접근에 따른 속도 저하, 메모리 단편화 등의 잠재적인 성능 문제를 안고 있습니다.

## `moqui_rev`의 새로운 메모리 관리 아키텍처 (설계)

`moqui_rev`는 이러한 문제점을 해결하기 위해 `mqi_memory_optimization.hpp` 파일에 새로운 메모리 관리 아키텍처를 설계했습니다. 주요 개선 사항은 다음과 같습니다.

### 1. 상수 메모리(Constant Memory)의 적극적인 활용

- **개념**: GPU에는 작지만 매우 빠른 접근 속도를 제공하고 캐싱이 지원되는 상수 메모리 영역이 있습니다. 시뮬레이션 동안 변하지 않거나 모든 스레드가 동일하게 접근하는 데이터를 이곳에 저장하면 성능을 크게 향상시킬 수 있습니다.
- **설계**:
    - 물리 상수, 자주 사용하는 물질 정보, 에너지-저지능 테이블 등 작고 빈번하게 접근되는 데이터를 저장하기 위한 `__constant__` 변수들(`g_physics_constants`, `g_constant_materials` 등)을 선언했습니다.
    - `upload_..._to_constant_memory`와 같은 API를 설계하여 시뮬레이션 시작 시점에 이 데이터들을 상수 메모리에 한 번만 업로드하도록 했습니다.

```cpp
// moqui_rev/moqui/base/mqi_memory_optimization.hpp

#ifdef __CUDACC__
/// Constant memory for physics constants
extern __constant__ physics_constants_device_t g_physics_constants;
#endif

// API to upload data to constant memory
bool upload_physics_constants_to_constant_memory(
    const physics_constants_device_t *physics_consts);
```

### 2. 세분화된 메모리 풀링 (Segmented Memory Pooling)

- **개념**: 데이터의 접근 패턴(읽기 전용, 읽기/쓰기 등)에 따라 메모리 풀을 나누어 관리함으로써 캐시 효율성을 높이고 메모리 접근을 최적화합니다.
- **설계**:
    - `segmented_memory_pool_t` 구조체를 통해 데이터 특성에 따라 3가지 종류의 메모리 풀을 정의했습니다.
        - `physics_tables_pool`: 읽기 전용 데이터를 위한 풀
        - `particle_data_pool`: 읽기/쓰기가 빈번한 입자 데이터를 위한 풀
        - `statistics_pool`: 선량처럼 결과가 누적되는 데이터를 위한 풀
    - `allocate_from_..._pool` API를 통해 각 풀에서 필요한 메모리를 할당받도록 설계했습니다.

### 3. 메모리 성능 분석 및 진단 도구

- `moqui_v0`에는 없던 메모리 성능 분석 기능이 새롭게 설계되었습니다.
- `memory_performance_t` 구조체와 관련 API를 통해 메모리 대역폭, 캐시 적중률, 메모리 접근 패턴 등을 모니터링하여 성능 병목 지점을 쉽게 진단할 수 있도록 했습니다.

## 현재 구현 상태 및 결론

`mqi_upload_data.hpp` 파일 분석 결과, `moqui_rev`의 현재 데이터 업로드 로직은 여전히 `moqui_v0`와 동일한 방식을 사용하고 있으며, `mqi_memory_optimization.hpp`에 설계된 새로운 API들은 아직 실제 코드에 적용되지 않은 상태입니다.

결론적으로, `moqui_rev`는 GPU 메모리 사용 방식에 있어 **매우 진보적인 아키텍처를 설계**했지만, 이는 아직 **완전히 구현되지 않은 미래의 청사진**에 가깝습니다. 현재 코드는 `moqui_v0`와 유사하게 동작하지만, 이 새로운 설계가 완전히 구현되면 메모리 사용 효율과 시뮬레이션 속도 측면에서 상당한 성능 향상을 기대할 수 있습니다. 이는 `moqui` 프로젝트가 장기적인 성능 개선 계획을 가지고 점진적으로 발전하고 있음을 보여줍니다.