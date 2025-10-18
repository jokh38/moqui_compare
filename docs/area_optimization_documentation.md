# 기능: 계산 영역 최적화 (Voxel-BVH 및 Dynamic LOD)

`moqui_rev`는 몬테카를로 시뮬레이션의 성능을 획기적으로 향상시키기 위해, **실제로 계산이 필요한 영역만 효율적으로 찾아내고, 상황에 따라 계산의 정밀도까지 동적으로 조절**하는 정교한 계산 영역 최적화 기능을 도입했습니다. 이는 `moqui_v0`가 전체 영역을 균일하게 계산하던 방식에서 크게 발전한 부분입니다.

이 기능은 **Voxel-BVH 하이브리드 시스템**과 **Dynamic LOD(Level of Detail) 시스템**이라는 두 가지 핵심 기술을 기반으로 합니다.

## 1. Voxel-BVH: 효율적인 공간 탐색 및 필터링

`moqui_v0`에서는 입자와 지오메트리의 교차점을 찾기 위해 모든 복셀을 순차적으로 검사했을 가능성이 높습니다. 이는 계산량이 매우 많은 비효율적인 방식입니다. `moqui_rev`는 이 문제를 해결하기 위해 `mqi_voxel_bvh.hpp`에 정의된 **Voxel-BVH 하이브리드 시스템**을 도입했습니다.

### 시스템 구성 요소

1.  **Voxel Mask (1차 필터)**:
    - 빔의 시점(Beam-Eye View)에서 지오메트리를 2D 평면에 투영하여 만드는 비트맵 마스크입니다.
    - 입자가 빔 경로에 들어올 때, 비용이 많이 드는 3D 교차점 계산을 수행하기 전에 이 2D 마스크를 통과하는지 여부를 먼저 매우 빠르게 검사합니다.
    - 만약 마스크를 통과하지 않는다면, 해당 입자는 더 이상 계산할 필요 없이 즉시 기각(cull)됩니다.

2.  **BVH (Bounding Volume Hierarchy, 2차 필터)**:
    - Voxel Mask를 통과한 입자에 대해서만 BVH를 이용한 3D 공간 탐색을 수행합니다.
    - BVH는 전체 3D 계산 공간을 계층적인 경계 상자(Bounding Box) 트리로 구성한 자료구조입니다.
    - 입자 궤적이 특정 경계 상자와 교차하지 않으면, 그 경계 상자에 포함된 모든 하위 공간은 계산에서 제외됩니다. 이를 통해 관련 없는 넓은 영역을 통째로 건너뛸 수 있어 탐색 속도가 매우 빠릅니다.

### 동작 방식 (Coarse-to-Fine)

이 하이브리드 시스템은 다음과 같은 **"Coarse-to-Fine" (거친 것에서 정밀한 것으로)** 필터링 전략을 사용합니다.

`입자 발생` -> `1. Voxel Mask 통과 여부 검사 (빠른 기각)` -> `2. BVH 트리 탐색 (효율적 공간 탐색)` -> `3. 실제 복셀과의 교차점 계산` -> `4. 물리 상호작용 계산`

이 다단계 필터링 구조는 계산이 불필요한 입자와 공간을 초기 단계에서 효율적으로 제거함으로써 전체 시뮬레이션 성능을 크게 향상시킵니다.

```cpp
// moqui_rev/moqui/base/mqi_voxel_bvh.hpp

// Voxel-BVH 하이브리드 시스템의 인터페이스
template <typename R>
class voxel_bvh_hybrid : public geometry_interface<R> {
    // ...
private:
  std::vector<bvh_node<R>> bvh_nodes_; // BVH 트리 노드
  voxel_mask<R> voxel_mask_;           // 2D Voxel Mask
    // ...
};
```

## 2. Dynamic LOD: 지능적인 계산 정밀도 조절

`moqui_rev`는 여기서 한 단계 더 나아가 `mqi_dynamic_lod.hpp`에 **Dynamic LOD (Level of Detail) 시스템**을 설계했습니다. 이는 단순히 계산할 영역을 찾는 것을 넘어, **"상황에 맞게 계산을 얼마나 정밀하게 할 것인가"**를 동적으로 결정하는 지능적인 제어 시스템입니다.

### 핵심 개념

-   **다중 상세 수준(Multiple LODs)**: `dynamic_lod_manager`는 여러 개의 지오메트리 표현을 관리합니다. 각 표현은 서로 다른 정밀도와 성능 비용을 가집니다.
    -   **COARSE**: 가장 낮은 정밀도, 가장 빠른 속도 (예: 단순화된 경계 상자)
    -   **MEDIUM**: 중간 정밀도, 균형 잡힌 속도 (예: 위에서 설명한 Voxel-BVH 시스템)
    -   **FINE**: 가장 높은 정밀도, 가장 느린 속도 (예: 모든 복셀을 정밀하게 계산)

-   **동적 LOD 선택**: 시뮬레이션 중 `dynamic_lod_manager`는 실시간으로 상황을 판단하여 최적의 LOD 레벨을 선택합니다.
    -   **거리 기반 선택**: 빔 소스와 멀리 떨어진 영역은 낮은 LOD(COARSE)로 계산하여 속도를 높입니다.
    -   **성능 기반 선택**: 목표 성능(예: 초당 처리 입자 수)을 설정하고, 현재 성능이 목표에 미치지 못하면 LOD를 동적으로 낮춰 속도를 확보하고, 성능에 여유가 있으면 LOD를 높여 정확도를 향상시킵니다.

```cpp
// moqui_rev/moqui/base/mqi_dynamic_lod.hpp

// LOD 시스템의 중심 컨트롤 타워
template <typename R>
class dynamic_lod_manager {
    // ...
public:
  // 현재 상황에 맞는 최적의 LOD 레벨을 반환
  geometry_complexity_t
  get_optimal_lod(const vec3<R> &view_position, const vec3<R> &object_position,
                  float current_performance_ms = 0.0f) const;

  // 현재 LOD 레벨에 맞는 지오메트리 객체를 반환
  geometry_interface<R> *get_current_geometry() const;
    // ...
};
```

## 결론

`moqui_rev`의 계산 영역 최적화 기능은 `moqui_v0`의 단일 레벨 계산 방식과 비교하여 혁신적인 발전을 이루었습니다. **Voxel-BVH**를 통해 불필요한 계산 영역을 효율적으로 제거하고, **Dynamic LOD** 시스템을 통해 계산의 정밀도까지 지능적으로 제어함으로써, 한정된 계산 자원을 가장 중요한 영역에 집중적으로 사용할 수 있게 되었습니다. 이는 시뮬레이션의 정확도를 유지하면서도 전체 실행 시간을 크게 단축시킬 수 있는 매우 강력한 최적화 전략입니다.