# TMA (Tensor Memory Accelerator) Guide

## 개요

`large_tma.cu`는 NVIDIA GPU의 메모리 복사 방식을 **조건부로 선택**할 수 있는 커널입니다:
- **cp.async**: Ampere (sm_80+) - 비동기 메모리 복사
- **TMA**: Hopper (sm_90+) - 하드웨어 가속 텐서 메모리 복사

## TMA vs cp.async 비교

| 특성 | cp.async (sm_80+) | TMA (sm_90+) |
|------|-------------------|--------------|
| **지원 GPU** | Ampere (A100, RTX 30xx) | Hopper (H100, H200) |
| **대역폭** | 높음 | 매우 높음 |
| **레이턴시** | 낮음 | 매우 낮음 |
| **설정 복잡도** | 간단 | 복잡 (Tensor Descriptor) |
| **스레드 참여** | 모든 스레드 | 단일 스레드 (하드웨어 브로드캐스트) |

## 코드 구조

### 1. 템플릿 기반 조건부 컴파일

```cpp
template<bool UseTMA>
__global__ void mma_m16n8k16_unified(...) {
    if constexpr (UseTMA) {
        // TMA path (Hopper)
        load_tile_tma(...);
    } else {
        // cp.async path (Ampere)
        load_tile_async(...);
    }
}
```

### 2. 런타임 디스패치

```cpp
void launch_gemm(..., bool use_tma) {
    if (use_tma) {
        mma_m16n8k16_unified<true><<<...>>>(...);
    } else {
        mma_m16n8k16_unified<false><<<...>>>(...);
    }
}
```

### 3. GPU 아키텍처 감지

```cpp
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, device);
int sm_version = prop.major * 10 + prop.minor;

if (use_tma && sm_version < 90) {
    printf("WARNING: TMA requires sm_90+, falling back\n");
    use_tma = false;
}
```

## 컴파일 방법

### Ampere (A100, RTX 30xx) - cp.async만

```bash
# TMA 비활성화, sm_80 타겟
make large_tma

# 실행 (cp.async 사용)
./large_tma 0
```

### Hopper (H100) - TMA 지원

```bash
# TMA 활성화, sm_90 타겟
make large_tma_hopper

# 실행 (TMA 사용)
./large_tma_hopper 1
```

### 직접 컴파일

```bash
# Ampere용 (cp.async만)
nvcc -arch=sm_80 -DUSE_TMA=0 large_tma.cu -o large_tma_ampere

# Hopper용 (TMA 포함)
nvcc -arch=sm_90 -DUSE_TMA=1 large_tma.cu -o large_tma_hopper
```

## 실행 방법

프로그램은 명령줄 인자로 메모리 복사 방식을 선택합니다:

```bash
# 0 = cp.async (기본값)
./large_tma 0

# 1 = TMA (Hopper에서만 작동, 아니면 자동 폴백)
./large_tma 1
```

## TMA의 핵심 차이점

### cp.async (Ampere)
```cuda
// 모든 스레드가 참여
for (int chunk = tid; chunk < 32; chunk += blockDim.x) {
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :: "l"(smem_dst), "l"(gmem_src)
    );
}
```

### TMA (Hopper)
```cuda
// 단일 스레드만 명령 발행, 하드웨어가 브로드캐스트
if (tid == 0) {
    asm volatile(
        "cp.async.bulk.shared.global [%0], [%1], %2;\n"
        :: "l"(smem_dst), "l"(gmem_src), "n"(bytes)
    );
}
```

## 성능 특성

### cp.async 장점
- ✅ Ampere부터 사용 가능 (넓은 호환성)
- ✅ 구현이 직관적
- ✅ 세밀한 제어 가능

### TMA 장점
- ✅ **더 높은 대역폭** (하드웨어 최적화)
- ✅ **더 적은 레지스터 사용** (단일 스레드 발행)
- ✅ **2D/3D 타일 지원** (Tensor Descriptor)
- ✅ **L2 캐시 힌트** 자동 최적화

## 실제 TMA 사용 시 고려사항

현재 코드는 **단순화된 TMA 구현**입니다. 완전한 TMA 사용을 위해서는:

### 1. Tensor Map 생성 (Host)
```cpp
CUtensorMap tensor_map;
cuTensorMapEncodeTiled(
    &tensor_map,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
    2,                    // rank (2D)
    gmem_ptr,             // global address
    dims,                 // dimensions
    strides,              // strides
    tile_dims,            // tile dimensions
    ...
);
```

### 2. TMA 명령 사용 (Device)
```cuda
asm volatile(
    "cp.async.bulk.tensor.2d.shared.global.tile.L2::cache_hint "
    "[%0], [%1, {%2, %3}];\n"
    :: "l"(smem_ptr), "l"(tensor_map),
       "r"(coord_m), "r"(coord_n)
);
```

### 3. Cluster 그룹 활용
```cuda
// Hopper는 Thread Block Cluster를 지원
__cluster_dims__(2, 2, 1)
__global__ void kernel(...) {
    // Multi-block coordination
}
```

## 벤치마크 예상 결과

**A100 (Ampere)**:
- cp.async: ~1.5 TB/s 대역폭
- TMA: 사용 불가

**H100 (Hopper)**:
- cp.async: ~2.0 TB/s 대역폭
- TMA: ~3.0 TB/s 대역폭 (1.5x 개선)

## 제한사항

1. **TMA 구현 단순화**
   - 현재는 bulk copy로 구현
   - 완전한 버전은 Tensor Descriptor 필요

2. **정렬 요구사항**
   - TMA는 128바이트 정렬 필요
   - 코드에서 `__align__(128)` 사용

3. **아키텍처 의존성**
   - TMA는 Hopper 전용
   - Ampere에서는 자동으로 cp.async로 폴백

## 다음 단계

1. **Tensor Descriptor 통합**
   - `cudaTensorMapEncodeTiled` 사용
   - 2D 메모리 레이아웃 최적화

2. **Cluster 그룹 활용**
   - `__cluster_dims__` 사용
   - Multi-block 협업

3. **벤치마킹**
   - cp.async vs TMA 성능 비교
   - 다양한 행렬 크기 테스트

## 참고 자료

- [CUDA Programming Guide - TMA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-memory-access)
- [Hopper Architecture Whitepaper](https://resources.nvidia.com/en-us-tensor-core)
- [PTX ISA Reference - cp.async.bulk](https://docs.nvidia.com/cuda/parallel-thread-execution/)

