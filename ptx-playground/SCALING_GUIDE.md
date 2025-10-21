# PTX Tensor Core GEMM - Scaling Guide

## 개요

`simple.cu`의 고정 크기 구현을 임의 크기로 확장하는 방법을 설명합니다.

## 핵심 변경사항

### 1. 원본 (simple.cu)
```
- 고정 크기: C[16x8] = A[16x16] * B[16x8]
- 단일 블록, 32 스레드 (1 warp)
- K 루프 없음 (K=16 고정)
```

### 2. 확장 버전 (large.cu)
```
- 임의 크기: C[M x N] = A[M x K] * B[K x N]
- 그리드: (M/16) x (N/8) 블록
- 각 블록: 32 스레드 (1 warp)
- K 차원 루프로 누적
```

## 확장 전략

### M, N 차원 확장 (공간 병렬화)
```cuda
// 블록 그리드로 출력 행렬 타일링
dim3 grid(M/16, N/8);   // 각 블록이 16x8 타일 담당

// 각 블록의 출력 위치
int tile_m = blockIdx.x * 16;
int tile_n = blockIdx.y * 8;
```

**핵심**: 하나의 mma.m16n8k16 명령이 16x8 출력을 생성하므로, 각 블록이 하나의 타일을 담당합니다.

### K 차원 확장 (반복 누적)
```cuda
unsigned c_reg[2] = {0u, 0u};  // 누적기 초기화

for (int k_base = 0; k_base < K; k_base += 16) {
    // 1. GMEM -> SMEM: A[tile_m:+16, k_base:+16], B[k_base:+16, tile_n:+8]
    // 2. SMEM -> REG: ldmatrix
    // 3. MMA: c_reg += a_reg * b_reg
}
```

**핵심**: mma 명령의 K 차원은 16으로 고정이므로, K 전체를 16씩 나누어 반복 계산합니다.

## 메모리 접근 패턴

### A 행렬 로딩 (각 K 반복마다)
```
Global: A[tile_m : tile_m+16, k_base : k_base+16]
- 연속된 16x16 타일
- 16행, 각 행 16개 원소 (32바이트)
- cp.async로 2개의 16바이트 청크로 복사
```

### B 행렬 로딩 (각 K 반복마다)
```
Global: B[k_base : k_base+16, tile_n : tile_n+8]
- Stride N인 16개 행에서 각각 8개 원소 추출
- 각 행마다 별도 cp.async 호출
```

## 성능 고려사항

### 장점
1. **Tensor Core 활용**: FP16 연산으로 높은 처리량
2. **cp.async**: 비동기 메모리 복사로 레이턴시 숨김
3. **레지스터 누적**: K 루프 동안 c_reg는 레지스터에 유지

### 제한사항
1. **크기 제약**: M은 16의 배수, N은 8의 배수, K는 16의 배수
2. **SMEM 사용량**: 블록당 768 바이트 (16x16 + 16x8)
3. **warp 동기화**: 전체 warp가 함께 작동해야 함

## 더 큰 확장 가능성

### 블록당 여러 타일 처리
현재는 1 블록 = 1 warp = 1 타일 (16x8)입니다.
더 큰 타일을 원한다면:

```cuda
// 예: 블록당 32x16 출력 (4개의 mma 명령)
// - 2 warps 사용
// - 또는 1 warp가 4번 mma 호출
```

### Double Buffering
K 루프의 성능 향상을 위해:
```cuda
__shared__ half As[2][16*16];  // Double buffer
__shared__ half Bs[2][16*8];

// 다음 타일 비동기 로드하면서 현재 타일 계산
```

### 더 큰 MMA 명령
```
mma.m16n8k16  -> 출력 16x8
mma.m16n8k8   -> K=8 (더 작은 K 단계)
mma.m8n8k4    -> TF32용
```

## 컴파일 & 실행

```bash
# 컴파일 (Ampere 이상 필요)
make large

# 실행
./large

# 또는
make run_large
```

## 검증

코드는 CPU 참조 구현과 비교하여 정확성을 검증합니다:
- 허용 오차: 1e-2 (FP16 정밀도)
- Identity 행렬 테스트로 쉬운 디버깅

## 다음 단계

1. **더 큰 행렬 테스트**: 256x256, 1024x1024 등
2. **벤치마크**: cuBLAS와 비교
3. **최적화**: Double buffering, 스레드 블록 튜닝
4. **FP32 출력**: mma의 FP16 입력, FP32 누적 버전

