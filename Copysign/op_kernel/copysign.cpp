#include "kernel_operator.h"
using namespace AscendC;

constexpr int64_t BUFFER_NUM(1);

template <class T> class KernelCopysign {

  public:
    __aicore__ inline KernelCopysign() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR out,
                                int64_t former_length, int64_t tail_length,
                                int64_t former_num, int64_t tile_length) {
        int64_t coreID = GetBlockIdx();
        coreLength = (coreID < former_num) ? former_length : tail_length;

        tileLength = tile_length;
        tileNum = (coreLength / tileLength);
        lastTileLength = coreLength - tileLength * tileNum;

        if (lastTileLength > 0) tileNum++;
        else lastTileLength = tileLength;

        int64_t coreFormerLength =
            (coreID < former_num)
                ? (former_length * coreID)
                : (tail_length * coreID +
                   (former_length - tail_length) * former_num);

        // printf("#%d: %d %d %d\n", coreID, coreFormerLength, coreLength,
        //        tileLength);

        pipe.InitBuffer(inputQue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(otherQue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(outQue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(cmpBuf, tileLength * sizeof(uint8_t));

        inputGM.SetGlobalBuffer((__gm__ T *)input + coreFormerLength,
                                coreLength);
        otherGM.SetGlobalBuffer((__gm__ T *)other + coreFormerLength,
                                coreLength);
        outGM.SetGlobalBuffer((__gm__ T *)out + coreFormerLength, coreLength);
    }
    __aicore__ inline void Process() {
        int64_t loopCount = tileNum;
        for (int64_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

  private:
    __aicore__ inline void CopyIn(int64_t progress) {
        int64_t currentLength =
            (progress == tileNum - 1) ? lastTileLength : tileLength;

        LocalTensor<T> inputLocal = inputQue.AllocTensor<T>();
        LocalTensor<T> otherLocal = otherQue.AllocTensor<T>();

        DataCopy(inputLocal, inputGM[progress * tileLength], currentLength);
        DataCopy(otherLocal, otherGM[progress * tileLength], currentLength);

        inputQue.EnQue(inputLocal);
        otherQue.EnQue(otherLocal);
    }
    __aicore__ inline void Compute(int64_t progress) {
        int64_t currentLength =
            (progress == tileNum - 1) ? lastTileLength : tileLength;

        LocalTensor<T> inputLocal = inputQue.DeQue<T>();
        LocalTensor<T> otherLocal = otherQue.DeQue<T>();
        LocalTensor<T> outLocal = outQue.AllocTensor<T>();
        LocalTensor<uint8_t> cmp = cmpBuf.Get<uint8_t>();

        CompareScalar(cmp, otherLocal, T(0), CMPMODE::GE, currentLength);
        Abs(inputLocal, inputLocal, currentLength);
        Muls(otherLocal, inputLocal, T(-1), currentLength);
        Select(outLocal, cmp, inputLocal, otherLocal,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, currentLength);

        outQue.EnQue<T>(outLocal);
        inputQue.FreeTensor(inputLocal);
        otherQue.FreeTensor(otherLocal);
    }
    __aicore__ inline void CopyOut(int64_t progress) {
        int64_t currentLength =
            (progress == tileNum - 1) ? lastTileLength : tileLength;

        LocalTensor<T> outLocal = outQue.DeQue<T>();
        DataCopy(outGM[progress * tileLength], outLocal, currentLength);
        outQue.FreeTensor(outLocal);
    }

  private:
    TPipe pipe;
    GlobalTensor<T> inputGM;
    GlobalTensor<T> otherGM;
    GlobalTensor<T> outGM;
    TQue<QuePosition::VECIN, 1> inputQue, otherQue;
    TQue<QuePosition::VECOUT, 1> outQue;
    TBuf<TPosition::VECCALC> cmpBuf;

    int64_t coreLength;
    int64_t tileNum;
    int64_t tileLength;
    int64_t lastTileLength;
};

template <> class KernelCopysign<bfloat16_t> {
    using T = bfloat16_t;

  public:
    __aicore__ inline KernelCopysign() {}
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR other, GM_ADDR out,
                                int64_t former_length, int64_t tail_length,
                                int64_t former_num, int64_t tile_length) {
        int64_t coreID = GetBlockIdx();
        coreLength = (coreID < former_num) ? former_length : tail_length;

        tileLength = tile_length;
        tileNum = (coreLength / tileLength);
        lastTileLength = coreLength - tileLength * tileNum;

        if (lastTileLength > 0) tileNum++;
        else lastTileLength = tileLength;

        int64_t coreFormerLength =
            (coreID < former_num)
                ? (former_length * coreID)
                : (tail_length * coreID +
                   (former_length - tail_length) * former_num);

        pipe.InitBuffer(inputQue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(otherQue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(outQue, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(inputBuf, tileLength * sizeof(float));
        pipe.InitBuffer(otherBuf, tileLength * sizeof(float));
        pipe.InitBuffer(cmpBuf, tileLength * sizeof(uint8_t));

        inputGM.SetGlobalBuffer((__gm__ T *)input + coreFormerLength,
                                coreLength);
        otherGM.SetGlobalBuffer((__gm__ T *)other + coreFormerLength,
                                coreLength);
        outGM.SetGlobalBuffer((__gm__ T *)out + coreFormerLength, coreLength);
    }
    __aicore__ inline void Process() {
        int64_t loopCount = tileNum;
        for (int64_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

  private:
    __aicore__ inline void CopyIn(int64_t progress) {
        int64_t currentLength =
            (progress == tileNum - 1) ? lastTileLength : tileLength;

        LocalTensor<T> inputLocal = inputQue.AllocTensor<T>();
        LocalTensor<T> otherLocal = otherQue.AllocTensor<T>();

        DataCopy(inputLocal, inputGM[progress * tileLength], currentLength);
        DataCopy(otherLocal, otherGM[progress * tileLength], currentLength);

        inputQue.EnQue(inputLocal);
        otherQue.EnQue(otherLocal);
    }
    __aicore__ inline void Compute(int64_t progress) {
        int64_t currentLength =
            (progress == tileNum - 1) ? lastTileLength : tileLength;

        LocalTensor<T> inputLocal = inputQue.DeQue<T>();
        LocalTensor<T> otherLocal = otherQue.DeQue<T>();
        LocalTensor<T> outLocal = outQue.AllocTensor<T>();

        LocalTensor<float> inputFloat = inputBuf.Get<float>();
        LocalTensor<float> otherFloat = otherBuf.Get<float>();
        LocalTensor<uint8_t> cmp = cmpBuf.Get<uint8_t>();

        Cast(otherFloat, otherLocal, RoundMode::CAST_NONE, currentLength);
        CompareScalar(cmp, otherFloat, float(0), CMPMODE::GE, currentLength);

        Cast(inputFloat, inputLocal, RoundMode::CAST_NONE, currentLength);
        Abs(inputFloat, inputFloat, currentLength);
        Muls(otherFloat, inputFloat, float(-1), currentLength);

        Select(otherFloat, cmp, inputFloat, otherFloat,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, currentLength);
        Cast(outLocal, otherFloat, RoundMode::CAST_TRUNC, currentLength);

        outQue.EnQue<T>(outLocal);
        inputQue.FreeTensor(inputLocal);
        otherQue.FreeTensor(otherLocal);
    }
    __aicore__ inline void CopyOut(int64_t progress) {
        int64_t currentLength =
            (progress == tileNum - 1) ? lastTileLength : tileLength;

        LocalTensor<T> outLocal = outQue.DeQue<T>();
        DataCopy(outGM[progress * tileLength], outLocal, currentLength);
        outQue.FreeTensor(outLocal);
    }

  private:
    TPipe pipe;
    GlobalTensor<T> inputGM;
    GlobalTensor<T> otherGM;
    GlobalTensor<T> outGM;
    TQue<QuePosition::VECIN, 1> inputQue, otherQue;
    TQue<QuePosition::VECOUT, 1> outQue;
    TBuf<TPosition::VECCALC> inputBuf, otherBuf, cmpBuf;

    int64_t coreLength;
    int64_t tileNum;
    int64_t tileLength;
    int64_t lastTileLength;
};

template <class T> class KernelCopysignBoardCastVector {

  public:
    __aicore__ inline KernelCopysignBoardCastVector() {}
    __aicore__ inline void
    Init(GM_ADDR input, GM_ADDR other, GM_ADDR out, int64_t former_vector,
         int64_t tail_vector, int64_t former_num, int64_t tile_vector,
         int64_t vector_length, int64_t align_vector_length, uint8_t board_cast,
         int64_t out_dim1, int64_t out_dim2, int64_t out_dim3) {

        int64_t coreID = GetBlockIdx();
        coreVector = (coreID < former_num) ? former_vector : tail_vector;
        vectorLength = vector_length;
        alignVectorLength = align_vector_length;

        tileVector = tile_vector;
        tileNum = (coreVector / tileVector);
        lastTileVector = coreVector - tileVector * tileNum;

        if (lastTileVector > 0) tileNum++;
        else lastTileVector = tileVector;

        coreFormerVector = ((coreID < former_num)
                                ? (former_vector * coreID)
                                : (tail_vector * coreID +
                                   (former_vector - tail_vector) * former_num));

        // printf("#%d: %d %d %d\n", coreID, coreFormerVector, coreVector,
        //        tileVector);

        inputStride1 = ((board_cast >> 0) & 1) ^ 1;
        inputStride2 = ((board_cast >> 1) & 1) ^ 1;
        inputStride3 = ((board_cast >> 2) & 1) ^ 1;
        otherStride1 = ((board_cast >> 3) & 1) ^ 1;
        otherStride2 = ((board_cast >> 4) & 1) ^ 1;
        otherStride3 = ((board_cast >> 5) & 1) ^ 1;

        outDim1 = out_dim1;
        outDim2 = out_dim2;
        outDim3 = out_dim3;
        inputDim1 = inputStride1 ? outDim1 : 1;
        inputDim2 = inputStride2 ? outDim2 : 1;
        inputDim3 = inputStride3 ? outDim3 : 1;
        otherDim1 = otherStride1 ? outDim1 : 1;
        otherDim2 = otherStride2 ? outDim2 : 1;
        otherDim3 = otherStride3 ? outDim3 : 1;

        int64_t alignTileLength = tileVector * alignVectorLength;

        pipe.InitBuffer(inputQue, 1, alignTileLength * sizeof(T));
        pipe.InitBuffer(otherQue, 1, alignTileLength * sizeof(T));
        pipe.InitBuffer(outQue, 1, alignTileLength * sizeof(T));
        pipe.InitBuffer(cmpBuf, alignTileLength * sizeof(uint8_t));

        inputGM.SetGlobalBuffer((__gm__ T *)input, inputDim1 * inputDim2 *
                                                       inputDim3 *
                                                       vectorLength);
        otherGM.SetGlobalBuffer((__gm__ T *)other, otherDim1 * otherDim2 *
                                                       otherDim3 *
                                                       vectorLength);
        outGM.SetGlobalBuffer((__gm__ T *)out,
                              outDim1 * outDim2 * outDim3 * vectorLength);
    }
    __aicore__ inline void Process() {
        int64_t loopCount = tileNum;
        for (int64_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

  private:
    __aicore__ inline void CopyIn(int64_t progress) {
        int64_t beginVector = progress * tileVector + coreFormerVector;
        int64_t currentVector =
            (progress == tileNum - 1) ? lastTileVector : tileVector;

        LocalTensor<T> inputLocal = inputQue.AllocTensor<T>();
        LocalTensor<T> otherLocal = otherQue.AllocTensor<T>();

        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(vectorLength * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        for (int64_t i = 0; i < currentVector; i++) {
            int64_t dim1 = (i + beginVector) / (outDim2 * outDim3);
            int64_t dim2 = (i + beginVector) / outDim3 % outDim2;
            int64_t dim3 = (i + beginVector) % outDim3;

            int64_t inputID = dim1 * inputStride1 * inputDim2 * inputDim3 +
                              dim2 * inputStride2 * inputDim3 +
                              dim3 * inputStride3;
            int64_t otherID = dim1 * otherStride1 * otherDim2 * otherDim3 +
                              dim2 * otherStride2 * otherDim3 +
                              dim3 * otherStride3;

            DataCopyPad(inputLocal[i * alignVectorLength],
                        inputGM[inputID * vectorLength], copyParams, padParams);
            DataCopyPad(otherLocal[i * alignVectorLength],
                        otherGM[otherID * vectorLength], copyParams, padParams);
        }

        inputQue.EnQue(inputLocal);
        otherQue.EnQue(otherLocal);
    }
    __aicore__ inline void Compute(int64_t progress) {
        int64_t currentLength =
            alignVectorLength *
            ((progress == tileNum - 1) ? lastTileVector : tileVector);

        LocalTensor<T> inputLocal = inputQue.DeQue<T>();
        LocalTensor<T> otherLocal = otherQue.DeQue<T>();
        LocalTensor<T> outLocal = outQue.AllocTensor<T>();
        LocalTensor<uint8_t> cmp = cmpBuf.Get<uint8_t>();

        CompareScalar(cmp, otherLocal, T(0), CMPMODE::GE, currentLength);
        Abs(inputLocal, inputLocal, currentLength);
        Muls(otherLocal, inputLocal, T(-1), currentLength);
        Select(outLocal, cmp, inputLocal, otherLocal,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, currentLength);

        outQue.EnQue<T>(outLocal);
        inputQue.FreeTensor(inputLocal);
        otherQue.FreeTensor(otherLocal);
    }
    __aicore__ inline void CopyOut(int64_t progress) {
        int64_t beginVector = progress * tileVector + coreFormerVector;
        int64_t currentVector =
            (progress == tileNum - 1) ? lastTileVector : tileVector;

        LocalTensor<T> outLocal = outQue.DeQue<T>();

        int64_t contentSize = vectorLength * sizeof(T);
        int64_t alignSize = alignVectorLength * sizeof(T);
        int64_t srcStride = (alignSize - (contentSize + 31) / 32 * 32) / 32;

        DataCopyExtParams copyParams{
            static_cast<uint16_t>(currentVector),
            static_cast<uint32_t>(vectorLength * sizeof(T)),
            static_cast<uint32_t>(srcStride), 0, 0};
        DataCopyPad(outGM[beginVector * vectorLength], outLocal, copyParams);

        outQue.FreeTensor(outLocal);
    }

  private:
    TPipe pipe;
    GlobalTensor<T> inputGM;
    GlobalTensor<T> otherGM;
    GlobalTensor<T> outGM;
    TQue<QuePosition::VECIN, 1> inputQue, otherQue;
    TQue<QuePosition::VECOUT, 1> outQue;
    TBuf<TPosition::VECCALC> cmpBuf;

    int64_t coreVector;
    int64_t coreFormerVector;
    int64_t tileNum;
    int64_t tileVector;
    int64_t lastTileVector;

    int64_t vectorLength;
    int64_t alignVectorLength;

    int8_t inputStride1, inputStride2, inputStride3;
    int8_t otherStride1, otherStride2, otherStride3;
    int64_t outDim1, outDim2, outDim3;
    int64_t inputDim1, inputDim2, inputDim3;
    int64_t otherDim1, otherDim2, otherDim3;
};

template <> class KernelCopysignBoardCastVector<bfloat16_t> {
    using T = bfloat16_t;

  public:
    __aicore__ inline KernelCopysignBoardCastVector() {}
    __aicore__ inline void
    Init(GM_ADDR input, GM_ADDR other, GM_ADDR out, int64_t former_vector,
         int64_t tail_vector, int64_t former_num, int64_t tile_vector,
         int64_t vector_length, int64_t align_vector_length, uint8_t board_cast,
         int64_t out_dim1, int64_t out_dim2, int64_t out_dim3) {

        int64_t coreID = GetBlockIdx();
        coreVector = (coreID < former_num) ? former_vector : tail_vector;
        vectorLength = vector_length;
        alignVectorLength = align_vector_length;

        tileVector = tile_vector;
        tileNum = (coreVector / tileVector);
        lastTileVector = coreVector - tileVector * tileNum;

        if (lastTileVector > 0) tileNum++;
        else lastTileVector = tileVector;

        coreFormerVector = ((coreID < former_num)
                                ? (former_vector * coreID)
                                : (tail_vector * coreID +
                                   (former_vector - tail_vector) * former_num));

        // printf("#%d: %d %d %d\n", coreID, coreFormerVector, coreVector,
        //        tileVector);

        inputStride1 = ((board_cast >> 0) & 1) ^ 1;
        inputStride2 = ((board_cast >> 1) & 1) ^ 1;
        inputStride3 = ((board_cast >> 2) & 1) ^ 1;
        otherStride1 = ((board_cast >> 3) & 1) ^ 1;
        otherStride2 = ((board_cast >> 4) & 1) ^ 1;
        otherStride3 = ((board_cast >> 5) & 1) ^ 1;

        outDim1 = out_dim1;
        outDim2 = out_dim2;
        outDim3 = out_dim3;
        inputDim1 = inputStride1 ? outDim1 : 1;
        inputDim2 = inputStride2 ? outDim2 : 1;
        inputDim3 = inputStride3 ? outDim3 : 1;
        otherDim1 = otherStride1 ? outDim1 : 1;
        otherDim2 = otherStride2 ? outDim2 : 1;
        otherDim3 = otherStride3 ? outDim3 : 1;

        // printf("%d %d %d\n", inputStride1, inputStride2, inputStride3);
        // printf("%d %d %d\n", otherStride1, otherStride2, otherStride3);
        // printf("%lld %lld %lld\n", inputDim1, inputDim2, inputDim3);
        // printf("%lld %lld %lld\n", otherDim1, otherDim2, otherDim3);
        // printf("%lld %lld %lld\n", outDim1, outDim2, outDim3);

        int64_t alignTileLength = tileVector * alignVectorLength;

        pipe.InitBuffer(inputQue, 1, alignTileLength * sizeof(T));
        pipe.InitBuffer(otherQue, 1, alignTileLength * sizeof(T));
        pipe.InitBuffer(outQue, 1, alignTileLength * sizeof(T));
        pipe.InitBuffer(inputBuf, alignTileLength * sizeof(float));
        pipe.InitBuffer(otherBuf, alignTileLength * sizeof(float));
        pipe.InitBuffer(cmpBuf, alignTileLength * sizeof(uint8_t));

        inputGM.SetGlobalBuffer((__gm__ T *)input, inputDim1 * inputDim2 *
                                                       inputDim3 *
                                                       vectorLength);
        otherGM.SetGlobalBuffer((__gm__ T *)other, otherDim1 * otherDim2 *
                                                       otherDim3 *
                                                       vectorLength);
        outGM.SetGlobalBuffer((__gm__ T *)out,
                              outDim1 * outDim2 * outDim3 * vectorLength);
    }
    __aicore__ inline void Process() {
        int64_t loopCount = tileNum;
        for (int64_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

  private:
    __aicore__ inline void CopyIn(int64_t progress) {
        int64_t beginVector = progress * tileVector + coreFormerVector;
        int64_t currentVector =
            (progress == tileNum - 1) ? lastTileVector : tileVector;

        LocalTensor<T> inputLocal = inputQue.AllocTensor<T>();
        LocalTensor<T> otherLocal = otherQue.AllocTensor<T>();

        DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(vectorLength * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

        for (int64_t i = 0; i < currentVector; i++) {
            int64_t dim1 = (i + beginVector) / (outDim2 * outDim3);
            int64_t dim2 = (i + beginVector) / outDim3 % outDim2;
            int64_t dim3 = (i + beginVector) % outDim3;

            int64_t inputID = dim1 * inputStride1 * inputDim2 * inputDim3 +
                              dim2 * inputStride2 * inputDim3 +
                              dim3 * inputStride3;
            int64_t otherID = dim1 * otherStride1 * otherDim2 * otherDim3 +
                              dim2 * otherStride2 * otherDim3 +
                              dim3 * otherStride3;

            DataCopyPad(inputLocal[i * alignVectorLength],
                        inputGM[inputID * vectorLength], copyParams, padParams);
            DataCopyPad(otherLocal[i * alignVectorLength],
                        otherGM[otherID * vectorLength], copyParams, padParams);
        }

        inputQue.EnQue(inputLocal);
        otherQue.EnQue(otherLocal);
    }
    __aicore__ inline void Compute(int64_t progress) {
        int64_t currentLength =
            alignVectorLength *
            ((progress == tileNum - 1) ? lastTileVector : tileVector);

        LocalTensor<T> inputLocal = inputQue.DeQue<T>();
        LocalTensor<T> otherLocal = otherQue.DeQue<T>();
        LocalTensor<T> outLocal = outQue.AllocTensor<T>();

        LocalTensor<float> inputFloat = inputBuf.Get<float>();
        LocalTensor<float> otherFloat = otherBuf.Get<float>();
        LocalTensor<uint8_t> cmp = cmpBuf.Get<uint8_t>();

        Cast(otherFloat, otherLocal, RoundMode::CAST_NONE, currentLength);
        CompareScalar(cmp, otherFloat, float(0), CMPMODE::GE, currentLength);

        Cast(inputFloat, inputLocal, RoundMode::CAST_NONE, currentLength);
        Abs(inputFloat, inputFloat, currentLength);
        Muls(otherFloat, inputFloat, float(-1), currentLength);

        Select(otherFloat, cmp, inputFloat, otherFloat,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, currentLength);
        Cast(outLocal, otherFloat, RoundMode::CAST_TRUNC, currentLength);

        outQue.EnQue<T>(outLocal);
        inputQue.FreeTensor(inputLocal);
        otherQue.FreeTensor(otherLocal);
    }
    __aicore__ inline void CopyOut(int64_t progress) {
        int64_t beginVector = progress * tileVector + coreFormerVector;
        int64_t currentVector =
            (progress == tileNum - 1) ? lastTileVector : tileVector;

        LocalTensor<T> outLocal = outQue.DeQue<T>();

        int64_t contentSize = vectorLength * sizeof(T);
        int64_t alignSize = alignVectorLength * sizeof(T);
        int64_t srcStride = (alignSize - (contentSize + 31) / 32 * 32) / 32;

        DataCopyExtParams copyParams{
            static_cast<uint16_t>(currentVector),
            static_cast<uint32_t>(vectorLength * sizeof(T)),
            static_cast<uint32_t>(srcStride), 0, 0};
        DataCopyPad(outGM[beginVector * vectorLength], outLocal, copyParams);

        outQue.FreeTensor(outLocal);
    }

  private:
    TPipe pipe;
    GlobalTensor<T> inputGM;
    GlobalTensor<T> otherGM;
    GlobalTensor<T> outGM;
    TQue<QuePosition::VECIN, 1> inputQue, otherQue;
    TQue<QuePosition::VECOUT, 1> outQue;
    TBuf<TPosition::VECCALC> inputBuf, otherBuf, cmpBuf;

    int64_t coreVector;
    int64_t coreFormerVector;
    int64_t tileNum;
    int64_t tileVector;
    int64_t lastTileVector;

    int64_t vectorLength;
    int64_t alignVectorLength;

    int8_t inputStride1, inputStride2, inputStride3;
    int8_t otherStride1, otherStride2, otherStride3;
    int64_t outDim1, outDim2, outDim3;
    int64_t inputDim1, inputDim2, inputDim3;
    int64_t otherDim1, otherDim2, otherDim3;
};

extern "C" __global__ __aicore__ void copysign(GM_ADDR input, GM_ADDR other,
                                               GM_ADDR out, GM_ADDR workspace,
                                               GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    if (tiling_data.board_cast) {
        KernelCopysignBoardCastVector<DTYPE_INPUT> op;
        op.Init(input, other, out, tiling_data.former_length,
                tiling_data.tail_length, tiling_data.former_num,
                tiling_data.tile_length, tiling_data.vector_length,
                tiling_data.align_vector_length, tiling_data.board_cast,
                tiling_data.out_dim1, tiling_data.out_dim2,
                tiling_data.out_dim3);
        op.Process();
    } else {
        KernelCopysign<DTYPE_INPUT> op;
        op.Init(input, other, out, tiling_data.former_length,
                tiling_data.tail_length, tiling_data.former_num,
                tiling_data.tile_length);
        op.Process();
    }
}