/// Ping-pong CUDA kernel server for testing Dr.Jit queue functionality.
/// Receives floats, adds (float_index + 1 + msg_type) to each, sends back.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <drjit/array.h>
#include <drjit/jit.h>
#include <drjit-core/jit.h>
#include <cuda_runtime.h>

namespace nb = nanobind;
using namespace nb::literals;

using cu_array_u32_1d = nb::ndarray<uint32_t, nb::ndim<1>, nb::device::cuda>;

#define CHECK(x)                                                               \
    do {                                                                       \
        cudaError_t err = x;                                                   \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error: %s at %s:%d\n",                       \
                    cudaGetErrorString(err), __FILE__, __LINE__);              \
            throw std::runtime_error(cudaGetErrorString(err));                 \
        }                                                                      \
    } while (0)

/// Stride between queue counters to avoid false sharing (in bytes)
constexpr uint32_t CounterStrideBytes = 64;

/// Stride in uint32_t elements (for pointer arithmetic)
constexpr uint32_t CounterStride = CounterStrideBytes / sizeof(uint32_t);

/// Compile-time log2 for power-of-2 values
constexpr uint32_t log2i(uint32_t n) {
    return (n <= 1) ? 0 : 1 + log2i(n >> 1);
}

/// Maximum number of floats per message
constexpr uint32_t MaxFloatsPerMsg = 8;

/// Size of a message in bytes (same for input and output)
constexpr uint32_t MsgSizeBytes = MaxFloatsPerMsg * sizeof(float);

/// Number of messages per batch
constexpr uint32_t BatchSize = 256;

/// Number of batches in the queue
constexpr uint32_t BatchCount = 64;

/// Bytes per batch
constexpr uint32_t BytesPerBatch = MsgSizeBytes * BatchSize;

/// Load with relaxed memory ordering
__device__ uint32_t load_cg(volatile uint32_t *p) {
    uint32_t result;
    asm volatile ("ld.relaxed.gpu.global.u32 %0, [%1];" : "=r"(result) : "l"(p) : "memory");
    return result;
}

/// Store with release memory ordering
__device__ void store_release(volatile uint32_t *p, uint32_t value) {
    asm volatile ("st.release.gpu.global.b32 [%0], %1;" : : "l"(p), "r"(value) : "memory");
}

/// Ping-pong server kernel: reads floats, adds (float_idx + 1 + msg_type), writes back.
/// State encoding: msg_type * (BatchSize * 2) + msg_count
__global__ void pingpong_server(
    uint32_t *tail_ctrs,           // Tail counters
    uint32_t *queue_ctrs,          // Queue head counters
    uint32_t *queue_state,         // Per-batch state
    float *queue_body,             // Message payload
    uint32_t *phase,               // Phase counter for shutdown
    uint32_t launch_phase,         // Phase at launch time
    uint32_t num_floats,           // Number of floats per message
    uint32_t msg_types             // Number of message types
) {
    // Early exit for dummy launches
    if (!queue_ctrs)
        return;

    constexpr uint32_t BatchSizeBits = log2i(BatchSize);
    constexpr uint32_t CountBits = BatchSizeBits + 1;
    constexpr uint32_t MsgTypeShift = CountBits * 2;
    constexpr uint32_t CountMask = (1u << CountBits) - 1;

    __shared__ uint32_t shared_offset, shared_msg_count, shared_msg_type;

    bool leader = threadIdx.x == 0;

    while (true) {
        if (leader) {
            // Atomically claim a batch slot
            uint32_t tail = atomicAdd(tail_ctrs, 1u);
            shared_offset = tail % BatchCount;

            while (atomicCAS(tail_ctrs + (shared_offset + 1) * CounterStride, 0, 1)) {
                if (load_cg(phase) != launch_phase)
                    return;
            }

            // Monitor the queue state
            uint32_t *state_p = queue_state + shared_offset;
            uint32_t state = load_cg(state_p);

            while (true) {
                shared_msg_type = state >> MsgTypeShift;
                shared_msg_count = state & CountMask;

                if (shared_msg_type && shared_msg_count == BatchSize)
                    break;

                if (load_cg(phase) != launch_phase)
                    return;

                state = load_cg(state_p);
            }
        }

        __syncthreads();

        uint32_t offset = shared_offset;
        uint32_t msg_count = shared_msg_count;
        uint32_t msg_type = shared_msg_type;

        // Check for shutdown after sync
        if (msg_count == 0)
            return;

        // Process the batch: read floats, add (float_idx + 1 + msg_type_offset), write back
        float *batch = queue_body + offset * BatchSize * MaxFloatsPerMsg;

        // For virtual queues, msg_type encodes the message type (1-indexed)
        // For single msg_type, we don't add any type offset
        float type_offset = (msg_types > 1) ? (float)(msg_type - 1) : 0.0f;

        for (uint32_t i = threadIdx.x; i < BatchSize * num_floats; i += blockDim.x) {
            uint32_t msg_idx = i / num_floats;
            uint32_t float_idx = i % num_floats;

            float *msg = batch + msg_idx * MaxFloatsPerMsg;
            msg[float_idx] += (float)(float_idx + 1) + type_offset;
        }

        __syncthreads();

        // Signal that responses are ready
        if (leader) {
            // Reset slot for next use
            tail_ctrs[(offset + 1) * CounterStride] = 0;

            // Signal completion by writing msg_count (matches mlp_block.cu)
            __threadfence();
            store_release(queue_state + offset, msg_count);
        }
    }
}

__global__ void set_phase(uint32_t *phase, uint32_t value) {
    *phase = value;
}

/// CUDA stream wrapper with automatic cleanup.
struct Stream {
    cudaStream_t stream = nullptr;

    Stream() { CHECK(cudaStreamCreate(&stream)); }
    ~Stream() {
        if (stream)
            cudaStreamDestroy(stream);
    }

    operator cudaStream_t() const { return stream; }
};

struct PingPongQueue {
    PingPongQueue(uint32_t num_floats, uint32_t msg_types = 1, uint32_t blocks = 1)
        : m_num_floats(num_floats), m_msg_types(msg_types), m_blocks(blocks) {

        if (num_floats > MaxFloatsPerMsg)
            throw std::runtime_error("num_floats exceeds MaxFloatsPerMsg");

        // Virtual queue is used when msg_types > 1
        bool virtual_queue = msg_types > 1;
        uint32_t queue_count = virtual_queue ? (msg_types + 1) : 1;

        m_ctr_size = queue_count * CounterStrideBytes;
        m_state_size = queue_count * BatchCount * sizeof(uint32_t);
        m_body_size = BatchCount * BytesPerBatch;
        m_tail_size = (BatchCount + 1) * CounterStrideBytes;

        // Allocate GPU memory
        CHECK(cudaMalloc(&m_buffer, m_ctr_size + m_state_size + m_body_size));
        CHECK(cudaMalloc(&m_tail_buf, m_tail_size));
        CHECK(cudaMalloc(&m_phase, sizeof(uint32_t)));

        // Initialize phase
        set_phase<<<1, 1>>>(m_phase, ++m_phase_ctr);
        CHECK(cudaPeekAtLastError());
    }

    ~PingPongQueue() {
        cudaFree(m_buffer);
        cudaFree(m_tail_buf);
        cudaFree(m_phase);
    }

    void reset_queue(cudaStream_t s) {
        CHECK(cudaMemsetAsync(m_buffer, 0, m_ctr_size + m_state_size, s));
        CHECK(cudaMemsetAsync(m_tail_buf, 0, m_tail_size, s));
    }

    void set_state(int state, void *stream) {
        if (state == m_state)
            return;

        if (state) {
            // Reset queue and start server
            reset_queue(m_stream);

            pingpong_server<<<m_blocks, 256, 0, m_stream>>>(
                m_tail_buf,
                (uint32_t *)m_buffer,
                (uint32_t *)(m_buffer + m_ctr_size),
                (float *)(m_buffer + m_ctr_size + m_state_size),
                m_phase,
                m_phase_ctr,
                m_num_floats,
                m_msg_types
            );
        } else {
            // Signal shutdown
            set_phase<<<1, 1, 0, (cudaStream_t)stream>>>(m_phase, ++m_phase_ctr);
        }

        CHECK(cudaPeekAtLastError());
        m_state = state;
    }

    static nb::capsule callback(nb::object queue_obj) {
        PingPongQueue *queue = nb::cast<PingPongQueue *>(queue_obj);

        QueueCallback *cb = new QueueCallback{
            // callback
            [](QueueCallback *cb, void *stream, int state) {
                ((PingPongQueue *)cb->payload_1)->set_state(state, stream);
            },

            // inc_ref
            [](QueueCallback *cb) { cb->ref_count++; },

            // dec_ref
            [](QueueCallback *cb) {
                if (--cb->ref_count == 0) {
                    nb::gil_scoped_acquire guard;
                    Py_DECREF((PyObject *)cb->payload_2);
                    delete cb;
                }
            },

            // payload_1: raw pointer to queue
            queue,

            // payload_2: prevent Python GC
            queue_obj.inc_ref().ptr(),

            // ref_count
            1
        };

        auto capsule_deleter = [](void *p) noexcept {
            QueueCallback *cb = (QueueCallback *)p;
            cb->dec_ref(cb);
        };

        return nb::capsule(cb, "queue_callback", capsule_deleter);
    }

    /// Get the buffer as a CUDA ndarray (keeps queue alive via handle)
    static cu_array_u32_1d get_buffer(nb::pointer_and_handle<PingPongQueue> ph) {
        size_t size = (ph.p->m_ctr_size + ph.p->m_state_size + ph.p->m_body_size) /
                      sizeof(uint32_t);
        return cu_array_u32_1d(
            (uint32_t *)ph.p->m_buffer,
            { size },
            ph.h,  // prevent GC of queue while buffer is in use
            {},
            nb::dtype<uint32_t>(),
            nb::device::cuda::value
        );
    }

    uint32_t msg_types() const { return m_msg_types; }
    uint32_t msg_max_size() const { return MsgSizeBytes; }
    uint32_t batch_size() const { return BatchSize; }
    uint32_t batches() const { return BatchCount; }
    uint32_t num_floats() const { return m_num_floats; }
    uintptr_t buffer_ptr() const { return (uintptr_t)m_buffer; }
    size_t buffer_size() const { return m_ctr_size + m_state_size + m_body_size; }

    nb::str repr() const {
        return nb::str(
            "PingPongQueue[\n"
            "  num_floats={},\n"
            "  msg_types={},\n"
            "  batch_size={},\n"
            "  batches={},\n"
            "  msg_max_size={} B\n"
            "]"
        ).format(m_num_floats, m_msg_types, BatchSize, BatchCount, MsgSizeBytes);
    }

private:
    uint8_t *m_buffer = nullptr;
    uint32_t *m_tail_buf = nullptr;
    uint32_t *m_phase = nullptr;
    uint32_t m_num_floats = 0;
    uint32_t m_msg_types = 1;
    uint32_t m_blocks = 1;
    uint32_t m_phase_ctr = 0;
    int m_state = 0;
    size_t m_ctr_size = 0;
    size_t m_state_size = 0;
    size_t m_body_size = 0;
    size_t m_tail_size = 0;
    Stream m_stream;
};

NB_MODULE(queue_ext, m) {
    m.attr("MaxFloatsPerMsg") = MaxFloatsPerMsg;
    m.attr("BatchSize") = BatchSize;
    m.attr("BatchCount") = BatchCount;
    m.attr("MsgSizeBytes") = MsgSizeBytes;

    nb::class_<PingPongQueue>(m, "PingPongQueue")
        .def(nb::init<uint32_t, uint32_t, uint32_t>(),
             "num_floats"_a, "msg_types"_a = 1, "blocks"_a = 1)
        .def("set_state", &PingPongQueue::set_state,
             "state"_a, "stream"_a = nb::none())
        .def_prop_ro("callback",
            [](nb::object self) { return PingPongQueue::callback(self); })
        .def("get_buffer", &PingPongQueue::get_buffer)
        .def_prop_ro("buffer_size", &PingPongQueue::buffer_size)
        .def_prop_ro("msg_types", &PingPongQueue::msg_types)
        .def_prop_ro("msg_max_size", &PingPongQueue::msg_max_size)
        .def_prop_ro("batch_size", &PingPongQueue::batch_size)
        .def_prop_ro("batches", &PingPongQueue::batches)
        .def_prop_ro("num_floats", &PingPongQueue::num_floats)
        .def_prop_ro("buffer_ptr", &PingPongQueue::buffer_ptr)
        .def("__repr__", &PingPongQueue::repr);
}
