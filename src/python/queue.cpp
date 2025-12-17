#include "queue.h"
#include "apply.h"
#include "base.h"
#include "traits.h"

struct Queue {
    /**
     * \brief Construct a message queue for GPU-based inter-kernel communication.
     *
     * \param buffer
     *     A CUDA 1D UInt32 array serving as the underlying storage for the queue.
     *
     * \param msg_types
     *     The number of distinct message types supported by this queue.
     *
     * \param msg_max_size
     *     Maximum size of a single message payload in bytes.
     *
     * \param batch_size
     *     Number of messages per batch.
     *
     * \param batches
     *     Number of batches in the queue.
     *
     * \param callback
     *     A capsule containing a QueueCallback for handling queue events.
     *
     * \param debug
     *     When set to true, enables additional runtime checks and diagnostics.
     */
    Queue(nb::handle_t<ArrayBase> buffer, uint32_t msg_types,
          uint32_t msg_max_size, uint32_t batch_size, uint32_t batches,
          nb::capsule callback, bool debug)
        : m_msg_types(msg_types), m_msg_max_size(msg_max_size),
          m_batch_size(batch_size), m_batches(batches), m_debug(debug) {
        nb::handle tp = buffer.type();
        const ArraySupplement &s = supp(tp);

        if ((JitBackend) s.backend != JitBackend::CUDA || s.ndim != 1 ||
            (VarType) s.type != VarType::UInt32)
            nb::raise_type_error("drjit.Queue(): 'buffer' must be a CUDA 1D UInt32 array!");

        m_buffer = s.index(inst_ptr(buffer));

        m_callback = (QueueCallback *) callback.data("queue_callback");
        m_callback->inc_ref(m_callback);

        jit_var_inc_ref(m_buffer);
    }

    struct Message {
        uint32_t ticket;
        mutable nb::object result;

        Message(uint32_t ticket) : ticket(ticket) { }
        Message() = delete;
        Message(const Message &) = delete;
        Message(Message &&f)
            : ticket(f.ticket), result(std::move(f.result)) {
            f.ticket = 0;
        }

        ~Message() {
            jit_var_dec_ref(ticket);
        }

        /**
         * \brief Retrieve the response data associated with this message.
         *
         * This method unpacks the response payload into Dr.Jit arrays matching
         * the types specified in \c args. The result is cached, so subsequent
         * calls return the same tuple without re-processing.
         *
         * \param args
         *     A PyTree of Dr.Jit array types specifying the expected response
         *     layout. Each type must be a CUDA-backed array type.
         *
         * \return
         *     A tuple of Dr.Jit arrays containing the response data.
         */
        nb::tuple response(nb::args args) const {
            if (!result) {
                struct RecvTraverseCallback : TraverseCallback {
                    dr::vector<VarType> recv_vt;
                    RecvTraverseCallback() { recv_vt.reserve(8); }

                    void operator()(nb::handle) override {
                        nb::raise_type_error("drjit.Queue.Message.recv(): expected a type PyTree!");
                    }

                    void traverse_unknown(nb::handle h) override {
                        if (!h.is_type() || !is_drjit_type(h))
                            nb::raise_type_error("drjit.Queue.Message.recv(): expected a type PyTree!");

                        const ArraySupplement &s = supp(h);
                        if ((JitBackend) s.backend != JitBackend::CUDA || s.ndim == 0)
                            nb::raise_type_error("drjit.Queue.Message.recv(): "
                                                 "Invalid type \"%s\"!", nb::type_name(h).c_str());

                        if (s.ndim > 1) {
                            for (uint32_t i = 0; i < s.shape[0]; ++i)
                                traverse_unknown(value_t(h));
                            return;
                        }

                        recv_vt.push_back((VarType) s.type);
                    }
                };
                RecvTraverseCallback rtc;
                traverse("drjit.Queue.Message.recv", rtc, args);

                drjit::detail::index32_vector recv_idx(rtc.recv_vt.size(), 0);
                jit_queue_recv(ticket, (uint32_t) rtc.recv_vt.size(), rtc.recv_vt.data(), recv_idx.data());

                struct RecvTransformCallback : TransformCallback {
                    const drjit::detail::index32_vector &indices;
                    mutable uint32_t index_ctr;
                    RecvTransformCallback(
                        const drjit::detail::index32_vector &indices)
                        : indices(indices), index_ctr(0) {
                    }

                    void operator()(nb::handle, nb::handle) override {
                        nb::raise_type_error("drjit.Queue.Message.recv(): internal error!");
                    }

                    nb::object transform_unknown(nb::handle h) const override {
                        if (!h.is_type() || !is_drjit_type(h))
                            nb::raise_type_error("drjit.Queue.Message.recv(): expected a type PyTree!");

                        const ArraySupplement &s = supp(h);
                        if ((JitBackend) s.backend != JitBackend::CUDA || s.ndim == 0)
                            nb::raise_type_error("drjit.Queue.Message.recv(): "
                                                 "Invalid type \"%s\"!", nb::type_name(h).c_str());

                        nb::object o = inst_alloc(h);

                        if (s.ndim > 1) {
                            nb::inst_zero(o);
                            for (uint32_t i = 0; i < s.shape[0]; ++i)
                                o[i] = transform_unknown(value_t(h));
                        } else {
                            if (index_ctr >= indices.size())
                                nb::raise("drjit.Queue.Message.recv(): ran out of indices!");

                            s.init_index(indices[index_ctr++], inst_ptr(o));
                            nb::inst_mark_ready(o);
                        }

                        return o;
                    }
                };

                RecvTransformCallback rtrc(recv_idx);

                result = transform("drjit.Queue.Message.recv", rtrc, args);
            }

            return nb::borrow<nb::tuple>(result);
        }
    };

    /**
     * \brief Send a message to the queue.
     *
     * This method enqueues a message with the given type identifier a PyTree of
     * Dr.Jit arrays that make up the message body.
     *
     * \param msg_id
     *     A CUDA 1D UInt32 array specifying the message type
     *
     * \param args
     *     A PyTree of Dr.Jit arrays containing the message payload.
     *
     * \return
     *     A \ref Message object that can be used to retrieve the response.
     */
    Message send(nb::handle_t<dr::ArrayBase> msg_id, nb::args args) const {
        struct SendTraverseCallback : TraverseCallback {
            drjit::detail::index32_vector send_idx;
            SendTraverseCallback() { send_idx.reserve(8); }

            void operator()(nb::handle h) override {
                const ArraySupplement &s = supp(h.type());
                if (!s.index)
                    return;
                send_idx.push_back_borrow((uint32_t) s.index(inst_ptr(h)));
            }
        };
        SendTraverseCallback stc;
        traverse("drjit.Queue.send", stc, args);

        const ArraySupplement &s = supp(msg_id.type());
        if ((JitBackend) s.backend != JitBackend::CUDA || s.ndim != 1 ||
            (VarType) s.type != VarType::UInt32)
            nb::raise_type_error("drjit.Queue.send(): 'msg_id' must be a CUDA 1D UInt32 array!");

        return Message(jit_queue_send(
            m_buffer, m_msg_types, m_msg_max_size, m_batch_size, m_batches,
            m_debug, s.index(inst_ptr(msg_id)), (uint32_t) stc.send_idx.size(),
            stc.send_idx.data(), m_callback));
    }

    ~Queue() {
        jit_var_dec_ref(m_buffer);
        if (m_callback)
            m_callback->dec_ref(m_callback);
    }

    nb::str repr() {
        return nb::str("drjit.Queue[\n"
                       "  buffer={:.2f} KiB,\n"
                       "  msg_types={},\n"
                       "  msg_max_size={} B,\n"
                       "  batch_size={},\n"
                       "  batches={},\n"
                       "  debug={}\n]")
            .format(jit_var_size(m_buffer) * sizeof(uint32_t) / 1024.f,
                    m_msg_types, m_msg_max_size, m_batch_size, m_batches,
                    m_debug ? 1 : 0);
    }
private:
    uint32_t m_buffer;
    uint32_t m_msg_types;
    uint32_t m_msg_max_size;
    uint32_t m_batch_size;
    uint32_t m_batches;
    bool m_debug;
    QueueCallback *m_callback;
};

void export_queue(nb::module_ &m) {
    auto queue = nb::class_<Queue>(m, "Queue", doc_Queue)
        .def(nb::init<nb::handle_t<ArrayBase>, uint32_t, uint32_t, uint32_t, uint32_t,
             nb::capsule, bool>(),
             "buffer"_a, "msg_types"_a, "msg_max_size"_a, "batch_size"_a,
             "batches"_a, "callback"_a, "debug"_a = false,
             doc_Queue_Queue)
        .def("send", &Queue::send, "msg_id"_a, "args"_a, doc_Queue_send)
        .def("__repr__", &Queue::repr);

    nb::class_<Queue::Message>(queue, "Message", doc_Queue_Message)
        .def("response", &Queue::Message::response, doc_Queue_Message_response);
}
