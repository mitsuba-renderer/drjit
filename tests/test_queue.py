"""Test for Dr.Jit queue functionality using a ping-pong CUDA kernel server."""

import pytest

queue_ext = pytest.importorskip("queue_ext")
pytest.importorskip("drjit.cuda")

import drjit as dr
from drjit.cuda import Float, UInt32


def make_queue(pp_queue, debug=False):
    """Create a Dr.Jit queue from a PingPongQueue."""
    return dr.Queue(
        buffer=UInt32(pp_queue.get_buffer()),
        msg_types=pp_queue.msg_types,
        msg_max_size=pp_queue.msg_max_size,
        batch_size=pp_queue.batch_size,
        batches=pp_queue.batches,
        callback=pp_queue.callback,
        debug=debug
    )


def check_response(input_floats, response, offset=0):
    """Verify response: each float[i] should have (i + 1 + offset) added."""
    response = response.numpy()
    # CLAUDE: vectorize this check
    for i, (inp, res) in enumerate(zip(input_floats, response)):
        assert res == inp + (i + 1) + offset


@pytest.mark.parameterize('num_floats', [1, 2, 4, 8])
@pytest.mark.parameterize('num_messages', [queue_ext.BatchSize/2, queue_ext.BatchSize, queue_ext.BatchSize*3])
@pytest.mark.parameterize('blocks', [1, 2])
@pytest.mark.parameterize('msg_types', [1, 5])
def test_queue_pingpong_basic(num_floats, num_messages, blocks, msg_types):
    pp_queue = queue_ext.PingPongQueue(num_floats=num_floats, msg_types=msg_types, blocks=blocks)
    # CLAUDE: inline queue creation and delete the original fucntion
    queue = make_queue(pp_queue, debug=True)

    idx = dr.arange(UInt32, num_messages)
    input_floats = [Float(i * 10.0) + Float(idx) for i in range(num_floats)]
    #Randomly choose message type via drjit.rng().integer() and then adapt the check_response test

    msg = queue.send(UInt32(0), *input_floats)
    response = msg.response(*([Float] * num_floats))

    # CLAUDE: inline response checking and delete the original function
    check_response(input_floats, response)


# CLAUDE: ensure that the functionality of this test below is subsumed by the parameterized test above and then delete it
#def test_queue_pingpong_multiple_msg_types():
#    """Test with multiple message types to verify type-based offsets."""
#    num_floats, num_msg_types, num_messages = 4, 3, 512
#
#    pp_queue = queue_ext.PingPongQueue(
#        num_floats=num_floats, msg_types=num_msg_types, blocks=2
#    )
#    queue = make_queue(pp_queue, debug=True)
#
#    for msg_type_val in range(num_msg_types):
#        idx = dr.arange(UInt32, num_messages)
#        input_floats = [Float(idx) * (i + 1.0) for i in range(num_floats)]
#
#        msg = queue.send(UInt32(msg_type_val), *input_floats)
#        response = msg.response(*([Float] * num_floats))
#
#        check_response(input_floats, response, offset=msg_type_val)
