# SPDX-License-Identifier: Apache-2.0
"""Shared-KV causal SDPA for T = 2..8 verification rows (bitwise equal to per-row paged SDPA).

The reference verification SDPA runs ttnn paged_scaled_dot_product_attention_decode with every row
in its own batch (eight cores per (row, KV head)): each row's K/V is streamed separately. Rows whose
floor(position/64) is equal share each core's K-chunk range and the tree-reduction schedule
(verify_sdpa_common.hpp), so the reference compute kernel (sdpa_flash_decode.cpp, unmodified,
identical compile-time configuration) runs once per (pass, row block, KV head) item with the 6 Q
heads of 4 rows packed into one 32-row tile, and each row keeps its own causal mask. All per-row
arithmetic is row-local (QK/PV matmuls, row max/sum, exp, rescale), so outputs equal the reference
bit for bit. One launch runs every item: T rows in B = ceil(T/4) row blocks give at most B + 1
items (rows past a 64-token chunk boundary form pass 1), each on its own 32-core slot. The two
items of one pass read the same K/V chunks, so their cores split the DRAM reads (K / V) and swap
halves over the NoC.

Q is packed by rope_kv_many (ROPE_Q_PACKED): [1, 1, 128 * B, 256], tile row (kv_head * B + b)
holds rows 4b..4b+3 x the head's 6 Q heads in tile rows 0..23 (24..31 zero).
"""

import struct

import ttnn

_ROOT = "models/demos/blackhole/qwen36/tt/attention/"
_COMPUTE = "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/sdpa_flash_decode.cpp"
_TILE = 2048
KV_HEADS = 4
HEAD_DIM = 256
ROWS_PER_TILE = 4
CORES_PER_HEAD = 8  # the reference split per (row, KV head)
K_CHUNK = 64        # reference q_chunk 32 / k_chunk 64 program config
MAX_ROUNDS = 6
GRID_X = 11
# Reference reader DRAM barrier cadence (get_barrier_read_threshold for the 64-core, 2-row-block launch).
READ_BARRIER_TILES = 4
KV_SLOTS = 4  # K / V CB depth in chunks: consumed, exchanged, and two in flight from DRAM


def blocks(rows):
    return (rows + ROWS_PER_TILE - 1) // ROWS_PER_TILE


def packed_q_shape(rows):
    return (1, 1, 32 * KV_HEADS * blocks(rows), HEAD_DIM)


def _tree_params(core_id, n):
    """sdpa_decode_device_operation.hpp get_tree_reduction_params (n a power of two)."""
    children = [0xFFFFFFFF] * MAX_ROUNDS
    parent = send_at = 0xFFFFFFFF
    num_children = active_rounds = 0
    if n <= 1:
        return True, parent, send_at, num_children, active_rounds, children
    vid = (n - 1) - core_id
    rounds = (n - 1).bit_length()
    is_root = core_id == 0
    for r in range(rounds):
        mask = (2 << r) - 1
        if vid & mask == mask:
            child = vid - (1 << r)
            if child < n:
                children[r] = (n - 1) - child
                num_children += 1
                active_rounds = r + 1
    if not is_root:
        trailing = ((~vid) & -(~vid)).bit_length() - 1
        parent_vid = vid + (1 << trailing)
        parent = (n - 1) - parent_vid if parent_vid < n else 0
        send_at = trailing
    return is_root, parent, send_at, num_children, active_rounds, children


def _accessors(tensors):
    args = []
    for tensor in tensors:
        accessor = ttnn.TensorAccessorArgs(tensor)
        assert not accessor.get_common_runtime_args(), "verify_sdpa accessors must be compile-time"
        args.extend(accessor.get_compile_time_args())
    return args


def _bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _pack_two_bf16(value):
    bf = struct.unpack("<I", struct.pack("<f", value))[0] >> 16
    return (bf << 16) | bf


def _program(q, cache_key, cache_value, positions, page_table, output, scale, rows):
    device = q.device()
    slots = blocks(rows) + 1
    cores_per_slot = KV_HEADS * CORES_PER_HEAD
    active = slots * cores_per_slot
    core_list = [ttnn.CoreCoord(i % GRID_X, i // GRID_X) for i in range(active)]
    phys = [device.worker_core_from_logical_core(c) for c in core_list]
    core_set = ttnn.CoreRangeSet({ttnn.CoreRange(c, c) for c in core_list})
    width = int(page_table.shape[-1])
    block = int(cache_key.shape[2])
    st = width * block // 32
    dht = vdht = HEAD_DIM // 32
    sk = K_CHUNK // 32
    rounds = (CORES_PER_HEAD - 1).bit_length()
    pos_stick = 64
    table_stick = ((width * 4 + 63) // 64) * 64

    reader_ct = [rows, dht, vdht, sk, K_CHUNK, int(positions.padded_shape[-1]) * 4, KV_HEADS, block // 32,
                 CORES_PER_HEAD, slots, READ_BARRIER_TILES, 1, 2, KV_SLOTS] + _accessors(
                     (q, cache_key, cache_value, positions, page_table))
    writer_ct = [rows, vdht, sk, _pack_two_bf16(1.0), active, 0, K_CHUNK, CORES_PER_HEAD] + _accessors((output,))
    pnht = 1
    dst = 8  # fp32_dest_acc_en = False
    qk_w = min(sk, dst)
    qk_h = min(pnht, dst // qk_w) if qk_w == sk else 1
    out_w = min(vdht, dst)
    out_h = min(pnht, dst // out_w) if out_w == vdht else 1
    compute_ct = [st, dht, vdht, pnht, sk, dht, qk_w, qk_h, pnht // qk_h, sk // qk_w, 1, sk, out_w, out_h,
                  pnht // out_h, vdht // out_w, 1, cores_per_slot, K_CHUNK, CORES_PER_HEAD, 1, 1, 0, 0, dst, 0,
                  1, 0, _bits(scale), 0, rounds, block]
    mul_bcast = min(pnht * sk, dst)
    defines = [("EXP_APPROX_MODE", "0"), ("DHT_GRANULARITY", "8"), ("LOG2_DHT_GRANULARITY", "3"),
               ("SUB_EXP_GRANULARITY", str(min(sk, dst))), ("LOG2_SUB_EXP_GRANULARITY", str(min(sk, dst).bit_length() - 1)),
               ("MUL_BCAST_GRANULARITY", str(mul_bcast)), ("LOG2_MUL_BCAST_GRANULARITY", str(mul_bcast.bit_length() - 1)),
               ("STATS_GRANULARITY", str(min(sk, dst))), ("LOG2_STATS_GRANULARITY", str(min(sk, dst).bit_length() - 1))]

    reader_rt, writer_rt, compute_rt = [], [], []
    addresses = [int(t.buffer_address()) for t in (q, cache_key, cache_value, positions, page_table)]
    for i, core in enumerate(core_list):
        slot, within = divmod(i, cores_per_slot)
        head, member = divmod(within, CORES_PER_HEAD)
        is_root, parent, send_at, num_children, active_rounds, children = _tree_params(member, CORES_PER_HEAD)
        base = slot * cores_per_slot + head * CORES_PER_HEAD
        group = phys[base:base + CORES_PER_HEAD]
        twins = [phys[s * cores_per_slot + within] for s in range(slots)]
        reader_rt.append((core, addresses + [table_stick, head, slot, member, *[p.x for p in twins],
                                             *[p.y for p in twins]]))
        writer_rt.append((core, [int(output.buffer_address()), head, slot, member, int(is_root), parent, send_at,
                                 *children, *[p.x for p in group], *[p.y for p in group]]))
        compute_rt.append((core, [int(member == 0), int(within == 0), head, slot, member, within, 0xFFFFFFFF,
                                  int(is_root), parent, send_at, num_children, active_rounds, *children]))
    reader = ttnn.KernelDescriptor(
        kernel_source=_ROOT + "verify_sdpa_reader.cpp", core_ranges=core_set, compile_time_args=reader_ct,
        runtime_args=reader_rt, config=ttnn.ReaderConfigDescriptor())
    writer = ttnn.KernelDescriptor(
        kernel_source=_ROOT + "verify_sdpa_writer.cpp", core_ranges=core_set, compile_time_args=writer_ct,
        runtime_args=writer_rt, config=ttnn.WriterConfigDescriptor())
    compute = ttnn.KernelDescriptor(
        kernel_source=_COMPUTE, core_ranges=core_set, compile_time_args=compute_ct, defines=defines,
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False,
                                            dst_full_sync_en=False, math_approx_mode=True))

    full = ttnn.TileDescriptor(ttnn.Tile((32, 32)))
    kv_bytes = 1088 if cache_key.dtype == ttnn.bfloat8_b else _TILE
    out_tiles = pnht * vdht

    def cb(index, tiles, dtype=ttnn.bfloat16, page=_TILE, tile=full):
        return ttnn.CBDescriptor(total_size=tiles * page, core_ranges=core_set,
                                 format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, page, tile)])

    def stick(index, size, dtype):
        return ttnn.CBDescriptor(total_size=size, core_ranges=core_set,
                                 format_descriptors=[ttnn.CBFormatDescriptor(index, dtype, size)])

    cbs = [
        cb(0, pnht * dht),
        cb(1, sk * dht * KV_SLOTS, cache_key.dtype, kv_bytes),
        cb(2, sk * vdht * KV_SLOTS, cache_value.dtype, kv_bytes),
        cb(3, pnht * sk),
        cb(5, 1), cb(6, pnht), cb(7, pnht),
        stick(8, pos_stick, ttnn.int32), stick(15, pos_stick, ttnn.int32), stick(9, table_stick, ttnn.int32),
        cb(11, 1), cb(12, 1),
        cb(24, pnht * sk), cb(25, out_tiles), cb(26, out_tiles),
        *[cb(i, pnht) for i in (27, 28, 29, 30, 31, 21, 22)],
        cb(23, out_tiles), cb(16, out_tiles), cb(17, pnht), cb(18, pnht),
        # Children write at send_at_round offsets: one block per reduction round suffices.
        cb(19, (out_tiles + 2 * pnht) * rounds),
        cb(20, out_tiles),
    ]
    # 0: tree reduction (reference); 1: K/V partner credit; 2: K/V partner half arrived.
    semaphores = [ttnn.SemaphoreDescriptor(id=i, core_ranges=core_set, initial_value=0) for i in range(3)]
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], cbs=cbs, semaphores=semaphores)


def verify_sdpa(q, cache_key, cache_value, positions, page_table, scale, output=None):
    """Causal attention of T verification rows -> [1, T, 24*256] BF16 DRAM."""
    rows = int(positions.shape[0])
    if not 2 <= rows <= 8:
        raise ValueError("verify_sdpa serves 2..8 verification rows")
    if tuple(int(v) for v in q.shape) != packed_q_shape(rows) or q.dtype != ttnn.bfloat16:
        raise ValueError(f"verify_sdpa expects packed BF16 Q {packed_q_shape(rows)}")
    if (cache_key.dtype != cache_value.dtype or cache_key.dtype not in (ttnn.bfloat16, ttnn.bfloat8_b)
            or int(cache_key.shape[1]) != KV_HEADS or int(cache_key.shape[3]) != HEAD_DIM
            or int(cache_key.shape[2]) % 32):
        raise ValueError("verify_sdpa expects BF16/BFP8 paged caches [blocks,4,block,256]")
    if (tuple(positions.shape) != (rows,) or positions.dtype != ttnn.int32
            or int(page_table.shape[0]) != rows or page_table.dtype != ttnn.int32):
        raise ValueError("verify_sdpa expects int32 positions [T] and page_table [T, W]")
    if output is None:
        output = ttnn.empty((1, rows, 24 * HEAD_DIM), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=q.device(), memory_config=ttnn.DRAM_MEMORY_CONFIG)
    program = _program(q, cache_key, cache_value, positions, page_table, output, scale, rows)
    ttnn.generic_op([q, cache_key, cache_value, positions, page_table, output], program)
    return output
