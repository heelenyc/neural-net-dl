import pyopencl as cl
import numpy

# 获取平台和设备
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]

# 创建上下文和命令队列
context = cl.Context([device])
queue = cl.CommandQueue(context, device)

# 创建内核源代码
kernel_source = """
__kernel void vec_add(__global float *a, __global float *b, __global float *c) {
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}
"""

# 创建程序和内核
program = cl.Program(context, kernel_source).build()
vec_add = program.vec_add

for i in range(1, 100000):
    # 创建输入数据
    m = 1024
    a = numpy.random.rand(m).astype(numpy.float32)
    b = numpy.random.rand(m).astype(numpy.float32)
    c = numpy.zeros_like(a)

    # 创建GPU内存缓冲区并传输数据
    a_gpu_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=a.nbytes)
    b_gpu_buf = cl.Buffer(context, cl.mem_flags.READ_ONLY, size=b.nbytes)
    c_gpu_buf = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, size=c.nbytes)

    # 将数据从Host传输到Device
    cl.enqueue_copy(queue, a_gpu_buf, a)
    cl.enqueue_copy(queue, b_gpu_buf, b)
    # queue.enqueue_write_buffer(a_gpu_buf, a)
    # queue.enqueue_write_buffer(b_gpu_buf, b)

    # 设置内核参数并执行
    global_size = m
    local_size = 64
    vec_add(queue, (global_size,), (local_size,), a_gpu_buf, b_gpu_buf, c_gpu_buf)

    # 从设备读取结果到Host
    cl.enqueue_copy(queue, c, c_gpu_buf)
    # queue.enqueue_read_buffer(c_gpu_buf, c)

    # 检查结果
    print("Result:", c)
    print("Expected:", a + b)
