"""
全局线程池单例

入参: get_thread_pool(max_workers=None)
方法: 双重检查锁创建 ThreadPoolExecutor，模块加载时 atexit 注册清理
出参: concurrent.futures.ThreadPoolExecutor 实例
"""
import atexit
import logging
import multiprocessing
import threading
import concurrent.futures

# CPU 核心数与默认线程池大小常量
# 做什么: 提供给调用方按需引用的系统资源指标
# 为什么: 历史 batch_dialog 与 raster_batch 各自定义，统一后消除分歧
CPU_COUNT = multiprocessing.cpu_count()
DEFAULT_THREAD_POOL_SIZE = max(1, CPU_COUNT - 1)  # 留一核给主线程/UI

# 全局线程池实例与互斥锁（模块级单例）
_THREAD_POOL = None
_THREAD_POOL_LOCK = threading.RLock()


def get_thread_pool(max_workers=None):
    """
    获取或创建全局线程池（单例）

    入参:
        max_workers: 工作线程数，None 时用 DEFAULT_THREAD_POOL_SIZE
    方法: 双重检查锁 → 已存在且未关闭则返回 → 否则按 max_workers 创建
    出参: concurrent.futures.ThreadPoolExecutor

    做什么: 为批量处理/渔网分割提供共享线程池，避免每次创建销毁开销
    为什么: 取代 batch_dialog 的 GLOBAL_THREAD_POOL 与 raster_batch 的 THREAD_POOL
            两份独立实现；并补全 raster_batch 缺失的 _shutdown 重启检查
    """
    global _THREAD_POOL
    if max_workers is None:
        max_workers = DEFAULT_THREAD_POOL_SIZE

    with _THREAD_POOL_LOCK:
        if _THREAD_POOL is None or _THREAD_POOL._shutdown:
            _THREAD_POOL = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
            logging.info(f"创建线程池: {max_workers}个工作线程 (CPU: {CPU_COUNT}核)")
        return _THREAD_POOL


def cleanup_thread_pool():
    """
    程序退出时清理线程池资源

    入参: 无
    方法: shutdown(wait=False) 并置 None，吞掉异常避免退出时报错
    出参: 无

    做什么: 释放线程池资源
    为什么: batch_dialog 历史有此逻辑但 raster_batch 缺失（隐性 bug），统一补全
    """
    global _THREAD_POOL
    if _THREAD_POOL is not None:
        try:
            _THREAD_POOL.shutdown(wait=False)
            _THREAD_POOL = None
        except Exception as e:
            logging.error(f"关闭线程池时出错: {str(e)}")


# 模块加载时注册退出钩子（与原 batch_dialog 行为一致）
atexit.register(cleanup_thread_pool)
