from concurrent.futures import ThreadPoolExecutor


def prefetch(items,load):
    it=iter(items)
    with ThreadPoolExecutor(max_workers=1) as pool:
        try:current=pool.submit(load,next(it))
        except StopIteration:return
        while current is not None:
            result=current.result()
            try:current=pool.submit(load,next(it))
            except StopIteration:current=None
            yield result
