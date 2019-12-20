import multiprocessing


class config:
    use_milp = True
    numprocesses_milp = multiprocessing.cpu_count()
    dyn_krelu = False
    use_2relu = False
    use_3relu = False
    numproc_krelu = multiprocessing.cpu_count()