import time


def eta(t0, done, total):
    if done == 0:
        return "--:--:--"
    return time.strftime(
        "%H:%M:%S", time.gmtime((time.time() - t0) / done * (total - done))
    )


def main():
    pass


if __name__ == "__main__":
    main()
