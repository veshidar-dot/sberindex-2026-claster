"""Полный прогон одной командой: сравнение методов -> итог и динамика -> внешняя проверка -> лендинг.

    python -m pulse.run
"""
import time

from . import compare, export, external, final


def main():
    for name, step in (("сравнение методов", compare.main), ("итог, уверенность, динамика", final.main),
                       ("внешняя проверка (Росстат, мобильность)", external.main), ("лендинг", export.main)):
        t = time.time()
        print(f">> {name}", flush=True)
        step()
        print(f"   {time.time() - t:.0f} с", flush=True)


if __name__ == "__main__":
    main()
