from distutils.core import setup, Extension

def main():
    setup(name="myadd",
          version="1.0.0",
          description="Python interface for the myadd C library function",
          author="Nadiah",
          author_email="nadiah@nadiah.org",
          ext_modules=[Extension("myadd", ["myadd.cpp"])],
          )


if __name__ == "__main__":
    main()