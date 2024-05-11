from distutils.ccompiler import new_compiler, show_compilers
from pathlib import Path

def compile_kalman():

    if not Path("./generator/multikalman/kalman.so").is_file():
        try:
            compiler = new_compiler()
            #compiler.add_include_dir("./generator/multikalman/")
            objects = compiler.compile(['./generator/multikalman/kalman.c', './generator/multikalman/vector_op.c'])
            compiler.link_shared_lib(objects, 'multikalman',  output_dir="./generator/multikalman/")
        except Exception as e: 
            print(e)

