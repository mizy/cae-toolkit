#!/usr/bin/env python3
"""Build a platform-specific self-contained CLI with Python/Gmsh and Rust inside."""
import argparse
from pathlib import Path
import platform
import subprocess
import sys
import gmsh

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/cli-release')
    args=parser.parse_args()
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        parser.error('This packaging configuration has only been validated on macOS arm64')
    output=args.output_dir.resolve()
    subprocess.run(['cargo','build','--release','--locked','--manifest-path',str(ROOT/'core/Cargo.toml')],check=True)
    binary=ROOT/'core/target/release/structural-equations'
    gmsh_lib=Path(gmsh.lib._name).resolve()
    subprocess.run([sys.executable,'-m','PyInstaller','--noconfirm','--onefile',
        '--name','structural-solver','--distpath',str(output),
        '--workpath',str(output/'build'),'--specpath',str(output/'build'),
        '--paths',str(ROOT/'scripts'), '--add-binary',f'{binary}:.',
        '--add-binary',f'{gmsh_lib}:.',
        '--exclude-module','matplotlib','--exclude-module','torch',
        '--exclude-module','pyvista','--exclude-module','vtk',
        '--exclude-module','tkinter','--exclude-module','sklearn',
        str(ROOT/'scripts/analyze.py')],check=True)
    print(output/'structural-solver')


if __name__=='__main__':main()
