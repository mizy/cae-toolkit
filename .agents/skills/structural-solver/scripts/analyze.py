#!/usr/bin/env python3
"""Analyze model + conditions with the shared Rust CPU/GPU core."""
import argparse
import base64
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from serve_web import prepare_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, nargs='?', help='JSON containing model and conditions')
    parser.add_argument('--model', type=Path, help='single closed STL file')
    parser.add_argument('--conditions', type=Path, help='physical conditions JSON for --model')
    parser.add_argument('--units', choices=['m','mm'], help='required for --model; STL has no units')
    parser.add_argument('--backend', choices=['auto','gpu','cpu'], default='auto')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.model:
        if args.input or not args.conditions or not args.units:
            parser.error('--model requires --conditions and --units, without positional input JSON')
    elif not args.input or args.conditions or args.units:
        parser.error('supply input.json, or --model part.stl --conditions conditions.json --units mm')
    try:
        if args.model:
            if args.model.suffix.lower() != '.stl' or args.model.stat().st_size > 10*1024*1024:
                raise ValueError('--model requires an STL file of at most 10 MB')
            request = {'model': {'geometry':'stl','units':args.units,'name':args.model.name,
                                'stl_base64':base64.b64encode(args.model.read_bytes()).decode()},
                       'conditions':json.loads(args.conditions.read_text())}
        else:
            request = json.loads(args.input.read_text())
        case = prepare_case(request['model'], request['conditions'])
        if getattr(sys, 'frozen', False):
            binary = Path(sys._MEIPASS) / 'structural-equations'
        else:
            binary = Path(__file__).resolve().parents[1] / 'core/target/release/structural-equations'
        with tempfile.TemporaryDirectory(prefix='structural-equations-') as folder:
            path = Path(folder) / 'case.json'
            path.write_text(json.dumps(case))
            run = subprocess.run([str(binary),str(path),args.backend], capture_output=True, text=True)
            if run.returncode:
                raise RuntimeError(run.stderr.strip() or 'Equation solve failed')
            result = json.loads(run.stdout)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({'model':case['model'],'conditions':case['conditions'],
            'coordinates':case['coordinates'],'metadata':case['metadata'],**result}, allow_nan=False))
        print(f"{result['backend']}: residual={result['relativeResidual']:.4g}, iterations={result['iterations']}")
        if result.get('fallbackReason'):
            print(f"CPU fallback: {result['fallbackReason']}", file=sys.stderr)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
