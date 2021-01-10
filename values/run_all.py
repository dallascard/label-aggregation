import os
from glob import glob
from optparse import OptionParser
from subprocess import call


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    #parser.add_option('--issue', type=str, default='immigration',
    #                  help='Issue: default=%default')
    #parser.add_option('--by-issue', action="store_true", default=False,
    #                  help='Divide data by issue: default=%default')

    (options, args) = parser.parse_args()

    indir = 'values'
    baseout = 'values_out'

    files = sorted(glob(os.path.join(indir, '*.jsonlist')))
    for infile in files[:2]:
        basename = os.path.basename(infile).split('.')[0]
        outdir = os.path.join(baseout, basename)
        cmd = ['python', 'run.py', infile, outdir, '--response-field', 'count', '--iter', '10000']
        print(' '.join(cmd))
        call(cmd)


if __name__ == '__main__':
    main()
