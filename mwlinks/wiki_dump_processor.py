"""
Go through the Wikipedia dumps and read anchor related information.

Usage:
    wiki_dump_processor -h | --help
    wiki_dump_processor <dump-file>

Options:
    <dump-file>             Path to a set of XML dumps files
                            (pages meta history)
    -h --help               Prints this documentation
"""

import docopt
import mwxml
import mwtypes
from mwlinks.libs.page_parser import parse


def parse_text(dump_file, anchor_replace_map={}, write_text=False, write_anchor_text=False, link_prob=False):
    print(dump_file)

    dump = mwxml.Dump.from_file(mwtypes.files.reader(dump_file))

    for vals in parse(dump, True):
        print(vals)
        input("Wait.")


def main():
    args = docopt.docopt(__doc__, argv=None)

    dump_file = args['<dump-file>']

    parse_text(dump_file)
