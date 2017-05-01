"""
Extracts wikilinks from Wikipedia XML database dumps.

Generates a TSV dataset with one row per revision with the following fields.

 * mw_page.id,
 * mw_page.title,
 * revision.id,
 * revision.parent_id,
 * revision.timestamp,
 * user_type,
 * user_username,
 * user_id,
 * revision_minor,
 * wikilink.link,
 * wikilink.anchor,
 * wikilink.section_name,
 * wikilink.section_level,
 * wikilink.section_number

Usage:
    extract -h | --help
    extract [options] <dump-file>...

Options:
    --only-last-revision    Consider only the last revision for each page.
    <dump-file>             Path to a set of XML dumps files
                            (pages meta history)
    -h --help               Prints this documentation
"""

# import mw.xml_dump
import json
import sys
import argparse

import docopt
import mwxml
import mwtypes

from typing import IO, Optional, Union

from mwlinks.libs import wikilink_extractor
from mwlinks.libs.utils import tsv_encode


# def open_xml_file(path: Union[str, IO]):
#     """Open an xml file, decompressing it if necessary."""
#     f = mw.xml_dump.functions.open_file(
#         mw.xml_dump.functions.file(path)
#     )
#     return f


def main(argv=None):
    args = docopt.docopt(__doc__, argv=argv)

    dump_files = args['<dump-file>']
    last_revision = args['--only-last-revision']

    print(last_revision)

    input()

    run(dump_files, last_revision)


def run(dump_files, last_revision):

    for input_file_path in dump_files:
        dump = mwxml.Dump.from_file(mwtypes.files.reader(input_file_path))

        print("\t".join(('page_id', 'page_title', 'revision_id',
                         'revision_parent_id', 'revision_timestamp',
                         'user_type', 'user_username', 'user_id',
                         'revision_minor', 'wikilink.link', 'wikilink.anchor',
                         'wikilink.section_name', 'wikilink.section_level',
                         'wikilink.section_number')))

        for vals in wikilink_extractor.main(dump, last_revision):

            # for val in vals:
            #     print(val)

            print("\t".join(tsv_encode(val) for val in vals))


if __name__ == '__main__':
    main()
