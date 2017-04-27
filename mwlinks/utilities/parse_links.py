"""
Parse and write  wikilinks from Wikipedia XML database dumps.

Generates a anchor surface counting JSON objects and a wikipedia text with anchor repalced as the Freebase id.

Usage:
    extract -h | --help
    extract [options] <dump-file>...

Options:
    <dump-file>             Path to a set of XML dumps files
                            (pages meta history)
    -h --help               Prints this documentation
"""

import docopt
import mwxml
import mwtypes
from mwlinks.libs import wikilink_extractor
from mwlinks.libs.utils import tsv_encode
from typing import Iterable, Iterator
from mwlinks.libs.common import CaptureResult, Span
from mwlinks.libs.wikilink import Section
from mwlinks.libs.wikilink_extractor import Page, Revision
from mwlinks.libs import utils
import more_itertools


def main(argv=None):
    print(argv)

    args = docopt.docopt(__doc__, argv=argv)

    dump_files = args['<dump-file>']

    run(dump_files)


def parse_revision(mw_page: mwxml.Page,
                   only_last_revision: bool) -> Iterator[Revision]:
    revisions = more_itertools.peekable(mw_page)
    for mw_revision in revisions:
        utils.dot()

        is_last_revision = not utils.has_next(revisions)
        if only_last_revision and not is_last_revision:
            continue

        text = utils.remove_comments(mw_revision.text or '')

        wikilinks = ((wikilink, span)
                     for wikilink, span
                     in wikilink.wikilinks(text, wikilink.sections(text)))

        yield Revision(
            id=mw_revision.id,
            parent_id=mw_revision.parent_id,
            user=mw_revision.user,
            minor=mw_revision.minor,
            comment=mw_revision.comment,
            model=mw_revision.model,
            format=mw_revision.format,
            timestamp=mw_revision.timestamp.to_json(),
            text=text,
            wikilinks=wikilinks
        )


def parse_page(dump: Iterable[mwxml.Page],
               only_last_revision: bool) -> Iterator[Page]:
    for mw_page in dump:
        utils.log("Processing", mw_page.title)

        # Skip non-articles
        if mw_page.namespace != 0:
            utils.log('Skipped (namespace != 0)')
            continue

        revisions_generator = parse_revision(
            mw_page,
            only_last_revision=only_last_revision,
        )

        yield Page(
            id=mw_page.id,
            namespace=mw_page.namespace,
            title=mw_page.title,
            revisions=revisions_generator,
        )

    pass


def run(dump_files):
    for input_file_path in dump_files:
        dump = mwxml.Dump.from_file(mwtypes.files.reader(input_file_path))

        for vals in wikilink_extractor.main(dump, True):
            print("\t".join(tsv_encode(val) for val in vals))

            input("Wait.")
