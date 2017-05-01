"""
Go through the Wikipedia dumps and read anchor related information.

Usage:
    wiki_dump_processor -h | --help
    wiki_dump_processor [options] <dump-file>

Options:
    --write-text            Write raw Wikipedia text
    --write-anchor          Write anchor replaced text
    --write-both            Write both text
    --output-path=OUTPUT    Output directory
    --link-prob             Calculate link probability
    <dump-file>             Path to a set of XML dumps files
                            (pages meta history)
    -h --help               Prints this documentation
"""
from operator import itemgetter
from typing import Iterable, Tuple

import docopt
import mwxml
import mwtypes
import os

import sys

from mwlinks.libs import page_parser
from mwlinks.libs.wikilink import Wikilink
from mwlinks.libs.common import Span
import json
import logging
from mwlinks.libs.WikiExtractor import Extractor
from multiprocessing import Pool, Value, Lock, Queue
from io import StringIO


class SurfaceLinkMap:
    def __init__(self):
        self.__surfaces = []
        self.__surface_indices = {}
        self.__links = []
        self.__index = 0

    def add_surface_link(self, anchor, target):
        try:
            anchor_index = self.__surface_indices[anchor]
        except KeyError:
            anchor_index = self.__index
            self.__surfaces.append(anchor)
            self.__surface_indices[anchor] = anchor_index
            self.__index += 1

        if anchor_index < len(self.__links):
            try:
                self.__links[anchor_index][target] += 1
            except KeyError:
                self.__links[anchor_index][target] = 1
        else:
            self.__links.append({target: 1})

    def get_links(self):
        return self.__links

    def get_anchors(self):
        return self.__surfaces

    def get_anchor_indices(self):
        return self.__surface_indices


def parse(lock, out_strings, anchor_replace_map, sl, item, do_link_prob):
    # sys.stdout.write("\r[%s] Searched %d articles." % (datetime.datetime.now().time(), context_counter.value))
    wiki_id, title, redirect, revision_id, wiki_links, text = item

    if redirect:
        return

    # for name, _ in outputs.items():
    #     out_strings[name] = StringIO()

    for name, out in out_strings.items():
        if name == "origin":
            clean_wiki_text(wiki_links, wiki_id, revision_id, title, text, out, use_plain_text=True,
                            freebase_map=anchor_replace_map)
            # file_out.write(out_strings[name])

        if name == "replaced":
            clean_wiki_text(wiki_links, wiki_id, revision_id, title, text, out, use_link=True,
                            freebase_map=anchor_replace_map)
            # file_out.write(out_strings[name])

        if name == "both":
            clean_wiki_text(wiki_links, wiki_id, revision_id, title, text, out, True, True,
                            freebase_map=anchor_replace_map)
            # file_out.write(out_strings[name])

    if do_link_prob:
        with lock:
            # Currently do not calculate the link probability for the surface terms.
            accumulate_link_prob(sl, wiki_links)


def process(q: Queue, v: Value, lock: Lock, outputs, anchor_replace_map, sl, link_prob):
    while True:
        item = q.get()
        if item is None:
            break

        out_strings = {}

        for name, _ in outputs.items():
            out_strings[name] = StringIO()

        parse(lock, out_strings, anchor_replace_map, sl, item, link_prob)

        with lock:
            v.value += 1
            # print("processing", item)
            sys.stdout.write("\rProcessed %d documents." % v.value)
            sys.stdout.flush()

            for name, out in outputs.items():
                out.write(out_strings[name].getvalue())


def parse_dump(dump_file, output_path, anchor_replace_map={}, write_text=False, write_anchor_text=False,
               write_both=False, link_prob=False, num_cpu=5):
    logging.info("Starting dump parsing.")
    print("Start parsing the dump.")

    dump = mwxml.Dump.from_file(mwtypes.files.reader(dump_file))

    outputs = {}

    if write_text:
        outputs["origin"] = open(os.path.join(output_path, "origin.txt"), 'w')
    if write_anchor_text:
        outputs["replaced"] = open(os.path.join(output_path, "replaced.txt"), 'w')
    if write_both:
        outputs["both"] = open(os.path.join(output_path, "both.txt"), 'w')

    sl = SurfaceLinkMap()

    v = Value('i', 0)
    lock = Lock()
    q = Queue(maxsize=num_cpu)
    p = Pool(num_cpu, initializer=process, initargs=(q, v, lock, outputs, anchor_replace_map, sl, link_prob))

    for element in page_parser.parse(dump, True):
        q.put(element)

    p.close()
    p.join()

    print("")

    if link_prob:
        out = os.io.open(os.path.join(output_path, "prob.json"), encoding='UTF-8', mode='w')
        write_as_json(sl, out)

    logging.info("All done.")


def div_or_nan(numerator, divisor):
    return float('nan') if divisor == 0 else numerator * 1.0 / divisor


def write_as_json(surface_link_map, f):
    # readme = {"surface": 0,  "targets": 4}

    surfaces = surface_link_map.get_anchors()
    surface_links = surface_link_map.get_links()

    count = 0

    for index, surface in enumerate(surfaces):
        surface_info = []

        links = surface_links[index]
        # num_appearance = surface_text_count[index]

        num_linked = 0

        surface_info.append(surface)
        # surface_info.append(num_appearance)

        for link, link_count in links.iteritems():
            num_linked += link_count

        surface_info.append(num_linked)
        # surface_info.append(div_or_nan(num_linked, num_appearance))

        targets = {}
        for link, link_count in links.iteritems():
            targets[link] = (link_count, div_or_nan(link_count, num_linked))
        surface_info.append(targets)

        f.write(json.dumps(surface_info).decode('utf-8'))
        f.write(u"\n")

        count += 1
        sys.stdout.write("\rWrote %d surfaces." % count)
    print("")


def accumulate_link_prob(sl: SurfaceLinkMap, wikilinks: Iterable[Tuple[Wikilink, Span]]):
    for link, span in wikilinks:
        anchor = link.anchor
        target = link.link
        sl.add_surface_link(anchor, target)


def write_cleaned_text(id, revid, title, text, out):
    Extractor(id, revid, title, text.split("\n")).extract(out)


def clean_wiki_text(wikilinks, pageid, revid, title, text, out, use_plain_text=False, use_link=False, freebase_map={}):
    if use_plain_text:
        write_cleaned_text(pageid, revid, title, format_anchor(wikilinks, text), out)
    if use_link:
        write_cleaned_text(pageid, revid, title, format_anchor(wikilinks, text, freebase_map), out)


def format_anchor(wikilinks: Iterable[Tuple[Wikilink, Span]], text, freebase_map=None):
    sorted_links = sorted(wikilinks, key=itemgetter(1))

    for link, span in reversed(sorted_links):
        if freebase_map:
            fb_id = get_freebase_id(freebase_map, link.link)
            if fb_id:
                text = replace_by_index(text, span.begin, span.end, fb_id)
            else:
                text = replace_by_index(text, span.begin, span.end, link.link)
        else:
            text = replace_by_index(text, span.begin, span.end, link.anchor)

    return text


def get_freebase_id(freebase_map, wiki_title):
    try:
        return freebase_map[wiki_title.replace(" ", "_")]
    except KeyError:
        return None


def replace_by_index(text, begin, end, replacement):
    # print("replacing " + text[begin:end] + " by " + replacement)
    # input()
    return text[:begin] + replacement + text[end:]


def main():
    args = docopt.docopt(__doc__, argv=None)

    dump_file = args['<dump-file>']
    write_text = args['--write-text']
    write_anchor_text = args['--write-anchor']
    write_both = args['--write-both']
    link_prob = args['--link-prob']

    output_path = args['--output-path']

    # logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    handler = logging.FileHandler('dump.log')
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(handler)

    parse_dump(dump_file, output_path, write_text=write_text, write_anchor_text=write_anchor_text,
               write_both=write_both, link_prob=link_prob, num_cpu=8)
