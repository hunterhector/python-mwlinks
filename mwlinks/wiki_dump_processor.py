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
    --freebase-map=FB_MAP   Freebase mapping file
    --redirect-path=REDIRECTS   Redirect file
    <dump-file>             Path to a set of XML dumps files
                            (pages meta history)
    -h --help               Prints this documentation
"""
from operator import itemgetter
from typing import Iterable, Tuple, Dict, Set

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
from multiprocessing import Pool, Value, Lock, Queue, Manager
from io import StringIO
import datetime

links_to_ignore = {"File", "Category"}


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


def parse(lock, out_files, failed_pages, redirects, freebase_map, sl, item, do_link_prob):
    wiki_id, title, redirect, revision_id, wiki_links, text = item

    # TODO: process the links first before.

    if redirect:
        return

    for key, out_file in out_files.items():
        out = StringIO()
        if key == "origin":
            clean_wiki_text(wiki_links, wiki_id, revision_id, title, text, out, use_plain_text=True,
                            freebase_map=freebase_map, redirects=redirects)
            out_file.write(out.getvalue())

        if key == "replaced":
            clean_wiki_text(wiki_links, wiki_id, revision_id, title, text, out, use_link=True,
                            freebase_map=freebase_map, redirects=redirects)
            out_file.write(out.getvalue())

        if key == "both":
            clean_wiki_text(wiki_links, wiki_id, revision_id, title, text, out, True, True,
                            freebase_map=freebase_map, redirects=redirects)
            both_text = out.getvalue()
            out_file.write(mix_up(both_text))

    if do_link_prob:
        with lock:
            # Currently do not calculate the link probability for the surface terms.
            accumulate_link_prob(sl, wiki_links, redirects, freebase_map, failed_pages)


def process_links():
    pass

def mix_up(text):
    lines = text.strip().split("\n")
    middle = len(lines) // 2

    first_half = lines[:middle]
    second_half = lines[middle:]

    mixed_text = []
    for f, s in zip(first_half, second_half):
        mixed_text.append(f)
        mixed_text.append(s)
    return "\n".join(mixed_text) + "\n"


def process(q: Queue, v: Value, lock: Lock, outputs, redirects, anchor_replace_map, sl, link_prob, failed_pages):
    while True:
        item = q.get()
        if item is None:
            break

        parse(lock, outputs, failed_pages, redirects, anchor_replace_map, sl, item, link_prob)

        with lock:
            v.value += 1
            sys.stdout.write("\r[%s] Processed %d documents." % (datetime.datetime.now().time(), v.value))
            sys.stdout.flush()


def parse_dump(dump_file, output_path, redirects={}, wiki_2_fb={}, write_text=False, write_anchor_text=False,
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
    failed_pages = Manager().dict()
    p = Pool(num_cpu, initializer=process,
             initargs=(q, v, lock, outputs, redirects, wiki_2_fb, sl, link_prob, failed_pages))

    count = 100
    for element in page_parser.parse(dump, True):
        q.put(element)
        count -= 1
        if count == 0:
            break

    print("\nParsing done.")
    p.close()

    for key, out in outputs.items():
        out.close()

    out_wiki_not_found = open(os.path.join(output_path, "wiki_not_found.txt"), encoding='UTF-8', mode='w')
    for p, _ in failed_pages.items():
        print(p)
        out_wiki_not_found.write(p)
        out_wiki_not_found.write("\n")
    out_wiki_not_found.close()

    print("")

    if link_prob:
        out = open(os.path.join(output_path, "prob.json"), encoding='UTF-8', mode='w')
        write_as_json(sl, out)

    print("All done.")


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


def accumulate_link_prob(sl: SurfaceLinkMap, wikilinks: Iterable[Tuple[Wikilink, Span]],
                         redirects: Dict, freebase_map: Dict, failed_pages):
    for link, span in wikilinks:
        anchor = link.anchor
        target = link.link

        fb_id = get_freebase_id(freebase_map, redirects, target)

        if fb_id:
            sl.add_surface_link(anchor, fb_id)
        else:
            # print("Accumulation failed " + target)
            failed_pages[target] = '0'

            # print("%d pages failed " %len(failed_pages))


def write_cleaned_text(id, revid, title, text, out):
    Extractor(id, revid, title, text.split("\n")).extract(out)


def clean_wiki_text(wikilinks, pageid, revid, title, text, out, use_plain_text=False, use_link=False, redirects={},
                    freebase_map={}):
    if use_plain_text:
        write_cleaned_text(pageid, revid, title, format_anchor(wikilinks, text, redirects), out)
    if use_link:
        write_cleaned_text(pageid, revid, title, format_anchor(wikilinks, text, redirects, freebase_map), out)


def format_anchor(wikilinks: Iterable[Tuple[Wikilink, Span]], text, redirects: Dict, freebase_map=None):
    sorted_links = sorted(wikilinks, key=itemgetter(1))

    for link, span in reversed(sorted_links):
        if freebase_map:
            fb_id = get_freebase_id(freebase_map, redirects, link.link)
            if fb_id:
                text = replace_by_index(text, span.begin, span.end, fb_id)
            else:
                text = replace_by_index(text, span.begin, span.end, link.link)
        else:
            text = replace_by_index(text, span.begin, span.end, link.anchor)

    return text


def is_ignore_link(link: Wikilink):
    link_parts = link.link.split(":")
    if len(link_parts) > 1:
        link_type = link_parts[0]
        if link_type in links_to_ignore:
            return True

    return False


def get_freebase_id(freebase_map: Dict, redirects, link):
    wiki_title = get_wiki_title(link)

    if wiki_title in redirects:
        wiki_title = redirects[wiki_title]

    try:
        return freebase_map[wiki_title]
    except KeyError:
        return None


def get_wiki_title(link):
    """
    Normalize the link name of the link, such as replacing space, and first letter capitalization. 
    See: https://en.wikipedia.org/wiki/Wikipedia:Naming_conventions_(capitalization)#Software_characteristics
    :param link: 
    :return: 
    """
    return link.replace(" ", "_").title()


def replace_by_index(text, begin, end, replacement):
    # print("replacing " + text[begin:end] + " by " + replacement)
    # input()
    return text[:begin] + replacement + text[end:]


def read_wiki_fb_mapping(mapping_file):
    wiki_2_fb = {}
    with open(os.path.join(mapping_file)) as mapping:
        for line in mapping:
            fb_id, wikipage_name = line.strip().split("\t")[0:2]
            formatted_wiki_name = wikipage_name.replace(" ", "_")
            wiki_2_fb[formatted_wiki_name] = fb_id
    return wiki_2_fb


def main():
    args = docopt.docopt(__doc__, argv=None)

    dump_file = args['<dump-file>']
    write_text = args['--write-text']
    write_anchor_text = args['--write-anchor']
    write_both = args['--write-both']
    link_prob = args['--link-prob']

    free_base_mapping = args['--freebase-map']
    output_path = args['--output-path']
    redirect_path = args['--redirect-path']

    sys.path.append("../../projects/KnowledgeIR")
    from linker.data import data_utils

    logging.info("Loading redirect pages.")
    redirects = data_utils.run_or_load(os.path.join(output_path, "redirects.pickle"), data_utils.load_redirects,
                                       redirect_path)
    logging.info("Done")

    print("Loading Wikipedia to Freebase.")
    wiki_2_fb = read_wiki_fb_mapping(free_base_mapping)
    print("Done.")

    # logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
    # handler = logging.FileHandler('dump.log')
    # handler.setLevel(logging.INFO)
    # handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    # logging.getLogger().addHandler(handler)

    parse_dump(dump_file, output_path, write_text=write_text, write_anchor_text=write_anchor_text,
               write_both=write_both, link_prob=link_prob, wiki_2_fb=wiki_2_fb, redirects=redirects, num_cpu=8)
