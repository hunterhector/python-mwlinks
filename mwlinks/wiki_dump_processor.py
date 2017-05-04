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
import logging.config
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


def parse(out_files, item):
    wiki_id, title, redirect, revision_id, sorted_links, text = item

    for key, out_file in out_files.items():
        out = StringIO()
        if key == "origin":
            clean_wiki_text(sorted_links, wiki_id, revision_id, title, text, out, use_plain_text=True)
            out_file.write(out.getvalue())

        if key == "replaced":
            clean_wiki_text(sorted_links, wiki_id, revision_id, title, text, out, use_link=True)
            out_file.write(out.getvalue())

        if key == "both":
            clean_wiki_text(sorted_links, wiki_id, revision_id, title, text, out, True, True)
            both_text = out.getvalue()
            out_file.write(mix_up(both_text))


def process_links(wikilinks: Iterable[Tuple[Wikilink, Span]], freebase_map, redirects, gotchas, misses):
    links = []
    for link, span in wikilinks:
        if not is_ignore_link(link):
            wiki_title = get_wiki_title(link.link)
            if wiki_title in redirects:
                wiki_title = redirects[wiki_title]

            if wiki_title in freebase_map:
                links.append((link.anchor, freebase_map[wiki_title], span))
                gotchas[wiki_title] = 0
            else:
                misses[wiki_title] = 0

    return links


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


def process(q: Queue, v: Value, lock: Lock, outputs):
    while True:
        item = q.get()
        if item is None:
            break

        parse(outputs, item)

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
    found_pages = Manager().dict()

    p = Pool(num_cpu, initializer=process, initargs=(q, v, lock, outputs))

    # count = 100
    for wiki_id, title, redirect, revision_id, wiki_links, text in page_parser.parse(dump, True):
        links = process_links(wiki_links, wiki_2_fb, redirects, found_pages, failed_pages)
        sorted_links = sorted(links, key=itemgetter(2))
        element = (wiki_id, title, redirect, revision_id, sorted_links, text)
        q.put(element)

        if link_prob:
            # Currently do not calculate the link probability for the surface terms.
            for anchor, fb_id, span in sorted_links:
                sl.add_surface_link(anchor, fb_id)

                # count -= 1
                # if count == 0:
                #     break

    print("\nParsing done.")
    p.close()

    for key, out in outputs.items():
        out.close()

    out_wiki_not_found = open(os.path.join(output_path, "wiki_not_found.txt"), encoding='UTF-8', mode='w')
    out_wiki_not_found.write("%d links found, %d links missed.\n" % (len(found_pages), len(failed_pages)))
    out_wiki_not_found.write("================================\n")
    for p, _ in failed_pages.items():
        out_wiki_not_found.write(p)
        out_wiki_not_found.write("\n")
    out_wiki_not_found.close()

    print("%d links found, %d links missed." % (len(found_pages), len(failed_pages)))

    if link_prob:
        write_as_json(sl, os.path.join(output_path, "prob.json"))

    print("All done.")


def div_or_nan(numerator, divisor):
    return float('nan') if divisor == 0 else numerator * 1.0 / divisor


def write_as_json(surface_link_map, out_path):
    surfaces = surface_link_map.get_anchors()
    surface_links = surface_link_map.get_links()

    count = 0

    all_surface_info = {}

    for index, surface in enumerate(surfaces):
        surface_info = []

        links = surface_links[index]

        num_linked = 0

        # surface_info.append(surface)
        # surface_info.append(num_appearance)

        for link, link_count in links.items():
            num_linked += link_count

        surface_info.append(num_linked)
        # surface_info.append(div_or_nan(num_linked, num_appearance))

        targets = {}
        for link, link_count in links.items():
            targets[link] = (link_count, div_or_nan(link_count, num_linked))
        surface_info.append(targets)

        all_surface_info[surface] = surface_info

        # f.write(json.dumps(surface_info).decode('utf-8'))
        # f.write(u"\n")

        # out = open(os.path.join(output_path, "prob.json"), encoding='UTF-8', mode='w')
        count += 1
        sys.stdout.write("\rCollected %d surfaces." % count)

    with open(out_path, 'w') as out:
        json.dump(all_surface_info, out, indent=4)

    print("\nFinished writing surface information.")


def accumulate_link_prob(sl: SurfaceLinkMap, links):
    for anchor, fb_id, span in links:
        sl.add_surface_link(anchor, fb_id)


def write_cleaned_text(id, revid, title, text, out):
    Extractor(id, revid, title, text.split("\n")).extract(out)


def clean_wiki_text(links, pageid, revid, title, text, out, use_plain_text=False, use_link=False):
    if use_plain_text:
        write_cleaned_text(pageid, revid, title, format_anchor(links, text), out)
    if use_link:
        write_cleaned_text(pageid, revid, title, format_anchor(links, text, True), out)


def format_anchor(links, text, use_freebase=False):
    for anchor, fb_id, span in reversed(links):
        if use_freebase:
            text = replace_by_index(text, span.begin, span.end, fb_id)
        else:
            text = replace_by_index(text, span.begin, span.end, anchor)

    return text


def is_ignore_link(link: Wikilink):
    link_parts = link.link.split(":")
    if len(link_parts) > 1:
        link_type = link_parts[0]
        if link_type in links_to_ignore:
            return True

    return False


def get_wiki_title(link):
    """
    Normalize the link name of the link, such as replacing space, and first letter capitalization. 
    See: https://en.wikipedia.org/wiki/Wikipedia:Naming_conventions_(capitalization)#Software_characteristics
    :param link: 
    :return: 
    """
    return cap_first(link.replace(" ", "_"))


def cap_first(s):
    return s[:1].upper() + s[1:]


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

    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    root.handlers = []
    handler = logging.FileHandler(os.path.join(output_path, "dump.log"))
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    root.addHandler(handler)

    parse_dump(dump_file, output_path, write_text=write_text, write_anchor_text=write_anchor_text,
               write_both=write_both, link_prob=link_prob, wiki_2_fb=wiki_2_fb, redirects=redirects, num_cpu=8)
