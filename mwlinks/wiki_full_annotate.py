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
import logging
from mwlinks.libs.WikiExtractor import Extractor
from multiprocessing import Pool, Value, Lock, Queue, Manager
from io import StringIO
import datetime
import json


def write_anchor_spotted(output, freebase_map, redirects, item):
    wiki_id, title, redirect, revision_id, wiki_links, text = item

    if redirect:
        return

    out = StringIO()

    extract_cleaned_text(wiki_id, revision_id, title, text, out)

    wiki_text = out.getvalue()

    spotted_data = {}
    spotted_data['bodyText'] = wiki_text
    spotted_data['title'] = title
    spotted_data['spot'] = {}
    spotted_data['spot']['bodyText'] = find_spots_in_text(wiki_text, title, wiki_links, freebase_map, redirects)

    spotted_data_json = json.dumps(spotted_data)

    output.write(spotted_data_json)
    output.write("\n")


def find_spots_in_text(text, title, anchors, freebase_map, redirects):
    all_spots = []

    # Find out the title's entity.
    title_fb_id = get_freebase_id(freebase_map, redirects, title)
    title_length = len(title.split(" "))

    surface_2_spots = {}

    # Add title entity in the surface search.
    surface_2_spots[title_length] = {}
    surface_2_spots[title_length][title] = (title, title_fb_id)

    anchor_lengths = set()
    anchor_lengths.add(title_length)

    for link, span in anchors:
        anchor = link.anchor
        target = link.link

        fb_id = get_freebase_id(freebase_map, redirects, target)

        length = len(anchor.split(" "))
        anchor_lengths.add(length)

        if length not in surface_2_spots:
            surface_2_spots[length] = {}

        surface_2_spots[length][anchor] = (target, fb_id)

    tokens = text.split()

    for begin in range(len(tokens)):
        for l in anchor_lengths:
            end = begin + l

            spot_map = surface_2_spots[l]
            if end <= len(tokens):
                window_tokens = tokens[begin:end]
                window_text = " ".join(window_tokens)

                if window_text in spot_map:
                    target, fb_id = spot_map[window_text]

                    spot = {}
                    spot['loc'] = [begin, end]
                    spot['surface'] = window_text
                    spot['entity'] = {'wiki': target, 'freebase': fb_id}

                    all_spots.append(spot)

    return all_spots


def process_anchor_spot(q: Queue, v: Value, lock: Lock, output, wiki_2_fb, redirects):
    while True:
        item = q.get()
        if item is None:
            break

        write_anchor_spotted(output, wiki_2_fb, redirects, item)

        with lock:
            v.value += 1
            sys.stdout.write("\r[%s] Processed %d documents." % (datetime.datetime.now().time(), v.value))
            sys.stdout.flush()


def parse_as_spots(dump_file, output_path, redirects, wiki_2_fb={}, num_cpu=5):
    logging.info("Starting parsing the dump as spots.")
    print("Start parsing the dump as spots.")

    dump = mwxml.Dump.from_file(mwtypes.files.reader(dump_file))

    output = open(os.path.join(output_path, "wikipedia.json"), 'w')

    sl = SurfaceLinkMap()

    v = Value('i', 0)
    lock = Lock()
    q = Queue(maxsize=num_cpu)
    p = Pool(num_cpu, initializer=process_anchor_spot, initargs=(q, v, lock, output, wiki_2_fb, redirects))

    count = 100
    for element in page_parser.parse(dump, True):
        q.put(element)
        # count -= 1
        # if count == 0:
        #     break

    print("\nParsing done.")
    p.close()

    output.close()

    print("All done.")


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


def extract_cleaned_text(id, revid, title, text, out):
    Extractor(id, revid, title, text.split("\n")).extract(out)


def get_freebase_id(freebase_map: Dict, redirects, wiki_title):
    key = wiki_title.replace(" ", "_")

    if key in redirects:
        key = redirects[key]

    try:
        return freebase_map[key]
    except KeyError:
        return None


def read_wiki_fb_mapping(mapping_file):
    wiki_2_fb = {}
    with open(os.path.join(mapping_file)) as mapping:
        for line in mapping:
            fb_id, wikipage_name = line.strip().split("\t")[0:2]
            formatted_wiki_name = wikipage_name.replace(" ", "_")
            wiki_2_fb[formatted_wiki_name] = fb_id
    return wiki_2_fb


def get_redirects(args):
    output_path = args['--output-path']
    redirect_path = args['--redirect-path']

    from linker.data import data_utils

    logging.info("Loading redirect pages.")
    redirects = data_utils.run_or_load(os.path.join(output_path, "redirects.pickle"), data_utils.load_redirects,
                                       redirect_path)
    logging.info("Done")

    return redirects


def load_wiki_freebase(args):
    free_base_mapping = args['--freebase-map']

    print("Loading Wikipedia to Freebase.")
    wiki_2_fb = read_wiki_fb_mapping(free_base_mapping)
    print("Done.")

    return wiki_2_fb


def main():
    args = docopt.docopt(__doc__, argv=None)
    dump_file = args['<dump-file>']
    output_path = args['--output-path']
    sys.path.append("../../projects/KnowledgeIR")

    redirects = get_redirects(args)
    wiki_2_fb = load_wiki_freebase(args)

    parse_as_spots(dump_file, output_path, redirects, wiki_2_fb=wiki_2_fb, num_cpu=8)
