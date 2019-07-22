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
    --redirect-pickle=REDIRECT_PICKLE A path that is to loaded redirects from or save into
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

from collections import defaultdict
import sys

from mwlinks.libs import page_parser
from mwlinks.libs.wikilink import Wikilink
from mwlinks.libs.common import Span
import json
import logging
import logging.config
from mwlinks.libs.WikiExtractor import Extractor
from multiprocessing import Pool, Value, Lock, Queue
from io import StringIO
import datetime
import nltk

links_to_ignore = {"File", "Category", "wikt"}


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


def process_links(wikilinks: Iterable[Tuple[Wikilink, Span]], freebase_map,
                  redirects):
    links = []
    for link, span in wikilinks:
        if not is_ignore_link(link):
            wiki_title, fb_id = get_name_and_id(link.link, redirects,
                                                freebase_map)
            links.append((link.anchor, wiki_title, fb_id, span))
    return links


def get_name_and_id(title, redirects, freebase_map):
    wiki_title = format_wiki_title(title)
    if wiki_title in redirects:
        wiki_title = redirects[wiki_title]

    if wiki_title in freebase_map:
        return wiki_title, freebase_map[wiki_title]
    else:
        return wiki_title, None


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


class WriteTextConsumer:
    def __init__(self, output_path, write_text=False, write_anchor_text=False,
                 write_both=False, num_cpu=5):
        self.outputs = {}
        self.p = None

        if write_text:
            self.outputs["origin"] = open(
                os.path.join(output_path, "origin.txt"), 'w')
        if write_anchor_text:
            self.outputs["replaced"] = open(
                os.path.join(output_path, "replaced.txt"), 'w')
        if write_both:
            self.outputs["both"] = open(os.path.join(output_path, "both.txt"),
                                        'w')

        self.num_cpu = num_cpu

    def prepare_consumer(self, queue):
        v = Value('i', 0)
        lock = Lock()
        self.p = Pool(self.num_cpu, initializer=self.write_text_consumer,
                      initargs=(queue, v, lock, self.outputs))

    def write_text_consumer(self, q: Queue, v: Value, lock: Lock):
        while True:
            item = q.get()
            if item is None:
                break

            self.write_as_text(item)

            with lock:
                v.value += 1
                sys.stdout.write("\r[%s] Processed %d documents." % (
                datetime.datetime.now().time(), v.value))
                sys.stdout.flush()

    def write_as_text(self, item):
        wiki_id, title, title_fb_id, redirect, revision_id, links, text = item

        sorted_links = sorted([l for l in links if l[2] is not None],
                              key=itemgetter(2))

        if redirect:
            return

        for key, out_file in self.outputs.items():
            out = StringIO()
            if key == "origin":
                clean_wiki_text(sorted_links, wiki_id, revision_id, title, text,
                                out, use_plain_text=True)
                out_file.write(out.getvalue())

            if key == "replaced":
                clean_wiki_text(sorted_links, wiki_id, revision_id, title, text,
                                out, use_link=True)
                out_file.write(out.getvalue())

            if key == "both":
                clean_wiki_text(sorted_links, wiki_id, revision_id, title, text,
                                out, True, True)
                both_text = out.getvalue()
                out_file.write(mix_up(both_text))

    def close(self):
        self.p.close()
        for key, out in self.outputs.items():
            out.close()


class WikiSpotConsumer:
    def __init__(self, output_path, num_cpu=5):
        self.output = open(os.path.join(output_path, "wikipedia.json"), 'w')
        self.num_cpu = num_cpu
        self.p = None

    def prepare_consumer(self, queue: Queue):
        v = Value('i', 0)
        lock = Lock()
        self.p = Pool(self.num_cpu, initializer=self.spot_consumer,
                      initargs=(queue, v, lock))

    def spot_consumer(self, queue: Queue, v: Value, lock: Lock):
        while True:
            item = queue.get()
            if item is None:
                break

            self.write_anchor_spotted(item)

            with lock:
                v.value += 1
                sys.stdout.write("\r[%s] Processed %d documents." % (
                datetime.datetime.now().time(), v.value))
                sys.stdout.flush()

    def write_anchor_spotted(self, item):
        wiki_id, title, title_fb_id, redirect, revision_id, sorted_links, text = item

        # # Make the link as one single token (without spaces), including the anchor and wiki name.
        # named_sorted_links = [
        #     (anchor,
        #      "inline_wikilink++" + wiki_name + "++" + (fb_id if fb_id else "NONE") + "++" + "__".join(anchor.split()),
        #      fb_id, span) for (anchor, wiki_name, fb_id, span) in sorted_links]

        placeholder_links = []
        for anchor, wiki_name, fb_id, span in sorted_links:
            placeholder_links.append((anchor, "HectorReplacement", fb_id, span))

        placeholder_text = format_anchor(placeholder_links, text)

        if redirect:
            return

        out = StringIO()

        # Write the Wiki text to the StringIO object.
        write_cleaned_text(wiki_id, revision_id, title, placeholder_text, out)
        wiki_text = out.getvalue()
        title_text, body_text = wiki_text.split("\n\n", 1)

        title_tokens = nltk.word_tokenize(title_text)
        wiki_body_tokens = nltk.word_tokenize(body_text)
        title_entity = format_wiki_title(title)

        title = ' '.join(title_tokens)

        spotted_data = {}

        body_spots = self.find_spots_in_text(wiki_body_tokens, title,
                                             title_fb_id, sorted_links)

        spotted_data['bodyText'] = self.recover_text(wiki_body_tokens,
                                                     sorted_links)
        spotted_data['title'] = " ".join(title_tokens)
        spotted_data['spot'] = {}
        spotted_data['spot']['bodyText'] = body_spots
        spotted_data['spot']['title'] = [
            {"loc": [0, len(title_tokens)], "surface": [title],
             "entities": [{"wiki": title_entity,
                           "id": title_fb_id if title_fb_id else "None"}]}
        ]

        spotted_data_json = json.dumps(spotted_data)

        # print(spotted_data_json)

        self.output.write(spotted_data_json + "\n")

    @staticmethod
    def recover_text(wiki_text_tokens, sorted_links):
        split = ""
        recovered_text = ""

        link_index = 0

        for token in wiki_text_tokens:
            recovered_text += split
            if token == "HectorReplacement":
                anchor, wiki_name, fb_id, span = sorted_links[link_index]
                # link = sorted_links[link_index]
                actual_anchor_tokens = nltk.word_tokenize(anchor)
                recovered_text += " ".join(actual_anchor_tokens)
                link_index += 1
            else:
                recovered_text += token
            split = " "
        return recovered_text

    @staticmethod
    def find_spots_in_text(tokens, title, title_fb_id, sorted_links):
        """
        Find in the Wikipedia text additional spotting by using the surface->target links in the same page.
        :param tokens: 
        :param text: The actual page text.
        :param title: The title of the page, not normalized.
        :param sorted_links: The Wiki links in this page, sorted. With mapping to Freebase, and title normalized.
        :param freebase_map: A mapping from Wikipedia to Freebase.
        :param redirects: Redirect of pages.
        :return: 
        """
        all_spots = []

        # Store all possible surface forms, organized by token length.
        surface_2_spots = {}

        # Store possible surface form length.
        surface_form_lengths = set()

        for anchor, wiki_name, fb_id, span in reversed(sorted_links):
            length = len(nltk.word_tokenize(anchor))
            surface_form_lengths.add(length)

            if length not in surface_2_spots:
                surface_2_spots[length] = {}

            surface_2_spots[length][anchor] = {}

            # For the other surface anchors, we count the counts pointing to different entities.
            try:
                surface_2_spots[length][anchor][(wiki_name, fb_id)] += 1
                # print("%s:%d", wiki_name, surface_2_spots[length][anchor][(wiki_name, fb_id)])
            except KeyError:
                surface_2_spots[length][anchor][(wiki_name, fb_id)] = 1
                # print("First seen: %s:%d", wiki_name, surface_2_spots[length][anchor][(wiki_name, fb_id)])

        surface_2_sorted_spots = {}

        for length, anchor_map in surface_2_spots.items():
            surface_2_sorted_spots[length] = {}
            for anchor, entity_counts in anchor_map.items():
                tokenized_anchor = " ".join(nltk.word_tokenize(anchor))
                sorted_entity_counts = sorted(entity_counts.items(),
                                              key=itemgetter(1), reverse=True)
                surface_2_sorted_spots[length][
                    tokenized_anchor] = sorted_entity_counts

        # Find out the title's entity.
        title_entity = format_wiki_title(title)
        title_tokens = nltk.word_tokenize(title)
        title_length = len(title_tokens)
        tokenized_title = " ".join(title_tokens)

        if title_length not in surface_2_sorted_spots:
            surface_2_sorted_spots[title_length] = {}

        if tokenized_title not in surface_2_sorted_spots[title_length]:
            surface_2_sorted_spots[title_length][tokenized_title] = []

        surface_form_lengths.add(title_length)
        surface_2_sorted_spots[title_length][tokenized_title].insert(0, (
        (title_entity, title_fb_id), 0))

        offset = 0
        link_index = 0

        anchor_memory = {}

        for begin in range(len(tokens)):
            begin_token = tokens[begin]
            if begin_token == "HectorReplacement":
                # print(begin_token)
                anchor, wiki_name, fb_id, span = sorted_links[link_index]
                actual_anchor_tokens = nltk.word_tokenize(anchor)
                anchor_length = len(actual_anchor_tokens)
                end = begin + anchor_length

                spot = {'loc': [begin + offset, end + offset],
                        'surface': " ".join(actual_anchor_tokens),
                        'entities': [{'wiki': wiki_name, 'id': fb_id}]}
                all_spots.append(spot)

                # Remember which wiki name this anchor points to, and the token index where it happens.
                anchor_memory[anchor] = (wiki_name, fb_id, begin)

                # This token actual may contain multiple tokens, we add the token differences to the offset to
                # compensate this.
                offset += anchor_length - 1
                link_index += 1
            else:
                for l in surface_form_lengths:
                    end = begin + l

                    anchor_map = surface_2_sorted_spots[l]

                    if end <= len(tokens):
                        window_tokens = tokens[begin:end]
                        window_text = " ".join(window_tokens)

                        if window_text in anchor_map:
                            sorted_wiki_targets = anchor_map[window_text]
                            # Take from the top of the target.
                            (wiki_name, fb_id), pref = sorted_wiki_targets[0]

                            # If preference is 0, that means this is the title entity, top priority.
                            if not pref == 0:
                                # If we found it in the anchor memory, then use it to annotate the current one.
                                if window_text in anchor_memory:
                                    previous_wiki_name, previous_fb_id, happen_index = \
                                    anchor_memory[window_text]
                                    if begin - happen_index <= 50:
                                        # If something is found in the anchor memory, we use that as the annotation.
                                        wiki_name = previous_wiki_name
                                        fb_id = previous_fb_id
                                    else:
                                        # If it is expired, remove it.
                                        anchor_memory.pop(window_text, None)

                            spot = {'loc': [begin + offset, end + offset],
                                    'surface': window_text,
                                    'entities': [{'wiki': wiki_name,
                                                  'id': fb_id if fb_id else "None"}]}
                            all_spots.append(spot)

        return all_spots

    def close(self):
        self.p.close()
        self.output.close()


def parse_dump_producer(dump_file, output_path, consumer, redirects={},
                        wiki_2_fb={}, link_prob=False, num_cpu=5):
    logging.info("Starting dump parsing.")
    print("Start parsing the dump.")

    dump = mwxml.Dump.from_file(mwtypes.files.reader(dump_file))

    failed_pages = set()
    found_pages = set()

    q = Queue(maxsize=num_cpu)

    sl = SurfaceLinkMap()

    # The consumer will start looping and wait for items.
    consumer.prepare_consumer(q)

    for wiki_id, title, redirect, revision_id, wiki_links, text in page_parser.parse(
            dump, True):
        if redirect:
            # Ignoring redirect pages.
            continue

        title_wiki_name, title_fb_id = get_name_and_id(title, redirects,
                                                       wiki_2_fb)
        links = process_links(wiki_links, wiki_2_fb, redirects)

        for anchor, wiki_title, fb_id, span in links:
            if fb_id is not None:
                found_pages.add(wiki_title)
            else:
                failed_pages.add(wiki_title)

        element = (
        wiki_id, title, title_fb_id, redirect, revision_id, links, text)
        q.put(element)

        if link_prob:
            for anchor, wiki_name, fb_id, span in links:
                if fb_id is not None:
                    sl.add_surface_link(anchor, fb_id)

    print("\nParsing done.")
    consumer.close()

    # Logging additional debug information.
    out_wiki_not_found = open(os.path.join(output_path, "wiki_not_found.txt"),
                              encoding='UTF-8', mode='w')
    out_wiki_not_found.write("%d links found, %d links missed.\n" % (
    len(found_pages), len(failed_pages)))
    out_wiki_not_found.write("================================\n")
    for p in failed_pages:
        out_wiki_not_found.write(p)
        out_wiki_not_found.write("\n")
    out_wiki_not_found.close()

    print("%d links found, %d links missed." % (
    len(found_pages), len(failed_pages)))

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


def clean_wiki_text(links, pageid, revid, title, text, out,
                    use_plain_text=False, use_link=False):
    if use_plain_text:
        write_cleaned_text(pageid, revid, title, format_anchor(links, text),
                           out)
    if use_link:
        write_cleaned_text(pageid, revid, title,
                           format_anchor(links, text, True), out)


def format_anchor(links, text, use_freebase=False):
    for anchor, wiki_name, fb_id, span in reversed(links):
        if use_freebase:
            if fb_id is not None:
                text = replace_by_index(text, span.begin, span.end, fb_id)
        else:
            text = replace_by_index(text, span.begin, span.end, wiki_name)

    return text


def is_ignore_link(link: Wikilink):
    link_parts = link.link.split(":")
    if len(link_parts) > 1:
        link_type = link_parts[0]
        if link_type in links_to_ignore:
            return True

    return False


def format_wiki_title(link):
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
    redirect_pickle = args['--redirect-pickle']

    if redirect_pickle is None:
        redirect_pickle = os.path.join(output_path, "redirects.pickle")

    from linker.data import data_utils
    logging.info("Loading redirect pages.")
    print("Loading redirect pages.")
    redirects = data_utils.run_or_load(redirect_pickle,
                                       data_utils.load_redirects, redirect_path)
    print("Done.")
    logging.info("Done")

    if free_base_mapping is not None:
        logging.info("Loading Wikipedia to Freebase.")
        print("Loading Wikipedia to Freebase.")
        wiki_2_fb = read_wiki_fb_mapping(free_base_mapping)
        print("Done.")
        logging.info("Done")
    else:
        logging.info("No Freebase mapping read.")
        wiki_2_fb = defaultdict(lambda a: a)

    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    root.handlers = []
    handler = logging.FileHandler(os.path.join(output_path, "dump.log"))
    handler.setFormatter(
        logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    root.addHandler(handler)

    # consumer = WriteTextConsumer(output_path, write_text, write_anchor_text, write_both, 5)
    consumer = WikiSpotConsumer(output_path, 5)

    parse_dump_producer(dump_file, output_path, consumer, link_prob=link_prob,
                        wiki_2_fb=wiki_2_fb,
                        redirects=redirects, num_cpu=6)
