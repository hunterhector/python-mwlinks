# coding=utf-8
import datetime
import os
import sys

from . import data_utils
from .nif_parser import NIFParser

# if sys.version_info[0] == 2:
#     from .wiki_sql_linker import WbItemsPerSite

freebase_prefix = "http://rdf.freebase.com/ns/"
wikidata_prefix = "http://www.wikidata.org/entity/"
dbpeida_prefix = "http://dbpedia.org/resource/"


class FreebaseWikiMapper:
    def __init__(self, mapper_dir):
        self.fb_wiki_mapping_file = "fb_wiki_mapping.tsv"
        self.mapper_dir = mapper_dir

        if not os.path.exists(mapper_dir):
            os.makedirs(mapper_dir)

    def create_mapping_dbpedia(self, fb_wiki_mapping_path):
        mapping_file = os.path.join(self.mapper_dir, self.fb_wiki_mapping_file)

        if os.path.exists(mapping_file):
            print("Mapping file exists, not overwriting")
            return

        seen = set()

        with open(mapping_file, 'w') as out:
            count = 0
            for statements in NIFParser(fb_wiki_mapping_path):
                for s, v, o in statements:
                    if str(v) == 'http://www.w3.org/2002/07/owl#sameAs':
                        fb_id = data_utils.canonical_freebase_id((str(o).replace(freebase_prefix, "")))
                        wikipage_name = s.toPython().replace(dbpeida_prefix, "")

                        if wikipage_name not in seen:
                            out.write("%s\t%s\n" % (fb_id, wikipage_name))
                            count += 1
                            seen.add(wikipage_name)
                            sys.stdout.write("\r[%s] found %d pairs." % (datetime.datetime.now().time(), count))
            print("\nTotally %s mappings created." % count)

    def create_mapping_wiki_data_sql(self, fb_wd_mapping_path, wb_database_name, db_user_id, db_passwd):
        if sys.version_info[0] == 3:
            print("Error: MySQL not implemented for Python 3.")
            exit(1)

        wb_db = WbItemsPerSite(db_user_id, db_passwd, wb_database_name)

        mapping_file = os.path.join(self.mapper_dir, self.fb_wiki_mapping_file)

        if os.path.exists(mapping_file):
            print("Mapping file exists, not overwriting")
            return

        with open(mapping_file, 'w') as out:
            count = 0
            missed = 0
            for statements in NIFParser(fb_wd_mapping_path):
                for s, v, o in statements:
                    if str(v) == 'http://www.w3.org/2002/07/owl#sameAs':
                        fb_id = data_utils.canonical_freebase_id(str(s).replace(freebase_prefix, ""))
                        wd_id = str(o).replace(wikidata_prefix, "")
                        wikipage_id = wb_db.page_query(wd_id, "enwiki")

                        if wikipage_id:
                            out.write("%s\t%s\t%s\n" % (fb_id, wikipage_id, wd_id))
                            count += 1
                            sys.stdout.write("\r[%s] found %d pairs." % (datetime.datetime.now().time(), count))
                        else:
                            missed += 1

            print("\nTotally %s mappings created, %d freebase items are not mapped." % (count, missed))

    def read_wiki_fb_mapping(self):
        wiki_2_fb = {}
        with open(os.path.join(self.mapper_dir, self.fb_wiki_mapping_file)) as mapping:
            for line in mapping:
                fb_id, wikipage_name = line.strip().split("\t")[0:2]
                formatted_wiki_name = wikipage_name.replace(" ", "_")
                wiki_2_fb[formatted_wiki_name] = fb_id
        return wiki_2_fb


def main():
    # /media/hdd/hdd0/data/Freebase/fb2w.nt
    fb_wd_mapping = sys.argv[1]
    mapper_dir = sys.argv[2]

    mapper = FreebaseWikiMapper(mapper_dir)
    # mapper.create_mapping_wiki_data_sql(fb_wd_mapping, "wikidatawiki_wb_items_per_site", "hector", "hector")
    mapper.create_mapping_dbpedia(fb_wd_mapping)
    # wiki_2_fb = mapper.read_wiki_fb_mapping()
    #
    # print(wiki_2_fb["Mount_Everest"])
    # print(wiki_2_fb["Khasi–Khmuic_languages"])


if __name__ == '__main__':
    main()
