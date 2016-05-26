# MediaWiki Links

Extracts links from MediaWiki with a focus on Wikipedia.

This library provides a command-line utility for extracting wikilinks from
MediaWiki XML database dumps. 


    $ mwlinks -h

    This script provides access to a set of utilities for extracting links
    in Wikipedia.

    Utilities:
    * extract -- All wikilinks from articles

    Usage:
        mwlinks (-h | --help)
        mwlinks <utility> [-h | --help]

    Options:
        -h | --help  Shows this documentation
        <utility>    The name of the utility to run

# License and Credits
This tools is released under the MIT license.

This library is based on [wikidump](/youtux/wikidump) by Alessio Bogon (youtux)
and Cristian Consonni (CristianCantoro), also released under the MIT license.
