/*
   FALCON - The Falcon Programming Language.
   FILE: confparser_fm.cpp

   Configuration parser module
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 08 Aug 2013 18:28:02 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#define SRC "modules/native/feathers/confparser_fm.cpp"
#include "confparser_fm.h"
#include "confparser_ext.h"


/*#
   @module confparser Configuration file parser
   @ingroup feathers
   @brief Advanced configuration file parser (with sections and key categorization support).

   The ConfParser module is meant as a simple but powerful interface to .INI
   like configuration files, with some extra features that allows to bring a bit
   forward the traditional configuration file model and to circumvent some
   arbitrary limitation that makes using human readable configuration files a bit
   difficult in some complex configuration contexts.

   ConfParser module also maintains comments and tries respect the layout of the
   original INI file, so that it stays familiar for the user after a modify that
   has been caused by the Falcon module.

   @section confparser_ini_fformat Ini file format

   The ConfParser module parses INI files in the following format:
   @code
   ; Comment lines
   ; Comment lines...

   Key = value
   category.key = value

   [section_name]
   Key = value
   category.key = value
   Comments may be started either with ";" or "#" characters, and the colon ":" may be used instead of "=". So the above file may be rewritten in the more UNIX style:
   # Comment lines
   # Comment lines...

   Key: value
   category.key: value

   [section_name]
   Key: value
   category.key: value
   @endcode

   Values may be enclosed in quotes; in this case, Falcon escape sequences are
   correctly parsed. As comments starting with ";" or "#" may be placed
   also after a value, if a value itself contains one of those characters it
   should be enclosed by quotes, or part of the value will be considered a comment.

   For example:

   @code
   Key: "A complex value containing\nescapes # and comment" ; real comment
   @endcode

   @subsection confparser_multiple_values Multiple values

   Although it would be possible to put arbitrary lists into strings to save
   them on configuration files, and expand them in the program when reading them
   back, it is possible to store array of values in configuration files by
   declaring multiple times the same key.

   For example:
   @code
   Key = value1
   Key = value2
   Key = value3
   @endcode

   This will result in the three values to be returned in an array when the
   value of "Key" is asked.

   @subsection confparser_key_cat Key categories

   Keys can be categorized; tree-like or even recursive key groups can be
   represented with dots separating the key hierarchy elements.
   In example, the configuration of a complex program can be saved like that:

   @code
   ...
   UserPref.files.MaxSize = 100
   UserPref.files.DefaultDir = "/home/$user"
   ...
   ...
   UserPref.cache.MaxSize = 250
   UserPref.cache.path = "/var/cache/prog.cache"
   ...
   @endcode
   This lessen the need for traditional ini file "sections". Support for
   sections is provided both for backward compatibility with a well known file
   structure and because it still can be useful where it is known that one
   hierarchical level is enough for the configuration needs of an application.

   Sections are portion of ini files separated by the rest of it by an header in
   square brackets; the keys that are defined from the beginning of file to the
   first section heading, if present, are considered to belong to the "main"
   section. The main section can't be directly addressed, as it has not a name. All
   the methods accessing or manipulating keys have an extra optional parameter that
   can address a specific section by it's name. If the parameter is not specified,
   or if the parameter has nil value, then the methods will operate on the main
   section.
*/


namespace Falcon {
namespace Feathers {

ModuleConfparser::ModuleConfparser():
         Module("confparser", true)
{
   Falcon::Class *c_cparser = Falcon::Ext::confparser_create();
   addMantra(c_cparser);
}

ModuleConfparser::~ModuleConfparser()
{

}

}
}


/* end of confparser_fm.cpp */


