/*
   FALCON - The Falcon Programming Language.
   FILE: socket.cpp

   The configuration parser module.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 2006-05-09 15:50

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   The confparser module - main file.
*/
#include <falcon/module.h>
#include "confparser_ext.h"
#include "confparser_mod.h"

#include "version.h"
/*#
   @module feather_configparser The Configuration Parser Module

   The ConfParser module is meant as a simple but powerful interface to .INI
   like configuration files, with some extra features that allows to bring a bit
   forward the traditional configuration file model and to circumvent some
   arbitrary limitation that makes using human readable configuration files a bit
   difficult in some complex configuration contexts.

   ConfParser module also maintains comments and tries respect the layout of the
   original INI file, so that it stays familiar for the user after a modify that
   has been caused by the Falcon module.

   @section Ini file format

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

   In example:

   @code
   Key: "A complex value containing\nescapes # and comment" ; real comment
   @endcode

   @subsection Multiple values

   Although it would be possible to put arbitrary lists into strings to save
   them on configuration files, and expand them in the program when reading them
   back, it is possible to store array of values in configuration files by
   declaring multiple times the same key.

   @code
   In example:
   Key = value1
   Key = value2
   Key = value3
   @endcode

   This will result in the three values to be returned in an array when the
   value of "Key" is asked.

   @subsection Key categories

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
FALCON_MODULE_DECL( const Falcon::EngineData &data )
{
   #define FALCON_DECLARE_MODULE self

   // setup DLL engine common data
   data.set();

   Falcon::Module *self = new Falcon::Module();
   self->name( "confparser" );
   self->language( "en_US" );
   self->engineVersion( FALCON_VERSION_NUM );
   self->version( VERSION_MAJOR, VERSION_MINOR, VERSION_REVISION );

   //====================================
   // Message setting
   #include "confparser_st.h"

   // private class socket.
   Falcon::Symbol *c_cparser = self->addClass( "ConfParser", Falcon::Ext::ConfParser_init );
   self->addClassMethod( c_cparser, "read", Falcon::Ext::ConfParser_read );
   self->addClassMethod( c_cparser, "write", Falcon::Ext::ConfParser_write );
   self->addClassMethod( c_cparser, "get", Falcon::Ext::ConfParser_get );
   self->addClassMethod( c_cparser, "getOne", Falcon::Ext::ConfParser_getOne );
   self->addClassMethod( c_cparser, "getMultiple", Falcon::Ext::ConfParser_getMultiple );
   self->addClassMethod( c_cparser, "getSections", Falcon::Ext::ConfParser_getSections );
   self->addClassMethod( c_cparser, "getKeys", Falcon::Ext::ConfParser_getKeys );
   self->addClassMethod( c_cparser, "getCategoryKeys", Falcon::Ext::ConfParser_getCategoryKeys );
   self->addClassMethod( c_cparser, "getCategory", Falcon::Ext::ConfParser_getCategory );
   self->addClassMethod( c_cparser, "removeCategory", Falcon::Ext::ConfParser_removeCategory );
   self->addClassMethod( c_cparser, "getDictionary", Falcon::Ext::ConfParser_getDictionary );
   self->addClassMethod( c_cparser, "add", Falcon::Ext::ConfParser_add );
   self->addClassMethod( c_cparser, "set", Falcon::Ext::ConfParser_set );
   self->addClassMethod( c_cparser, "remove", Falcon::Ext::ConfParser_remove );
   self->addClassMethod( c_cparser, "addSection", Falcon::Ext::ConfParser_addSection );
   self->addClassMethod( c_cparser, "removeSection", Falcon::Ext::ConfParser_removeSection );
   self->addClassMethod( c_cparser, "clearMain", Falcon::Ext::ConfParser_clearMain );
   self->addClassProperty( c_cparser, "errorLine" );
   self->addClassProperty( c_cparser, "error" );

   return self;
}

/* end of socket.cpp */
