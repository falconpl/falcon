/*
   FALCON - The Falcon Programming Language.
   FILE: cmdline/parser_ext.h

   The command line parser class
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 06 Aug 2013 15:14:19 +0200

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_EXT_PARSER_H
#define FALCON_EXT_PARSER_H

#include <falcon/setup.h>
#include <falcon/falconclass.h>

namespace Falcon{
namespace Ext {
namespace CCmdLineParser {

class Parser: public FalconClass
{
   Parser(): FalconClass("Parser") {}
   virtual ~Parser() {}
};

}}}

#endif

/* end of parser_ext.h */
