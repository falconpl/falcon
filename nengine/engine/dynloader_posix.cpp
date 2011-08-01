/*
   FALCON - The Falcon Programming Language.
   FILE: dynloader_posix.cpp

   Native shared object based module loader -- POSIX standard ext.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 01 Aug 2011 16:07:56 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/dynloader_posix.cpp"

#include <falcon/dynloader.h>
#include <falcon/string.h>

namespace Falcon
{

const String& DynLoader::sysExtension()
{
   static String ext = ".so";
   return ext;
}

}

/* end of dynloader_posix.cpp */
