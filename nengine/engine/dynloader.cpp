/*
   FALCON - The Falcon Programming Language.
   FILE: dynloader.cpp

   Native shared object based module loader.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 01 Aug 2011 16:07:56 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/dynloader.cpp"

#include <falcon/dynloader.h>

namespace Falcon
{

DynLoader::DynLoader()
{}

DynLoader::~DynLoader()
{}
   
Module* DynLoader::load( const String& , const String&  )
{
   return 0;
}

}

/* end of dynloader.cpp */
