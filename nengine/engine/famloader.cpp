/*
   FALCON - The Falcon Programming Language.
   FILE: famloader.cpp

   Precompiled module deserializer.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Mon, 01 Aug 2011 16:07:56 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/famloader.cpp"

#include <falcon/famloader.h>
#include <falcon/datareader.h>

namespace Falcon
{

FAMLoader::FAMLoader()
{}

FAMLoader::~FAMLoader()
{}
   
Module* FAMLoader::load( DataReader* , const String&, const String&  )
{
   return 0;
}

}

/* end of famloader.cpp */
