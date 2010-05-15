/*
   FALCON - The Falcon Programming Language.
   FILE: dbi_handle.cpp

   Database Interface - Main handle driver
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 16 May 2010 00:09:13 +0200

   -------------------------------------------------------------------
   (C) Copyright 2010: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#include <falcon/dbi_handle.h>

namespace Falcon
{

DBIHandle::DBIHandle()
{
}


DBIHandle::~DBIHandle()
{
}

void DBIHandle::gcMark( uint32 )
{

}

FalconData* DBIHandle::clone() const
{
   return 0;
}

}

/* end of dbi_handle.h */
