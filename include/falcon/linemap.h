/*
   FALCON - The Falcon Programming Language.
   FILE: linemap.h

   Line map with module serialization support.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: dom ago 21 2005

   -------------------------------------------------------------------
   (C) Copyright 2004: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

/** \file
   Line map with module serialization support -header file-.
*/

#ifndef flc_linemap_H
#define flc_linemap_H

#include <falcon/genericmap.h>
#include <falcon/types.h>

namespace Falcon {

class Stream;

/** Line information map.
   The aim of this class is to map a certain code position with a line number in
   the source file, so to allow debugging of the modules.
   (int32,int32)
*/
class FALCON_DYN_CLASS LineMap: public Map
{
public:
   LineMap();
   ~LineMap() {}
   
   void addEntry( uint32 pcounter, uint32 line ) { insert( &pcounter, &line ); }
   bool save( Stream *out ) const;
   bool load( Stream *in );
};

}

#endif

/* end of linemap.h */
