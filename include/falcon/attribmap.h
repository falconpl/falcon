/*
   FALCON - The Falcon Programming Language.
   FILE: attribmap.h

   Attribute Map - specialized string - vardef map.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sat, 11 Jul 2009 20:42:43 +0200

   -------------------------------------------------------------------
   (C) Copyright 2009: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FLC_ATTRIBMAP_H
#define FLC_ATTRIBMAP_H

#include <falcon/setup.h>
#include <falcon/genericmap.h>
#include <falcon/traits.h>


namespace Falcon {

class VarDef;
class String;
class Stream;
class Module;

/** Specialized attribute map.
  It's actually just a String -> VarDef specialized map.
*/

class FALCON_DYN_CLASS AttribMap: public Map
{
public:
   AttribMap();
   AttribMap( const AttribMap& other );
   virtual ~AttribMap();

   void insertAttrib( const String& name, VarDef* vd );
   VarDef* findAttrib( const String& name );

   bool save( const Module* mod, Stream *out ) const;
   bool load( const Module* mod, Stream *out );
};

}

#endif

/* end of attribmap.h */
