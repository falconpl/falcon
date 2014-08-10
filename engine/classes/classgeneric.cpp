/*
   FALCON - The Falcon Programming Language.
   FILE: classgeneric.h

   Generic object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 12 Aug 2011 19:37:30 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#undef SRC
#define SRC "engine/classes/classgeneric.cpp"

#include <falcon/classes/classgeneric.h>
#include <falcon/genericdata.h>
#include <falcon/engine.h>
#include <falcon/stdhandlers.h>

namespace Falcon
{

ClassGeneric::ClassGeneric():
   Class("Generic")
{
}

ClassGeneric::~ClassGeneric()
{   
}

void ClassGeneric::dispose( void* self ) const
{
   delete static_cast<GenericData*>(self);
}

void* ClassGeneric::clone( void* self ) const
{
   return static_cast<GenericData*>(self)->clone();
}

void* ClassGeneric::createInstance() const
{
   return 0;
}


void ClassGeneric::describe( void* self, String& target, int, int maxlen ) const
{
   static_cast<GenericData*>(self)->describe(target);
   if( maxlen > 0 && target.length() > (unsigned) maxlen )
   {
      target.size( maxlen * target.manipulator()->charSize() );
   }
}


void ClassGeneric::gcMarkInstance( void* self, uint32 mark ) const
{
   GenericData* gi = static_cast<GenericData*>(self);
   gi->gcMark( mark );
}


bool ClassGeneric::gcCheckInstance( void* self, uint32 mark ) const
{
   GenericData* gi = static_cast<GenericData*>(self);
   return gi->gcCheck( mark );
}


const Class* GenericData::handler()
{
   static const Class* gen = Engine::handlers()->genericClass();
   return gen;
}

}

/* end of classgeneric.h */
