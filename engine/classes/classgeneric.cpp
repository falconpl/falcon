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
#include <falcon/genericitem.h>

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
   delete static_cast<GenericItem*>(self);
}

void* ClassGeneric::clone( void* self ) const
{
   return static_cast<GenericItem*>(self)->clone();
}

void* ClassGeneric::createInstance() const
{
   return 0;
}


void ClassGeneric::describe( void* self, String& target, int, int maxlen ) const
{
   static_cast<GenericItem*>(self)->describe(target);
   if( maxlen > 0 && target.length() > (unsigned) maxlen )
   {
      target.size( maxlen * target.manipulator()->charSize() );
   }
}


void ClassGeneric::gcMark( void* self, uint32 mark ) const
{
   GenericItem* gi = static_cast<GenericItem*>(self);
   gi->gcMark( mark );
}


bool ClassGeneric::gcCheck( void* self, uint32 mark ) const
{
   GenericItem* gi = static_cast<GenericItem*>(self);
   return gi->gcCheck( mark );
}


GenericItem::~GenericItem()
{
}

}

/* end of classgeneric.h */
