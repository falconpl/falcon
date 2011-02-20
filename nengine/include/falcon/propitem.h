/*
   FALCON - The Falcon Programming Language.
   FILE: propitem.h

   Property class handling standard items.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 20 Feb 2011 16:41:23 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_PROPITEM_H
#define FALCON_PROPITEM_H

#include <falcon/property.h>

namespace Falcon {

/** Property read or set in a simple item array.
 */
class PropItem: public Property
{
public:
   PropItem( const String& name, int32 id );
   virtual ~PropItem();

   /** Set the value of this property. */
   virtual void set( void* data, Item& target );

   /** Reads the value of this property. */
   virtual void get( void* data, Item& target );

private:
   int32 m_id;
};

}

#endif

/* end of propitem.h */
