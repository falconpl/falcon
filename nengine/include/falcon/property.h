/*
   FALCON - The Falcon Programming Language.
   FILE: property.h

   Abstract class property.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 20 Feb 2011 16:41:23 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_PROPERTY_H
#define FALCON_PROPERTY_H

#include <falcon/setup.h>
#include <falcon/types.h>
#include <falcon/string.h>


namespace Falcon {

class Property
{
public:
   Property( const String& name );
   virtual ~Property();

   const String& name() const { return m_name; }

   /** Set the value of this property.
      The default method in the bases class raises a read-only property error.
   */
   virtual void set( void* data, Item& target );

   /** Reads the value of this property.
      The default method in the bases class raises a write-only property error.
   */
   virtual void get( void* data, Item& target );

private:
   String m_name;
};

}

#endif

/* end of property.h */