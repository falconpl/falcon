/*
   FALCON - The Falcon Programming Language.
   FILE: propitem.h

   Property class handling methods.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 20 Feb 2011 16:41:23 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_PROPMETHOD_H
#define FALCON_PROPMETHOD_H

#include <falcon/property.h>

namespace Falcon {

class Function;

/** Property returning a Falcon function.
 */
class PropMethod: public Property
{
public:
   PropItem( const String& name, Function* mth );
   virtual ~PropItem();

   /** Set the value of this property. */
   virtual void set( void* data, Item& target );

   /** Reads the value of this property. */
   virtual void get( void* data, Item& target );

private:
   Function* m_mth;
};

}

#endif

/* end of propitem.h */
