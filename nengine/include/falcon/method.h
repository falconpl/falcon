/*
   FALCON - The Falcon Programming Language.
   FILE: method.h

   Encapsulation for user-defined Methods in ClassUser.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 07 Aug 2011 20:18:45 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_METHOD_H_
#define _FALCON_METHOD_H_

#include <falcon/setup.h>
#include <falcon/string.h>
#include <falcon/property.h>
#include <falcon/function.h>

namespace Falcon {

class ClassUser;
class Item;

class Method: public Function
{
public:
   Method( ClassUser* cls, const String& name, Module* mod = 0 );
   virtual ~Method() {}

private:
   class MethodProp: public Property
   {
   public:
      MethodProp( Method* mth, ClassUser* owner, const String& name );
      virtual ~MethodProp() {}
      
      virtual void set( void* instance, const Item& value );
      virtual void get( void* instance, Item& target );
   private:
      Method* m_mth;
   };
   
   MethodProp m_prop;
};

}

#endif	/* _FALCON_METHOD_H_ */

/* end of method.h */
