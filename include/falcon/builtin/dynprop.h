/*
   FALCON - The Falcon Programming Language.
   FILE: dynprop.h

   Falcon core module -- Dynamic property handlers (set, get, has).
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Thu, 07 Mar 2013 03:52:57 +0100

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CORE_DYNPROP_H
#define FALCON_CORE_DYNPROP_H

#include <falcon/pseudofunc.h>
#include <falcon/fassert.h>
#include <falcon/pstep.h>

namespace Falcon {
namespace Ext {

/*#
   @function get
   @brief Gets a property in an object
   @param item an item of any kind
   @param property The name of the property that is to be fetched
   @return the property value, if found
   @raise AccessError if the property is not found

   The returned value might be a method, if the property points
   to a function and the target class decides the function is a method
   of the @b item.
*/

/*#
   @method get BOM
   @brief Gets a property in an object
   @param property The name of the property that is to be fetched
   @return the property value, if found
   @raise AccessError if the property is not found

   The returned value might be a method, if the property points
   to a function and the target class decides the function is a method
   of the @b item.

   @see get
*/

class FALCON_DYN_CLASS Get: public PseudoFunction
{
public:
   Get();
   virtual ~Get();
   virtual void invoke( VMContext* vm, int32 nParams );

private:
   
   class FALCON_DYN_CLASS Invoke: public PStep
   {
   public:
      Invoke() { apply = apply_; }
      virtual ~Invoke() {}
      static void apply_( const PStep* ps, VMContext* vm );

   };

   Invoke m_invoke;
};



/*#
   @function set
   @brief Set a property in an object
   @param item an item of any kind
   @param property The name of the property that is to be set
   @param value The new value for the property
   @return the property value, if found
   @raise AccessError if the property cannot be set.

   Some classes, as the prototypes, accept new unexisting
   properties; other don't.
*/

/*#
   @method set BOM
   @brief Set a property in an object
   @param property The name of the property that is to be set
   @param value The new value for the property
   @return the property value, if found
   @raise AccessError if the property cannot be set.

   Some classes, as the prototypes, accept new non existing
   properties; other don't.

   @see get
   @see set
*/

class FALCON_DYN_CLASS Set: public PseudoFunction
{
public:
   Set();
   virtual ~Set();
   virtual void invoke( VMContext* vm, int32 nParams );

private:

   class FALCON_DYN_CLASS Invoke: public PStep
   {
   public:
      Invoke() { apply = apply_; }
      virtual ~Invoke() {}
      static void apply_( const PStep* ps, VMContext* vm );

   };

   Invoke m_invoke;
};



/*#
   @function has
   @brief Verifies if an object has a given property
   @param item an item of any kind.
   @param property The name of the property to be found.
   @return True if the property is present, false otherwise.

*/

/*#
   @method has BOM
   @brief Verifies if an object has a given property
   @param property The name of the property to be found.
   @return True if the property is present, false otherwise.

   @see get
   @see set
*/

class FALCON_DYN_CLASS Has: public PseudoFunction
{
public:
   Has();
   virtual ~Has();
   virtual void invoke( VMContext* vm, int32 nParams );

private:

   class FALCON_DYN_CLASS Invoke: public PStep
   {
   public:
      Invoke() { apply = apply_; }
      virtual ~Invoke() {}
      static void apply_( const PStep* ps, VMContext* vm );

   };

   Invoke m_invoke;
};



/*#
   @function properties
   @brief Returns a dictionary with all the public properties of the given object.
   @param item an item of any kind.
   @return A dictionary containing all the public properties.
*/

/*#
   @method properties BOM
   @brief Returns a dictionary with all the public properties of the given object.
   @return A dictionary containing all the public properties.

   @see get
   @see set
*/

class FALCON_DYN_CLASS Properties: public PseudoFunction
{
public:
   Properties();
   virtual ~Properties();
   virtual void invoke( VMContext* vm, int32 nParams );

private:

   class FALCON_DYN_CLASS Invoke: public PStep
   {
   public:
      Invoke() { apply = apply_; }
      virtual ~Invoke() {}
      static void apply_( const PStep* ps, VMContext* vm );

   };

   Invoke m_invoke;
};


}
}

#endif

/* end of dynprop.h */
