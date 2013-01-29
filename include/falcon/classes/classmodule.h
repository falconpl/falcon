/*
   FALCON - The Falcon Programming Language.
   FILE: classmodule.h

   Module object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 22 Feb 2012 19:50:45 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSMODULE_H_
#define _FALCON_CLASSMODULE_H_

#include <falcon/setup.h>
#include <falcon/class.h>
#include <falcon/function.h>

namespace Falcon
{

/*#
 * @class Module
 * @brief Reflection of Falcon module.
 *
 * @prop name The logical name of the module.
 * @prop uri A string containing the full URI where this module
 *       is located. Can be empty if the module wasn't created
 *       from a serialized resource.
 */

class FALCON_DYN_CLASS ClassModule: public Class
{
public:

   ClassModule();
   virtual ~ClassModule();

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void* createInstance() const;
   
   virtual void enumerateProperties( void*, Class::PropertyEnumerator& cb ) const;
   virtual void enumeratePV( void* instance, Class::PVEnumerator& cb ) const;
   virtual bool hasProperty( void*, const String& prop ) const;

   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const;
   virtual void restore( VMContext* ctx, DataReader* stream ) const;
   virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;

   virtual void describe( void* instance, String& target, int maxDepth = 3, int maxLength = 60 ) const;

   virtual void gcMarkInstance( void* instance, uint32 mark ) const;
   virtual bool gcCheckInstance( void* instance, uint32 mark ) const;
   //=============================================================

   virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;
   virtual void op_getProperty( VMContext* ctx, void* instance, const String& prop) const;
   
private:
   void restoreModule( Module* mod, DataReader* stream ) const;

   /*#
    * @method getAttribute Module
    * @brief Gets the desired attribute, if it exists.
    * @param name The name of the required attribute
    * @return value of the require attribute
    * @raise Access error if the attribute is not found
    *
    */
   class GetAttributeMethod: public Function {
   public:
      GetAttributeMethod();
      virtual ~GetAttributeMethod();
      void invoke( VMContext* ctx, int32 pCount = 0 );
   };

   /*#
    * @method setAttribute Mantra
    * @brief Sets or deletes the desired attribute if it exists
    * @param name The name of the required attribute
    * @optparam value The new value for the required attribute
    *
    * If @b value is not given, then the attribute is removed, if it exists.
    *
    * If @b value is given, the value is changed or created as required.
    * In this case, if the attribute doesn't exists, it is created.
    */
   class SetAttributeMethod: public Function {
   public:
      SetAttributeMethod();
      virtual ~SetAttributeMethod();
      void invoke( VMContext* ctx, int32 pCount = 0 );
   };

   mutable GetAttributeMethod m_getAttributeMethod;
   mutable SetAttributeMethod m_setAttributeMethod;
};

}

#endif

/* end of classmodule.h */
