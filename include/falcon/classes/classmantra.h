/*
   FALCON - The Falcon Programming Language.
   FILE: classmantra.h

   Handler for generic mantra entities.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 26 Feb 2012 00:30:54 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSMANTRA_H_
#define _FALCON_CLASSMANTRA_H_

#include <falcon/setup.h>
#include <falcon/class.h>
#include <falcon/function.h>

namespace Falcon
{

   /*# @class Mantra
    *  @brief Base class for functions and classes.
    *  @prop attributes Attributes associated with this mantra, in a dictionary.
    *  @prop name internal Name of this mantra.
    *  @prop location Full location (module, class, name) of this mantra.
    */
class FALCON_DYN_CLASS ClassMantra: public Class
{
public:

   ClassMantra();
   virtual ~ClassMantra();

   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void* createInstance() const;
   
   virtual void describe( void* instance, String& target, int, int ) const;
   virtual void gcMarkInstance( void* self, uint32 mark ) const;
   virtual bool gcCheckInstance( void* self, uint32 mark ) const;
   
   virtual void enumerateProperties( void* instance, Class::PropertyEnumerator& cb ) const;
   virtual void enumeratePV( void* instance, Class::PVEnumerator& cb ) const;
   virtual bool hasProperty( void* instance, const String& prop ) const;

   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const;
   virtual void restore( VMContext* ctx, DataReader* stream) const;
   // mantras have no flattening.
   
   //=============================================================

   virtual void op_isTrue( VMContext* ctx, void* self ) const;
   virtual void op_toString( VMContext* ctx, void* self ) const;

   virtual void op_getProperty( VMContext* ctx, void* instance, const String& prop) const;

protected:
   ClassMantra( const String& name, int64 type );

   /*#
    * @method getAttribute Mantra
    * @brief Gets the desired attribute, if it exists.
    * @param name The name of the required attribute
    * @return value of the require attribute
    * @raise Access error if the attribute is not found
    *
    */
   class FALCON_DYN_CLASS GetAttributeMethod: public Function {
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
   class FALCON_DYN_CLASS SetAttributeMethod: public Function {
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

/* end of classmantra.h */
