/*
   FALCON - The Falcon Programming Language.
   FILE: classerror.h

   Class for storing error in scripts.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Fri, 04 Feb 2011 18:39:36 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/


#ifndef FALCON_CLASSERROR_H
#define FALCON_CLASSERROR_H

#include <falcon/setup.h>
#include <falcon/error.h>
#include <falcon/class.h>

namespace Falcon {

class Stream;
class ErrorParam;

/*# The base class of all the error class hierarchy.
 
 Error classes have special meaning both on the script side and on the
 C++/embedding side.

 As any other scriptable entity, error classes are organized so that
 there is an "instance" which is what is directly visible at inner/C++ level,
 and a Class handler, which is the manipulator used by the VM and the script
 to know the object.

 This is the base class of all the handlers that should handle certain errors.

 @see The Falcon::Error C++ class description for an in-depth of the internals
 of this class.


   @prop code
   @prop description
   @prop extra

   @prop mantra
   @prop module
   @prop path
   @prop signature
   @prop line
   @prop chr

   @prop heading
   @prop trace
   @prop errors
   @prop raised
 */
class FALCON_DYN_SYM ClassError: public Class
{
public:

   ClassError( const String& name, bool registerInEngine = true );


   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void* createInstance() const;

   virtual Error* createError( const ErrorParam& params ) const;

   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;
   
   bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;
   void op_toString( VMContext* ctx, void* self ) const;
   
protected:
   virtual ~ClassError();

   /** Turns the standard error parameters sent by scripts into an ErrorParam.
    \param ctx The context where the parameters reside.
    \param pcount Number of parameters in the stack.
    \param params The ErrorParam initializer that will be filled.
    \param bThrow If true, will Throw an error in case of wrong parameters.
    \return false in case of parametetr errors if bThrow is not true.
    
    This method analyzes the parameters passed in the stack at error class or
    subclass invocation by scripts, and then uses those parameters to initialize
    an ErrorParam instance that, in turn, will initialize the C++ exception
    related to the subclass.
    
    In short, this gets the first parameters of the script call, if given, 
    and turns then into:
    - The error code.
    - The error description.
    - The error extra message.
    
    All the parameters are optional and could be nil if skipped.
    
    Also, it fills the line, module and symbol parameters by analyzing the
    current status of the context.
    
    \throws ParamError on invalid parameters.
    */
   bool invokeParams( VMContext* ctx, int pcount, ErrorParam& params, bool bThrow = true ) const;

   bool m_bRegistered;
};

}

#endif	/* FALCON_CLASSERROR_H */

/* end of classerror.h */
