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

/** The base class of all the error class hierarcy.
 All the hierarcy expects an ErrorParam instance as the
 creationParameter for the create() method.
 
 */
class FALCON_DYN_SYM ClassError: public Class
{
public:
   virtual void dispose( void* self ) const;
   virtual void* clone( void* source ) const;
   virtual void serialize( DataWriter* stream, void* self ) const;
   virtual void* deserialize( DataReader* stream ) const;

   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;
   // TODO: overload properties.

   void op_toString( VMContext* ctx, void* self ) const;
   
protected:
   ClassError( const String& name );
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
};

}

#endif	/* FALCON_CLASSERROR_H */

/* end of classerror.h */
