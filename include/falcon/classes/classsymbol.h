/*
   FALCON - The Falcon Programming Language.
   FILE: classsymbol.h

   Symbol class handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 27 Dec 2011 21:39:56 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSSYMBOL_H_
#define _FALCON_CLASSSYMBOL_H_

#include <falcon/setup.h>
#include <falcon/class.h>

namespace Falcon {

class Symbol;

/** Handler class for symbols.
 
 The class can host any symbol; when used to create new symbols in the code,
 it will generate DynSymbols.
 
 */
class ClassSymbol: public Class // TreeStep
{
public:
   ClassSymbol();
   virtual ~ClassSymbol(); 

   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   
   virtual void op_create( VMContext* ctx, int32 pcount ) const;
   
   virtual void enumerateProperties( void* instance, PropertyEnumerator& cb ) const;
   virtual void enumeratePV( void* instance, PVEnumerator& cb ) const;
   virtual bool hasProperty( void* instance, const String& prop ) const;
   
   virtual void op_getProperty( VMContext* ctx, void* instance, const String& prop) const;
   virtual void op_setProperty( VMContext* ctx, void* instance, const String& prop) const;
};

}

#endif 

/* end of classsymbol.h */
