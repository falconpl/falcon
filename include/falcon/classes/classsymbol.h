/*
   FALCON - The Falcon Programming Language.
   FILE: classsymbol.h

   Symbol class handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 03 Jan 2012 22:08:13 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSSYMBOL_H_
#define _FALCON_CLASSSYMBOL_H_

#include <falcon/setup.h>
#include <falcon/class.h>

namespace Falcon {

/** Handler class for dynamic symbols.
 
 The class can host dynamic symbol; when used to create new symbols in the code,
 it will generate DynSymbols.
 
 */
class ClassSymbol: public Class
{
public:
   ClassSymbol();
   virtual ~ClassSymbol(); 

   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   virtual void* createInstance() const;
   
   virtual bool op_init( VMContext* ctx, void* instance, int32 pcount ) const;
   
   virtual void enumerateProperties( void* instance, PropertyEnumerator& cb ) const;
   virtual void enumeratePV( void* instance, PVEnumerator& cb ) const;
   virtual bool hasProperty( void* instance, const String& prop ) const;
   
   virtual void op_getProperty( VMContext* ctx, void* instance, const String& prop) const;
   virtual void op_setProperty( VMContext* ctx, void* instance, const String& prop) const;

   virtual void op_call( VMContext* ctx, int pcount, void* instance ) const;
   
   void store( VMContext*, DataWriter* stream, void* instance ) const;
   void restore( VMContext*, DataReader* stream ) const;
};

}

#endif 

/* end of classsymbol.h */
