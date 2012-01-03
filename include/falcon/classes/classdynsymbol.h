/*
   FALCON - The Falcon Programming Language.
   FILE: classdynsymbol.h

   Symbol class handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 03 Jan 2012 22:08:13 +0100

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSDYNSYMBOL_H_
#define _FALCON_CLASSDYNSYMBOL_H_

#include <falcon/setup.h>
#include <falcon/class.h>

namespace Falcon {

class DynSymbol;

/** Handler class for dynamic symbols.
 
 The class can host dynamic symbol; when used to create new symbols in the code,
 it will generate DynSymbols.
 
 */
class ClassDynSymbol: public Class
{
public:
   ClassDynSymbol();
   virtual ~ClassDynSymbol(); 

   virtual void describe( void* instance, String& target, int depth = 3, int maxlen = 60 ) const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;
   
   virtual void op_create( VMContext* ctx, int32 pcount ) const;
   
   virtual void enumerateProperties( void* instance, PropertyEnumerator& cb ) const;
   virtual void enumeratePV( void* instance, PVEnumerator& cb ) const;
   virtual bool hasProperty( void* instance, const String& prop ) const;
   
   virtual void op_getProperty( VMContext* ctx, void* instance, const String& prop) const;
   virtual void op_setProperty( VMContext* ctx, void* instance, const String& prop) const;

   virtual void op_eval( VMContext* ctx, void* instance ) const;
};

}

#endif 

/* end of classdynsymbol.h */
