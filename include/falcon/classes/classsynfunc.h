/*
   FALCON - The Falcon Programming Language.
   FILE: classsynfunc.h

   Syntree based function object handler.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 26 Feb 2012 01:10:59 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_CLASSSYNFUNC_H_
#define FALCON_CLASSSYNFUNC_H_

#include <falcon/setup.h>
#include <falcon/classes/classfunction.h>


namespace Falcon
{

/**
Syntree based function object handler.
 */

class FALCON_DYN_CLASS ClassSynFunc: public ClassFunction
{
public:
   ClassSynFunc();
   virtual ~ClassSynFunc();
   
   virtual Class* getParent( const String& name ) const;
   virtual bool isDerivedFrom( const Class* cls ) const;
   virtual void enumerateParents( ClassEnumerator& cb ) const;
   virtual void* getParentData( Class* parent, void* data ) const;
   
   void describe( void* instance, String& target, int depth, int ) const;

   virtual void store( VMContext* ctx, DataWriter* stream, void* instance ) const;
   virtual void restore( VMContext* ctx, DataReader* stream ) const;
   virtual void flatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;
   virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;

   virtual void enumerateProperties( void* instance, Class::PropertyEnumerator& cb ) const;
   virtual void enumeratePV( void* instance, Class::PVEnumerator& cb ) const;
   virtual bool hasProperty( void* instance, const String& prop ) const;

   void op_getProperty( VMContext* ctx, void* instance, const String& prop) const;
};

}

#endif /* FUNCTION_H_ */

/* end of classsynfunc.h */
