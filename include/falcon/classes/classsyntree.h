/*
   FALCON - The Falcon Programming Language.
   FILE: classsyntree.h

   Base class for SynTree PStep handlers.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 27 Dec 2011 21:39:56 +0100

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSSYNTREE_H_
#define _FALCON_CLASSSYNTREE_H_

#include <falcon/setup.h>
#include <falcon/derivedfrom.h>

namespace Falcon {

class ClassTreeStep;
class ClassSymbol;

/** Handler class for Statement class.
 
 This handler manages the base statements as they are reflected into scripts,
 and has also support to handle the vast majority of serialization processes.
 
 */
class ClassSynTree: public DerivedFrom // TreeStep
{
public:
   ClassSynTree( ClassTreeStep* parent, ClassSymbol* sym );
   virtual ~ClassSynTree(); 

   virtual void* createInstance() const;
   virtual void dispose( void* instance ) const;
   virtual void* clone( void* instance ) const;

   virtual void restore( VMContext* ctx, DataReader* stream ) const;
   virtual void unflatten( VMContext* ctx, ItemArray& subItems, void* instance ) const;

   virtual void enumerateProperties( void* instance, PropertyEnumerator& cb ) const;
   virtual void enumeratePV( void* instance, PVEnumerator& cb ) const;
   virtual bool hasProperty( void* instance, const String& prop ) const;
   virtual void op_getProperty( VMContext* ctx, void* instance, const String& prop) const;
   virtual void op_setProperty( VMContext* ctx, void* instance, const String& prop) const;

   virtual void op_call( VMContext* ctx, int pcount, void* instance) const;

private:
   ClassSymbol* m_classSymbol;
};

}

#endif 

/* end of classsyntree.h */
