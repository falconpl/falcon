/*
   FALCON - The Falcon Programming Language.
   FILE: classpseudodict.h

   Standard language dictionary object handler
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: 

   -------------------------------------------------------------------
   (C) Copyright 2013: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_CLASSPSEUDODICT_H_
#define _FALCON_CLASSPSEUDODICT_H_

#include <falcon/setup.h>
#include <falcon/classes/classdict.h>

#include <falcon/pstep.h>

namespace Falcon
{

/**
 Class handling a dictionary as an item in a falcon script.
 */

class FALCON_DYN_CLASS ClassPseudoDict: public ClassDict
{
public:

   ClassPseudoDict();
   virtual ~ClassPseudoDict();

   //=============================================================   
   virtual void describe( void* instance, String& target, int maxDepth = 3, int maxLength = 60 ) const;
   virtual void restore( VMContext* ctx, DataReader* stream ) const;
   virtual bool hasProperty( void* instance, const String& prop ) const;

   //=============================================================

   virtual void op_setProperty( VMContext* ctx, void* self, const String& prop) const;
   virtual void op_getProperty( VMContext* ctx, void* self, const String& prop) const;
   virtual void op_toString( VMContext* ctx, void* self ) const;
};

}

#endif 

/* end of classpseudodict.h */
