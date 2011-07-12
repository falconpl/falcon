/*
   FALCON - The Falcon Programming Language.
   FILE: stmt_init.h

   Stastatement specialized in initialization of instances.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Tue, 12 Jul 2011 13:25:18 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_STMT_INIT_H
#define FALCON_STMT_INIT_H

#include <falcon/statement.h>

namespace Falcon
{

class Inheritance;

/** Stastatement specialized in initialization of FalconClass instances.

 This statement performs a direct call to a constructor of a base
 class stored in an Inheritance instance. The instance must be a FalconClass
 instance (this is checked by an assert in debug).

 The information in the Inheritance instance are used also to generate
 the needed parameters.

 The statement uses the self entity in the current call frame to repeat it
 to the constructor of the classes indicated in the inheritance entry.

 \note This statement doesn't own the Instance entry; it just references it.
 */
class FALCON_DYN_CLASS StmtInit: public Statement
{
public:
   
   StmtInit( Inheritance* inh, int32 line=0, int32 chr=0 );
   virtual ~StmtInit();

   void describe( String& tgt ) const;
   inline String describe() const { return PStep::describe(); }
   
   static void apply_( const PStep*, VMContext* ctx );

private:
   Inheritance* m_inheritance;
};

}

#endif

/* end of stmt_init.h */
