/*
   FALCON - The Falcon Programming Language.
   FILE: stmtglobal.h

   Syntactic tree item definitions -- global directive/statement
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Wed, 02 May 2012 21:18:16 +0200

   -------------------------------------------------------------------
   (C) Copyright 2012: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef _FALCON_GLOBAL_H_
#define _FALCON_GLOBAL_H_

#include <falcon/statement.h>

namespace Falcon {

class Symbol;
class Expression;
class VMContext;
class DataWriter;
class DataReader;

/** Handler for the global statement.

   Global statement is partially a directive, in the sense that it instructs
   the compiler to add a global declaration in the forming module,
   and partially a statement, as it makes the associated symbols
   visible as globals from the point where the declaration is performed
   on.

 */
class FALCON_DYN_CLASS StmtGlobal: public Statement
{
public:

   StmtGlobal( int32 line = 0, int32 chr = 0 );
   StmtGlobal( const StmtGlobal& other );
   virtual ~StmtGlobal();

   virtual void render( TextWriter* tw, int32 depth ) const;
   virtual StmtGlobal* clone() const { return new StmtGlobal(*this); }

   /**
    * Adds a symbol to the global definition using its name.
    */
   bool addSymbol( const String& name );

   bool addSymbol( const Symbol* var );

   void store( DataWriter* dw ) const;
   void restore( DataReader* dw );

private:
   class Private;
   StmtGlobal::Private* _p;
   
   bool alreadyAdded( const String& name ) const;

   static void apply_( const PStep*, VMContext* ctx );
};
   
}

#endif

/* end of stmtglobal.h */
