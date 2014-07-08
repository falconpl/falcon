/*
   FALCON - The Falcon Programming Language.
   FILE: stmtfastprint.h

   Syntactic tree item definitions -- Fast Print statement.
   -------------------------------------------------------------------
   Author: Giancarlo Niccolai
   Begin: Sun, 14 Aug 2011 12:41:20 +0200

   -------------------------------------------------------------------
   (C) Copyright 2011: the FALCON developers (see list in AUTHORS file)

   See LICENSE file for licensing details.
*/

#ifndef FALCON_STMTFASTPRINT_H
#define FALCON_STMTFASTPRINT_H

#include <falcon/statement.h>

namespace Falcon
{

/** Fastprint statement.
 The fastprint statement is a line beginning with ">" or ">>", printing
 everything that's on the line.

*/
class FALCON_DYN_CLASS StmtFastPrint: public Statement
{
public:
   StmtFastPrint( int line = 0, int chr = 0 );
   StmtFastPrint( const StmtFastPrint& other );
   virtual ~StmtFastPrint();

   virtual int arity() const;
   virtual TreeStep* nth( int32 n ) const;
   virtual bool setNth( int32 n, TreeStep* ts );
   virtual bool insert( int32 n, TreeStep* ts );
   virtual bool remove( int32 n );

   void add( Expression* expr );
   Expression* at( int n ) const;
   length_t size() const;

   virtual void render( TextWriter* tw, int32 depth ) const;
   virtual StmtFastPrint* clone() const { return new StmtFastPrint(*this); }
   bool isAddNL() const { return m_bAddNL; }

protected:
   /** Costructor for subclasses.
    *
    * The selector parameter is ignored, used to allow subclasses
    * to get this constructor that doesn't set a default add NL, nor
    * the handler class.
    */
   StmtFastPrint( int line, int chr, bool selector );
   class Private;
   Private* _p;

   bool m_bAddNL;
   static Mutex m_mtx;
   static void apply_( const PStep*, VMContext* ctx );
};

}

#endif

/* end of stmtfastprint.h */
